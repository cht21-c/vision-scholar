"""Small, transaction-oriented SQLite store. One connection per operation."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def now() -> str:
    return datetime.now(UTC).isoformat()


def uid() -> str:
    return uuid4().hex[:16]


def dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class ConflictError(ValueError):
    pass


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as con:
            con.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS papers (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, authors TEXT NOT NULL DEFAULT '',
                    year INTEGER, source_url TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE,
                    file_name TEXT NOT NULL, page_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pages (
                    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                    number INTEGER NOT NULL, text TEXT NOT NULL,
                    PRIMARY KEY(paper_id, number)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, paper_id TEXT NOT NULL REFERENCES papers(id)
                    ON DELETE CASCADE, page INTEGER NOT NULL, ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL, section TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS chunks_paper ON chunks(paper_id, page);
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL, history TEXT NOT NULL DEFAULT '[]'
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                    prompt TEXT NOT NULL, paper_ids TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL, mode TEXT NOT NULL, role TEXT NOT NULL DEFAULT '',
                    answer TEXT NOT NULL DEFAULT '', citations TEXT NOT NULL DEFAULT '[]',
                    verification TEXT NOT NULL DEFAULT '{}', usage TEXT NOT NULL DEFAULT '{}',
                    error TEXT, created_at TEXT NOT NULL, finished_at TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_run
                    ON runs(session_id) WHERE status IN ('queued', 'running');
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    type TEXT NOT NULL, data TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_run ON events(run_id, seq);
                CREATE TABLE IF NOT EXISTS notes (
                    id TEXT PRIMARY KEY, session_id TEXT REFERENCES sessions(id),
                    title TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, config TEXT NOT NULL,
                    result TEXT NOT NULL, created_at TEXT NOT NULL
                );
            """)
            # Additive migration preserves all pre-comparison sessions and runs.
            for table in ("sessions", "runs"):
                columns = {row["name"] for row in con.execute(f"PRAGMA table_info({table})")}
                for name, default in (("harness", "openharness"), ("retriever", "lsa")):
                    if name not in columns:
                        con.execute(f"ALTER TABLE {table} ADD COLUMN {name} TEXT "
                                    f"NOT NULL DEFAULT '{default}'")
            columns = {row["name"] for row in con.execute("PRAGMA table_info(sessions)")}
            if "decision_memory" not in columns:
                con.execute("ALTER TABLE sessions ADD COLUMN decision_memory TEXT "
                            "NOT NULL DEFAULT '[]'")

    @contextmanager
    def connection(self):
        con = sqlite3.connect(self.path, timeout=15)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=15000")
        try:
            with con:
                yield con
        finally:
            con.close()

    def rows(self, sql: str, params=()) -> list[dict]:
        with self.connection() as con:
            return [dict(row) for row in con.execute(sql, params).fetchall()]

    def one(self, sql: str, params=()) -> dict | None:
        result = self.rows(sql, params)
        return result[0] if result else None

    def papers(self) -> list[dict]:
        return self.rows("SELECT * FROM papers ORDER BY created_at DESC, id")

    def paper(self, paper_id: str) -> dict:
        value = self.one("SELECT * FROM papers WHERE id=?", (paper_id,))
        if value is None:
            raise KeyError("论文不存在")
        return value

    def chunks(self, paper_ids: list[str] | None = None) -> list[dict]:
        query = "SELECT c.*, p.title FROM chunks c JOIN papers p ON p.id=c.paper_id"
        if paper_ids:
            marks = ",".join("?" for _ in paper_ids)
            return self.rows(query + f" WHERE c.paper_id IN ({marks}) ORDER BY c.id", paper_ids)
        return self.rows(query + " ORDER BY c.id")

    def add_paper(self, paper: dict, pages: list[str], chunks: list[dict]) -> dict:
        with self.connection() as con:
            existing = con.execute("SELECT * FROM papers WHERE sha256=?", (paper["sha256"],)).fetchone()
            if existing:
                return dict(existing)
            keys = list(paper)
            con.execute(
                f"INSERT INTO papers ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})",
                [paper[k] for k in keys],
            )
            con.executemany(
                "INSERT INTO pages(paper_id,number,text) VALUES(?,?,?)",
                [(paper["id"], i + 1, text) for i, text in enumerate(pages)],
            )
            con.executemany(
                "INSERT INTO chunks(id,paper_id,page,ordinal,text,section) VALUES(?,?,?,?,?,?)",
                [(c["id"], paper["id"], c["page"], c["ordinal"], c["text"], c["section"])
                 for c in chunks],
            )
        return paper

    def create_session(self, title: str = "新的研究") -> dict:
        value = {"id": uid(), "title": title, "created_at": now(), "updated_at": now()}
        with self.connection() as con:
            con.execute(
                "INSERT INTO sessions(id,title,created_at,updated_at) VALUES(?,?,?,?)",
                tuple(value.values()),
            )
        return value

    def sessions(self) -> list[dict]:
        return self.rows(
            "SELECT id,title,created_at,updated_at,harness,retriever "
            "FROM sessions ORDER BY updated_at DESC"
        )

    def session(self, session_id: str) -> dict:
        value = self.one("SELECT * FROM sessions WHERE id=?", (session_id,))
        if value is None:
            raise KeyError("会话不存在")
        value["history"] = json.loads(value["history"])
        value["decision_memory"] = json.loads(value["decision_memory"])
        return value

    def create_run(self, session_id: str, prompt: str, paper_ids: list[str], mode: str,
                   harness: str = "openharness", retriever: str = "lsa") -> dict:
        self.session(session_id)
        if harness not in {"legacy", "openharness", "upgraded", "research"}:
            raise ValueError("未知运行版本")
        if retriever not in {"lsa", "neural"}:
            raise ValueError("未知检索器")
        for paper_id in paper_ids:
            self.paper(paper_id)
        run_id = uid()
        try:
            with self.connection() as con:
                con.execute("BEGIN IMMEDIATE")
                session = con.execute("SELECT * FROM sessions WHERE id=?",
                                      (session_id,)).fetchone()
                has_runs = con.execute("SELECT 1 FROM runs WHERE session_id=? LIMIT 1",
                                       (session_id,)).fetchone()
                if has_runs and (session["harness"], session["retriever"]) != (harness, retriever):
                    raise ConflictError("切换运行版本或检索器需要新建研究，以保持历史一致")
                con.execute(
                    """INSERT INTO runs(id,session_id,prompt,paper_ids,status,mode,created_at,
                    harness,retriever) VALUES(?,?,?,?,?,?,?,?,?)""",
                    (run_id, session_id, prompt, dump(paper_ids), "queued", mode, now(),
                     harness, retriever),
                )
                con.execute(
                    """UPDATE sessions SET title=CASE WHEN title='新的研究' THEN ? ELSE title END,
                    updated_at=?,harness=?,retriever=? WHERE id=?""",
                    (prompt[:36], now(), harness, retriever, session_id),
                )
        except sqlite3.IntegrityError as exc:
            raise ConflictError("该会话已有任务正在运行") from exc
        return self.run(run_id)

    @staticmethod
    def decode_run(row: dict) -> dict:
        for key in ("paper_ids", "citations", "verification", "usage"):
            row[key] = json.loads(row[key])
        return row

    def run(self, run_id: str) -> dict:
        row = self.one("SELECT * FROM runs WHERE id=?", (run_id,))
        if row is None:
            raise KeyError("任务不存在")
        return self.decode_run(row)

    def runs(self, session_id: str) -> list[dict]:
        return [self.decode_run(row) for row in self.rows(
            "SELECT * FROM runs WHERE session_id=? ORDER BY created_at", (session_id,)
        )]

    def update_run(self, run_id: str, **values) -> None:
        allowed = {"status", "role", "answer", "citations", "verification", "usage",
                   "error", "finished_at"}
        if not values or not values.keys() <= allowed:
            raise ValueError("Invalid run update")
        serialized = [dump(v) if k in {"citations", "verification", "usage"} else v
                      for k, v in values.items()]
        with self.connection() as con:
            con.execute(
                f"UPDATE runs SET {','.join(k + '=?' for k in values)} WHERE id=?",
                [*serialized, run_id],
            )

    def save_history(self, session_id: str, messages: list[dict], decisions=None) -> None:
        with self.connection() as con:
            con.execute("UPDATE sessions SET history=?,updated_at=? WHERE id=?",
                        (dump(messages), now(), session_id))
            if decisions is not None:
                con.execute("UPDATE sessions SET decision_memory=? WHERE id=?",
                            (dump(decisions), session_id))

    def event(self, run_id: str, event_type: str, data: dict) -> dict:
        timestamp = now()
        with self.connection() as con:
            seq = con.execute(
                "INSERT INTO events(run_id,type,data,created_at) VALUES(?,?,?,?)",
                (run_id, event_type, dump(data), timestamp),
            ).lastrowid
        return {"seq": seq, "run_id": run_id, "type": event_type, "data": data,
                "created_at": timestamp}

    def events(self, run_id: str, after: int = 0) -> list[dict]:
        rows = self.rows(
            "SELECT * FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT 500",
            (run_id, after),
        )
        for row in rows:
            row["data"] = json.loads(row["data"])
        return rows

    def recover_interrupted(self) -> int:
        rows = self.rows("SELECT id FROM runs WHERE status IN ('queued','running')")
        for row in rows:
            self.update_run(row["id"], status="interrupted", finished_at=now(),
                            error="服务重启中断了该任务；会话已保留，可以重新发送。")
            self.event(row["id"], "terminal", {"status": "interrupted"})
        return len(rows)

    def add_note(self, title: str, content: str, session_id: str | None = None,
                 idempotency_key: str | None = None) -> dict:
        if session_id:
            self.session(session_id)
        note_id = hashlib.sha256(idempotency_key.encode()).hexdigest()[:32] if idempotency_key else uid()
        value = {"id": note_id, "session_id": session_id, "title": title,
                 "content": content, "created_at": now()}
        with self.connection() as con:
            con.execute("INSERT OR IGNORE INTO notes VALUES(?,?,?,?,?)", tuple(value.values()))
            return dict(con.execute("SELECT * FROM notes WHERE id=?", (note_id,)).fetchone())

    def notes(self) -> list[dict]:
        return self.rows("SELECT * FROM notes ORDER BY created_at DESC LIMIT 100")

    def add_experiment(self, kind: str, config: dict, result: dict) -> dict:
        value = {"id": uid(), "kind": kind, "config": config, "result": result,
                 "created_at": now()}
        with self.connection() as con:
            con.execute("INSERT INTO experiments VALUES(?,?,?,?,?)",
                        (value["id"], kind, dump(config), dump(result), value["created_at"]))
        return value

    def experiments(self) -> list[dict]:
        rows = self.rows("SELECT * FROM experiments ORDER BY created_at DESC LIMIT 100")
        for row in rows:
            row["config"] = json.loads(row["config"])
            row["result"] = json.loads(row["result"])
        return rows
