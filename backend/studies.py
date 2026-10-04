"""Immutable study plans and a database-claimed, cancellable CPU worker."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import sklearn
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.datasets import load_digits

from backend.config import ROOT
from backend.store import ConflictError, Store, dump, now


class StudyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seeds: tuple[int, ...] = (17, 43, 89)
    pca_components: int = Field(default=24, ge=2, le=48)
    c_candidates: tuple[float, ...] = (0.1, 1.0, 10.0)
    noise_sigma: float = Field(default=3.0, ge=0, le=8)
    occlusion_size: int = Field(default=2, ge=1, le=4)

    @model_validator(mode="after")
    def bounded(self):
        if not 3 <= len(self.seeds) <= 5 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("需要 3–5 个不重复种子")
        if any(s < 0 or s > 100000 for s in self.seeds):
            raise ValueError("种子需位于 0–100000")
        if (not 1 <= len(self.c_candidates) <= 5
                or len(set(self.c_candidates)) != len(self.c_candidates)
                or any(not np.isfinite(c) or not 0 < c <= 100 for c in self.c_candidates)):
            raise ValueError("需要 1–5 个不同的 C，范围 (0,100]")
        return self


class StudyID(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan_id: str = Field(pattern=r"^[0-9a-f]{64}$")


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def fingerprint() -> dict:
    dataset = load_digits()
    return {
        "dataset_sha256": sha(dataset.data.tobytes() + dataset.target.tobytes()),
        "code_sha256": sha(b"".join((ROOT / path).read_bytes()
                                  for path in ("backend/studies.py", "backend/study_worker.py"))),
        "numpy": np.__version__, "sklearn": sklearn.__version__,
    }


def atomic_json(path: Path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(canonical(value))
    os.replace(temporary, path)


class StudyService:
    def __init__(self, store: Store, directory: Path):
        self.store, self.directory = store, directory
        directory.mkdir(parents=True, exist_ok=True)
        with store.connection() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS studies (
                    id TEXT PRIMARY KEY, plan TEXT NOT NULL, status TEXT NOT NULL,
                    result TEXT, error TEXT, created_at TEXT NOT NULL, finished_at TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_cpu_study
                    ON studies(status) WHERE status='running';
            """)

    def plan(self, config: StudyConfig) -> dict:
        plan = {
            "schema_version": 1, "dataset": "sklearn_digits",
            "methods": ["pixels_svm", "pca_svm"], "config": config.model_dump(mode="json"),
            "test_fraction": 0.2, "validation_fraction_of_remaining": 0.25,
            "selection": "C maximizing clean validation accuracy; ties choose smallest C; "
                         "scaler and PCA fitted on train only; no refit on validation",
            "conditions": ["clean", "noise", "occlusion"], "bootstrap_samples": 2000,
            "fingerprint": fingerprint(),
            "limitations": "Small digits study, not a large-paper reproduction. "
                          "Seed test sets overlap. Bootstrap is conditional on each fixed "
                          "split/trained pair, not a population generalization guarantee.",
        }
        plan_id = sha(canonical(plan))
        with self.store.connection() as con:
            con.execute("INSERT OR IGNORE INTO studies(id,plan,status,created_at) VALUES(?,?,?,?)",
                        (plan_id, dump(plan), "planned", now()))
        return self.read(plan_id)

    def read(self, plan_id: str) -> dict:
        StudyID(plan_id=plan_id)
        row = self.store.one("SELECT * FROM studies WHERE id=?", (plan_id,))
        if row is None:
            raise KeyError("研究方案不存在")
        row["plan"] = json.loads(row["plan"])
        row["result"] = json.loads(row["result"]) if row["result"] else None
        if sha(canonical(row["plan"])) != plan_id:
            raise ValueError("研究方案内容被修改，哈希不匹配")
        if row["result"]:
            for name, digest in row["result"]["evidence_files"].items():
                target = self.directory / plan_id / name
                if not target.is_file() or sha(target.read_bytes()) != digest:
                    raise ValueError("实验工件缺失或内容被修改")
        return row

    def list(self) -> list[dict]:
        rows = self.store.rows("SELECT id FROM studies ORDER BY created_at DESC LIMIT 30")
        return [self.read(row["id"]) for row in rows]

    def recover_interrupted(self):
        with self.store.connection() as con:
            con.execute("UPDATE studies SET status='interrupted',error=?,finished_at=? "
                        "WHERE status='running'", ("服务重启；可以重新执行同一方案", now()))

    async def run(self, plan_id: str) -> dict:
        row = self.read(plan_id)
        if row["status"] == "completed":
            return {**row, "reused": True}
        if row["plan"]["fingerprint"] != await asyncio.to_thread(fingerprint):
            raise ValueError("代码、依赖或数据指纹已变化，请创建新方案")
        try:
            with self.store.connection() as con:
                con.execute("BEGIN IMMEDIATE")
                current = con.execute("SELECT status FROM studies WHERE id=?",
                                      (plan_id,)).fetchone()["status"]
                if current == "completed":
                    return {**self.read(plan_id), "reused": True}
                if current == "running":
                    raise ConflictError("该研究正在运行；请稍后读取结果")
                con.execute("UPDATE studies SET status='running',error=NULL,finished_at=NULL "
                            "WHERE id=?", (plan_id,))
        except sqlite3.IntegrityError as exc:
            raise ConflictError("另一项 CPU 研究正在运行；请稍后重试") from exc
        directory = self.directory / plan_id
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "plan.json", row["plan"])
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "backend.study_worker", str(directory),
                cwd=ROOT, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            async with asyncio.timeout(180):
                stdout, stderr = await process.communicate()
            if process.returncode:
                raise RuntimeError("研究进程失败：" + stderr.decode(errors="replace")[-800:])
            result = json.loads((directory / "result.json").read_text())
            result["evidence_files"] = {
                path.name: sha(path.read_bytes()) for path in sorted(directory.iterdir())
                if path.suffix in {".json", ".npz"} and path.name != "result.json"
            }
            result["worker_result_sha256"] = sha((directory / "result.json").read_bytes())
            with self.store.connection() as con:
                con.execute("UPDATE studies SET status='completed',result=?,finished_at=? WHERE id=?",
                            (dump(result), now(), plan_id))
        except BaseException as exc:
            if process and process.returncode is None:
                process.kill()
                await process.wait()
            status = "interrupted" if isinstance(exc, asyncio.CancelledError) else "failed"
            with self.store.connection() as con:
                con.execute("UPDATE studies SET status=?,error=?,finished_at=? WHERE id=?",
                            (status, type(exc).__name__, now(), plan_id))
            raise
        return {**self.read(plan_id), "reused": False}

    def evidence_path(self, plan_id: str, name: str) -> Path:
        row = self.read(plan_id)
        if not row["result"] or name not in row["result"]["evidence_files"]:
            raise KeyError("实验工件不存在")
        return self.directory / plan_id / name
