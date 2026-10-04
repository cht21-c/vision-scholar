"""Paired real-model experiments; each trial has isolated notes, tools and traces."""
# ruff: noqa: E402 -- CLI bootstrap supports both direct script and module invocation.
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import re
import shutil
import sqlite3
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openharness.engine.messages import ConversationMessage, sanitize_conversation_messages
from openharness.engine.stream_events import AssistantTextDelta, AssistantTurnComplete, ErrorEvent

from backend.agent import SYSTEM_PROMPT, route_role
from backend.harnesses import build_engine
from backend.observed_client import ObservedClient
from backend.papers import PaperLibrary
from backend.retrieval import HybridIndex, verify_citations
from backend.store import Store
from backend.tools import ResearchTools
from backend.upgrades import verify_citations_with_ranges
from scripts.evaluate import REAL

OUTPUT = ROOT / "data" / "comparison"
LOG = [
    {"epoch": 1, "train_loss": 0.9, "val_loss": 1.0, "train_accuracy": 0.6, "val_accuracy": 0.55},
    {"epoch": 2, "train_loss": 0.4, "val_loss": 0.5, "train_accuracy": 0.85, "val_accuracy": 0.81},
    {"epoch": 3, "train_loss": 0.2, "val_loss": 0.65, "train_accuracy": 0.95, "val_accuracy": 0.77},
]
NORMAL = [{**c, "prompts": [c["question"]]} for c in REAL[:8]] + [
    {"id": "log", "mode": "experiment", "tools": ["analyze_log"],
     "prompts": ["调用工具分析以下合成训练日志，给出最佳准确率epoch、最佳loss epoch和"
                 "末轮泛化差距。说明这是用户提供的日志。\n" + json.dumps(LOG)]},
    {"id": "memory", "tools": ["save_note", "list_notes"],
     "prompts": ["请保存研究决策：试验标识 VX-7c91，先用digits教学基线，seed=73，PCA维度19。"
                 "不要执行实验。",
                 "调用 list_notes，核对刚才保存的试验标识、数据集、种子和PCA维度。"]},
]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def prepare_corpus(target):
    if target.exists():
        return
    store = Store(target)
    with store.connection() as con:
        con.execute("ATTACH DATABASE ? AS source", (str(ROOT / "data" / "scholar.db"),))
        for table in ("papers", "pages", "chunks"):
            con.execute(f"INSERT INTO main.{table} SELECT * FROM source.{table}")
    with sqlite3.connect(target) as con:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def paired(messages):
    expected = []
    for m in messages:
        if m["role"] == "assistant":
            if expected:
                return False
            expected = [b["id"] for b in m["content"] if b["type"] == "tool_use"]
        else:
            found = [b["tool_use_id"] for b in m["content"] if b["type"] == "tool_result"]
            if sorted(expected) != sorted(found):
                return False
            expected = []
    return not expected


def checks_for(case, turns, tools):
    all_events = [e for t in turns for e in t["events"]]
    complete_tools = [e for e in all_events if e["type"] == "ToolExecutionCompleted"]
    successful = {e["tool_name"] for e in complete_tools if not e["is_error"]}
    answer = "\n".join(t["answer"] for t in turns)
    checks = {
        "completed": all(t["answer"].strip() and not t["errors"] for t in turns),
        "required_tools": set(case.get("tools", [])) <= successful,
        "citations_valid": all(not t["verification"]["invalid"] for t in turns),
        "message_pairs": all(t["messages_paired"] for t in turns),
    }
    cited = {e.get("paper_id") for t in turns for e in t["verification"]["valid"]}
    if case.get("papers"):
        checks["paper_coverage"] = set(case["papers"]) <= cited
    if case.get("code"):
        checks["code_citation"] = any(
            e["id"].startswith("code:") for t in turns for e in t["verification"]["valid"])
    if case.get("refusal"):
        checks["uncertainty_marker"] = bool(re.search(
            r"无法确认|不能确认|无法确定|不能确定|无法.*(?:给出|回答|判断)|不足以|没有.*(?:证据|信息)",
            answer))
    if case["id"] == "digits":
        experiments = tools.store.experiments()
        exp = experiments[-1] if experiments else {}
        result = exp.get("result", {})
        checks["split_and_seed"] = (result.get("split") == {
            "train": 1077, "validation": 360, "test": 360} and result.get("seed") == 42)
        checks["experiment_id"] = bool(exp and exp["id"] in answer)
        checks["numerical_accuracy"] = bool(result.get("results")) and all(
            f"{r['test_accuracy'] * 100:.2f}" in answer for r in result.get("results", []))
    if case["id"] == "log":
        checks["log_numbers"] = ("2" in answer and ("18" in answer or "0.18" in answer))
    if case["id"] == "memory":
        checks["memory_recall"] = all(x in turns[-1]["answer"].lower()
                                      for x in ("vx-7c91", "digits", "73", "19"))
    if case.get("facts"):
        checks["facts_exact"] = all(all(t["fact_checks"].values()) for t in turns)
    return checks


async def run_trial(case, harness, repetition, settings, corpus, index, *, label="normal"):
    trial_id = f"{label}-{case['id']}-{harness}-r{repetition}"
    target = OUTPUT / "trials" / f"{trial_id}.json"
    if target.exists():
        return json.loads(target.read_text())
    directory = OUTPUT / "work" / trial_id
    directory.mkdir(parents=True, exist_ok=True)
    # An interrupted trial may have partially written side effects: restart it in isolation.
    database = directory / "scholar.db"
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(database) + suffix)
        if path.exists():
            path.unlink()
    shutil.copyfile(corpus, database)
    store = Store(database)
    library = PaperLibrary(store, ROOT / "data" / "papers")
    session = store.create_session(trial_id)
    role = route_role(case["prompts"][0], case.get("mode", "auto"))
    run = {"id": trial_id, "session_id": session["id"], "prompt": case["prompts"][0],
           "paper_ids": case.get("paper_ids", [])}
    tools = ResearchTools(store, library, index, directory, run,
                          idempotent_notes=harness == "upgraded")
    registry = tools.registry(role)
    prompt = case.get("system_prompt") or SYSTEM_PROMPT.format(
        role=role, catalog="; ".join(f"{p['id']}: {p['title']}" for p in store.papers()),
        paper_ids=run["paper_ids"], max_turns=8,
    )
    client = ObservedClient(settings["BENCH_API_KEY"], settings["BENCH_BASE_URL"],
                            thinking_disabled=True)
    started = time.monotonic()
    result = {"id": trial_id, "group": label, "case_id": case["id"], "harness": harness,
              "repetition": repetition, "created_at": datetime.now(UTC).isoformat(),
              "case": case, "system_prompt": prompt, "turns": [],
              "evaluation_basis": "项目预设工程判据；不是人评金标；语义另列AI复核"}
    try:
        async with AsyncSqliteSaver.from_conn_string(str(directory / "loop.db")) as saver:
            engine = build_engine(
                harness, client, registry, prompt, settings["BENCH_MODEL"],
                max_turns=8, checkpointer=saver, thread_id=trial_id,
                context_window=case.get("context_window", 100000),
                compact_threshold=case.get("compact_threshold", 85000),
            )
            engine.load_messages([ConversationMessage.model_validate(m)
                                  for m in case.get("history", [])])
            for question in case["prompts"]:
                tools.run["prompt"] = question
                tools.evidence = {}
                turn_start = time.monotonic()
                turn = {"prompt": question, "answer": "", "events": [], "errors": [],
                        "first_visible_seconds": None}
                async with asyncio.timeout(180):
                    async for event in engine.submit_message(question):
                        item = {"type": type(event).__name__, **asdict(event),
                                "seconds": time.monotonic() - turn_start}
                        # Pydantic objects nested in dataclasses need JSON normalization.
                        item = json.loads(json.dumps(item, ensure_ascii=False,
                                          default=lambda x: x.model_dump(mode="json")))
                        turn["events"].append(item)
                        if isinstance(event, AssistantTextDelta) and turn["first_visible_seconds"] is None:
                            turn["first_visible_seconds"] = time.monotonic() - turn_start
                        if isinstance(event, AssistantTurnComplete) and not event.message.tool_uses:
                            turn["answer"] = event.message.text
                        if isinstance(event, ErrorEvent):
                            turn["errors"].append(event.message)
                turn.update(
                    elapsed_seconds=time.monotonic() - turn_start,
                    verification=(verify_citations_with_ranges if harness == "upgraded"
                                  else verify_citations)(turn["answer"], tools.evidence, store),
                    history=[m.model_dump(mode="json") for m in engine.messages],
                    compact_metadata=copy.deepcopy(getattr(engine, "tool_metadata", {})),
                    protected_decisions=copy.deepcopy(getattr(engine, "decisions", [])),
                    evidence=copy.deepcopy(tools.evidence),
                )
                turn["messages_paired"] = paired(turn["history"])
                if case.get("facts"):
                    try:
                        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", turn["answer"].strip())
                        values = json.loads(text)
                    except ValueError:
                        values = {}
                    turn["fact_checks"] = {k: values.get(k) == v for k, v in case["facts"].items()}
                result["turns"].append(turn)
                store.save_history(session["id"], [
                    m.model_dump(mode="json") for m in sanitize_conversation_messages(engine.messages)])
            result["native_usage"] = engine.total_usage.model_dump()
            result["checks"] = checks_for(case, result["turns"], tools)
            result["passed"] = all(result["checks"].values())
    except Exception as exc:
        # Exception text never includes provider headers/body or API configuration.
        result.update(error=type(exc).__name__, passed=False)
        if "turn" in locals() and turn not in result["turns"]:
            result["incomplete_turn"] = turn
    finally:
        await client.close()
        result.update(calls=client.calls, usage=client.summary(),
                      elapsed_seconds=time.monotonic() - started)
        write_json(target, result)
    return result


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["canary", "normal", "context"], default="canary")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--harnesses", nargs="+", default=["legacy", "openharness"])
    args = parser.parse_args()
    settings = dotenv_values(ROOT / "data" / "private" / "benchmark.env")
    corpus = OUTPUT / "corpus.db"
    corpus.parent.mkdir(parents=True, exist_ok=True)
    prepare_corpus(corpus)
    index = HybridIndex(Store(corpus))
    index.search("residual mapping", limit=1)  # Warm shared retrieval outside task timings.
    if args.suite == "context":
        from scripts.context_cases import context_cases
        cases = context_cases()
    else:
        cases = [NORMAL[0], NORMAL[6], NORMAL[-1]] if args.suite == "canary" else NORMAL
    repetitions = 1 if args.suite == "canary" else args.repetitions
    jobs = []
    for repetition in range(1, repetitions + 1):
        for number, case in enumerate(cases):
            order = args.harnesses if (repetition + number) % 2 else args.harnesses[::-1]
            jobs.extend((case, h, repetition) for h in order)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(), "suite": args.suite,
        "sampling": {"temperature": 0, "thinking": "disabled"}, "max_output_tokens": 4500,
        "concurrency": args.concurrency, "repetitions": repetitions,
        "retriever": "BM25 + LSA + RRF + lexical rerank; shared and warmed",
        "model": "Endpoint identifier stored privately; actual response model in every trial",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "corpus": Store(corpus).papers(), "tasks": [(c["id"], h, r) for c, h, r in jobs],
    }
    artifact = args.suite
    if args.harnesses != ["legacy", "openharness"]:
        artifact += "-" + "-".join(args.harnesses)
    write_json(OUTPUT / f"{artifact}-manifest.json", manifest)
    slots, results, started = asyncio.Semaphore(args.concurrency), [], time.monotonic()
    async def work(job):
        async with slots:
            row = await run_trial(*job, settings, corpus, index, label=args.suite)
            results.append(row)
            elapsed = time.monotonic() - started
            eta = elapsed / len(results) * (len(jobs) - len(results))
            print(f"[{len(results):02}/{len(jobs)}] {row['id']} "
                  f"{'PASS' if row['passed'] else 'FAIL'} {row['elapsed_seconds']:.1f}s "
                  f"tokens={row['usage']['total_tokens']} ETA={eta:.0f}s", flush=True)
    # asyncio schedules trials; each independently checkpoints to its own directory.
    await asyncio.gather(*(work(job) for job in jobs))
    write_json(OUTPUT / f"{artifact}-summary.json", {
        "manifest": manifest, "total": len(results), "passed": sum(r["passed"] for r in results),
        "trials": [{k: r.get(k) for k in ("id", "harness", "case_id", "passed", "checks", "error",
                                        "usage", "elapsed_seconds")} for r in results],
    })


if __name__ == "__main__":
    asyncio.run(main())
