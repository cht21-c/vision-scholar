"""Identical scripted calls replayed across scheduler ablations; artificial delays."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict

from openharness.engine.messages import ToolUseBlock
from openharness.tools.base import ToolRegistry

from backend.config import ROOT
from backend.research_engine import CallCompleted, ToolScheduler
from backend.tools import Arguments, DomainTool, SearchInput
from scripts.compare_harnesses import write_json


async def replay(case, mode, repetition, protocol):
    state = {"value": 0, "active": 0, "peak": 0}
    entered = asyncio.Event()
    async def read(_):
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        entered.set()
        try:
            if case == "cancel_batch":
                await asyncio.Event().wait()
            await asyncio.sleep(protocol["read_delay_seconds"])
            return {"value": state["value"]}
        finally:
            state["active"] -= 1
    async def write(_):
        await asyncio.sleep(protocol["write_delay_seconds"])
        state["value"] += 1
        return {"value": state["value"]}
    async def fail(_):
        await asyncio.sleep(0.005)
        raise RuntimeError("controlled failure")
    registry = ToolRegistry()
    for name, handler, readonly, schema in [
        ("read", read, True, Arguments), ("write", write, False, Arguments),
        ("fail", fail, True, Arguments), ("validated", read, True, SearchInput),
    ]:
        registry.register(DomainTool(name, "", schema, handler, readonly))
    recipes = {
        "independent_reads": [("read", {})] * 6,
        "write_then_read": [("write", {}), ("read", {})],
        "read_write_read": [("read", {}), ("read", {}), ("write", {}), ("read", {})],
        "one_read_fails": [("read", {}), ("fail", {}), ("read", {})],
        "invalid_schema": [("validated", {})],
        "unknown_tool": [("unknown", {})],
        "cancel_batch": [("read", {})] * 5,
        "bound_concurrency": [("read", {})] * 12,
    }
    calls = [ToolUseBlock(id=f"{case}-{i}", name=n, input=a)
             for i, (n, a) in enumerate(recipes[case])]
    events, results, cancelled = [], [], False
    scheduler = ToolScheduler(registry, ROOT, mode=mode,
                              concurrency=protocol["max_concurrency"])
    started = time.monotonic()
    task = asyncio.create_task(scheduler.execute(calls, events.append))
    if case == "cancel_batch":
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
    try:
        results = await task
    except asyncio.CancelledError:
        cancelled = True
    elapsed = time.monotonic() - started
    if case == "cancel_batch":
        checks = {"cancelled": cancelled, "children_finished": state["active"] == 0}
    else:
        checks = {"results_paired": [r.tool_use_id for r in results] == [c.id for c in calls],
                  "bounded": state["peak"] <= protocol["max_concurrency"]}
        if case in {"write_then_read", "read_write_read"}:
            checks["read_observes_prior_write"] = json.loads(results[-1].content)["value"] == 1
        if case in {"one_read_fails", "invalid_schema", "unknown_tool"}:
            checks["error_result"] = sum(r.is_error for r in results) == 1
    return {"case": case, "mode": mode, "repetition": repetition, "calls": [c.model_dump() for c in calls],
            "results": [r.model_dump() for r in results], "events": [
                {"type": type(e).__name__, **asdict(e)} for e in events],
            "checks": checks, "contract_pass": all(checks.values()), "elapsed_seconds": elapsed,
            "peak_read_concurrency": state["peak"],
            "completion_events": sum(isinstance(e, CallCompleted) for e in events)}


async def main():
    protocol = json.loads((ROOT / "benchmarks/v3/protocol.json").read_text())["scheduler"]
    rows = []
    for repetition in range(1, protocol["repeats"] + 1):
        for case in protocol["cases"]:
            for mode in protocol["modes"]:
                rows.append(await replay(case, mode, repetition, protocol))
    target = ROOT / "data/v3/scheduler.json"
    write_json(target, {"protocol": protocol, "timings_are_artificial": True, "trials": rows})
    for mode in protocol["modes"]:
        selected = [r for r in rows if r["mode"] == mode]
        print(mode, sum(r["contract_pass"] for r in selected), "/", len(selected))
    print(target)


if __name__ == "__main__":
    asyncio.run(main())
