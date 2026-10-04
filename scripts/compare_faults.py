"""Deterministic scheduling/fault probes, explicitly separate from real model tasks."""
import asyncio
import time
import xml.etree.ElementTree as ET

from openharness.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from openharness.tools.base import ToolRegistry

from backend.harnesses import build_engine
from backend.mock_client import ScriptedClient
from backend.tools import Arguments, DomainTool
from scripts.compare_harnesses import OUTPUT, paired, write_json


def message(*calls, text=""):
    return ConversationMessage(role="assistant", content=[
        *[ToolUseBlock(name=name, input=args) for name, args in calls],
        *([TextBlock(text=text)] if text else []),
    ])


async def main():
    rows = []
    for repetition in range(1, 4):
        for harness in ("legacy", "openharness", "upgraded"):
            timings, registry = [], ToolRegistry()
            async def slow(_, measurements=timings):
                start = time.monotonic()
                await asyncio.sleep(0.2)
                measurements.append({"start": start, "end": time.monotonic()})
                return {"mock": True, "delay_seconds": 0.2}
            for name in ("a", "b", "c"):
                registry.register(DomainTool(name, "Mock delayed read", Arguments, slow))
            client = ScriptedClient([message(("a", {}), ("b", {}), ("c", {})),
                                     message(text="finished")])
            engine = build_engine(harness, client, registry, "Run mock tools.", "scripted")
            start = time.monotonic()
            _ = [e async for e in engine.submit_message("run")]
            total = time.monotonic() - start
            rows.append({
                "harness": harness, "repetition": repetition, "total_seconds": total,
                "tool_span_seconds": max(t["end"] for t in timings) - min(t["start"] for t in timings),
                "messages_paired": paired([m.model_dump(mode="json") for m in engine.messages]),
                "timings": [{"start": t["start"] - start, "end": t["end"] - start}
                            for t in timings],
            })
    tests = []
    source = OUTPUT / "regression.xml"
    if source.exists():
        for test in ET.parse(source).iter("testcase"):
            if test.get("classname") not in {"tests.test_harness_comparison", "tests.test_upgrades"}:
                continue
            tests.append({"name": test.get("name"), "seconds": float(test.get("time")),
                          "passed": test.find("failure") is None and test.find("error") is None})
    write_json(OUTPUT / "faults.json", {
        "basis": "固定脚本Mock与0.2秒人工延迟；不是自然工具选择能力或真实服务加速。",
        "scheduling": rows, "tests": tests,
        "attribution": "串行是该LangGraph基线的实现选择，LangGraph本身也能并行；"
                       "429/timeout重试由共有传输层负责，取消/进程恢复终态由共有应用层负责。",
    })
    print("Recorded", len(rows), "scheduling trials and", len(tests), "fault/upgrade tests")


if __name__ == "__main__":
    asyncio.run(main())
