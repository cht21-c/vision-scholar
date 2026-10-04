import asyncio
import json
from dataclasses import asdict

import pytest
from openharness.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from openharness.engine.stream_events import ErrorEvent
from openharness.tools.base import ToolRegistry

from backend.agent import AgentService
from backend.artifacts import ArtifactInput, ArtifactStore
from backend.harnesses import build_engine
from backend.mock_client import ScriptedClient
from backend.research_engine import (
    CallCompleted,
    CallStarted,
    ContextAssembler,
    ToolScheduler,
    assert_pairs,
)
from backend.tools import Arguments, DomainTool, SearchInput, inspect_code


def assistant(*calls, text=""):
    return ConversationMessage(role="assistant", content=[
        *[ToolUseBlock(name=name, input=args) for name, args in calls],
        *([TextBlock(text=text)] if text else [])])


def history():
    messages = []
    for i in range(12):
        messages.append(ConversationMessage.from_user_text(f"unrelated topic {i}"))
        call = assistant(("lookup", {}))
        messages += [call, ConversationMessage(role="user", content=[
            ToolResultBlock(tool_use_id=call.tool_uses[0].id, content="data " * 1800)]),
                     assistant(text="observed")]
    messages.append(ConversationMessage.from_user_text("当前完整请求必须保存。"))
    return messages


def test_budget_preserves_full_history_and_balanced_turns(tmp_path):
    messages = history()
    before = [m.model_dump() for m in messages]
    assembler = ContextAssembler(ArtifactStore(tmp_path), budget_chars=6000)
    view, stats = assembler.assemble(messages, "system", [])
    assert_pairs(view)
    assert stats["request_chars"] <= 6000 < stats["original_chars"]
    assert stats["summary_calls"] == 0
    assert [m.model_dump() for m in messages] == before
    assert view[-1].text == "当前完整请求必须保存。"
    for message in view:
        for block in message.content:
            if isinstance(block, ToolResultBlock):
                reference = json.loads(block.content)
                restored = assembler.artifacts.read(ArtifactInput(
                    artifact_id=reference["artifact_id"], offset=2000, limit=3000))
                assert restored["text"] == ("data " * 1800)[2000:5000]
    assert len(view) < len(messages)


def test_mandatory_overflow_fails_without_silent_truncation(tmp_path):
    messages = [ConversationMessage.from_user_text("完整用户请求" * 2000)]
    with pytest.raises(ValueError, match="必要上下文"):
        ContextAssembler(ArtifactStore(tmp_path), 1000).assemble(messages, "", [])
    assert messages[0].text == "完整用户请求" * 2000


def test_fitting_request_keeps_large_evidence_inline(tmp_path):
    messages = history()[:4]
    messages.append(ConversationMessage.from_user_text("接着研究"))
    view, stats = ContextAssembler(ArtifactStore(tmp_path), budget_chars=18000).assemble(
        messages, "system", [])
    assert [m.model_dump() for m in view] == [m.model_dump() for m in messages]
    assert not stats["artifacts"]


def test_artifact_tamper_and_cross_session_rejected(tmp_path):
    artifacts = ArtifactStore(tmp_path / "one")
    key = artifacts.put("alpha " * 2000 + "needle DEV-12345")
    assert "DEV-12345" in artifacts.read(ArtifactInput(artifact_id=key, query="needle"))["text"]
    with pytest.raises(ValueError):
        ArtifactStore(tmp_path / "two").read(ArtifactInput(artifact_id=key))
    (tmp_path / "one" / f"{key}.txt").write_text("tampered")
    with pytest.raises(ValueError, match="哈希"):
        artifacts.read(ArtifactInput(artifact_id=key))


@pytest.mark.parametrize("malformation", ["orphan", "missing", "extra", "duplicate"])
def test_pair_validator_rejects_invalid_sequences(malformation):
    call = assistant(("read", {}))
    cid = call.tool_uses[0].id
    result = ConversationMessage(role="user", content=[ToolResultBlock(tool_use_id=cid, content="")])
    cases = {"orphan": [result], "missing": [call],
             "extra": [call, result.model_copy(update={"content": result.content * 2})],
             "duplicate": [call, result, call, result]}
    with pytest.raises(ValueError):
        assert_pairs(cases[malformation])


@pytest.mark.parametrize("mode,expected", [("serial", 1), ("parallel", 0), ("barrier", 1)])
async def test_write_read_visibility_and_actual_handler_intervals(tmp_path, mode, expected):
    state = {"value": 0}
    async def write(_):
        await asyncio.sleep(0.04)
        state["value"] = 1
        return state.copy()
    async def read(_):
        await asyncio.sleep(0.005)
        return state.copy()
    registry = ToolRegistry()
    registry.register(DomainTool("write", "", Arguments, write, False))
    registry.register(DomainTool("read", "", Arguments, read))
    calls = assistant(("write", {}), ("read", {})).tool_uses
    events = []
    results = await ToolScheduler(registry, tmp_path, mode=mode).execute(calls, events.append)
    assert json.loads(results[1].content)["value"] == expected
    starts = {e.call_id: e.started_at for e in events if isinstance(e, CallStarted)}
    ends = {e.call_id: e.metadata for e in events if isinstance(e, CallCompleted)}
    assert all(ends[c.id]["started_at"] == starts[c.id] for c in calls)
    if mode == "barrier":
        assert starts[calls[1].id] >= ends[calls[0].id]["finished_at"]
    assert [r.tool_use_id for r in results] == [c.id for c in calls]


async def test_bounded_parallel_reads_failure_and_schema_results(tmp_path):
    active, peak = 0, 0
    async def read(_):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {}
    async def fail(_):
        raise RuntimeError("private diagnostic")
    registry = ToolRegistry()
    registry.register(DomainTool("read", "", Arguments, read))
    registry.register(DomainTool("fail", "", Arguments, fail))
    registry.register(DomainTool("search", "", SearchInput, read))
    calls = assistant(*([("read", {})] * 8), ("fail", {}), ("search", {}), ("unknown", {})).tool_uses
    events = []
    result = await ToolScheduler(registry, tmp_path, concurrency=3).execute(calls, events.append)
    assert peak == 3 and active == 0
    assert [r.is_error for r in result] == [False] * 8 + [True] * 3
    assert len({e.call_id for e in events}) == len(calls)
    assert "private diagnostic" not in json.dumps([asdict(e) for e in events])


async def test_cancel_cleans_children_and_does_not_continue_model(tmp_path):
    active, cancelled = asyncio.Event(), asyncio.Event()
    async def wait(_):
        active.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    registry = ToolRegistry()
    registry.register(DomainTool("wait", "", Arguments, wait))
    client = ScriptedClient([assistant(("wait", {})), assistant(text="not reached")])
    engine = build_engine("research", client, registry, "", "script",
                          artifacts=ArtifactStore(tmp_path))
    async def collect():
        return [e async for e in engine.submit_message("go")]
    task = asyncio.create_task(collect())
    await asyncio.wait_for(active.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set() and len(client.requests) == 1


@pytest.mark.parametrize("arm", ["research", "research_full"])
async def test_research_error_recovery_decisions_and_loop_limit(tmp_path, arm):
    registry = ToolRegistry()
    registry.register(DomainTool("search", "", SearchInput, lambda args: {"q": args.query}))
    client = ScriptedClient([
        assistant(("search", {})), assistant(("search", {"query": "fixed"})),
        assistant(text="recovered")])
    engine = build_engine(arm, client, registry, "system", "script",
                          artifacts=ArtifactStore(tmp_path))
    engine.load_messages([ConversationMessage.from_user_text("决策：seed=11"), assistant(text="ok")])
    events = [e async for e in engine.submit_message("修正：seed=13")]
    assert len(client.requests) == 3 and engine.messages[-1].text == "recovered"
    assert "seed=13" in client.requests[-1].system_prompt
    assert len(engine.decisions) == 2
    assert_pairs(engine.messages)
    assert sum(e.is_error for e in events if isinstance(e, CallCompleted)) == 1
    limited = build_engine(arm, ScriptedClient([assistant(("search", {"query": "go"}))]),
                           registry, "", "script", max_turns=1, artifacts=ArtifactStore(tmp_path))
    assert any(isinstance(e, ErrorEvent) for e in [
        e async for e in limited.submit_message("loop")])


async def test_research_integrates_agent_trace_and_persistence(context):
    config, store, library, index = context
    citation = inspect_code("vision_ops.py", "patchify")["citation"]
    client = ScriptedClient([assistant(("inspect_code", {"symbol": "patchify"})),
                             assistant(text=f"Source {citation}")])
    service = AgentService(config, store, library, index, lambda: client)
    await service.start()
    try:
        session = store.create_session()
        run = store.create_run(session["id"], "查看 patchify", [], "code", "research")
        service.launch(run["id"])
        await service.tasks[run["id"]]
        actual = store.run(run["id"])
        assert actual["status"] == "completed"
        assert len(actual["usage"]["context_views"]) == 2
        events = store.events(run["id"])
        start = next(e["data"] for e in events if e["type"] == "tool_start")
        end = next(e["data"] for e in events if e["type"] == "tool_end")
        assert start["call_id"] == end["call_id"]
        assert end["metadata"]["elapsed_seconds"] >= 0
        assert_pairs([ConversationMessage.model_validate(m)
                      for m in store.session(session["id"])["history"]])
    finally:
        await service.close()
