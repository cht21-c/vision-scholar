"""Fault injections: runtime behavior, not real-model accuracy claims."""
import asyncio
import json

import httpx
import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openai import AsyncOpenAI
from openharness.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from openharness.engine.query import MaxTurnsExceeded
from openharness.engine.stream_events import ErrorEvent, ToolExecutionCompleted
from openharness.tools.base import ToolRegistry

from backend.harnesses import build_engine
from backend.mock_client import ScriptedClient
from backend.observed_client import ObservedClient
from backend.tools import Arguments, DomainTool, SearchInput
from scripts.compare_harnesses import paired


def message(*calls, text=""):
    return ConversationMessage(role="assistant", content=[
        *[ToolUseBlock(name=n, input=a) for n, a in calls], *([TextBlock(text=text)] if text else []),
    ])


def engine_for(harness, client, registry, **kwargs):
    return build_engine(harness, client, registry, "Test harness behavior.", "scripted", **kwargs)


@pytest.mark.parametrize("harness", ["legacy", "openharness"])
async def test_fault_schema_unknown_and_exception_recovery(harness):
    registry = ToolRegistry()
    registry.register(DomainTool("search", "", SearchInput, lambda a: {"query": a.query}))
    registry.register(DomainTool("crash", "", Arguments,
                                lambda _: (_ for _ in ()).throw(RuntimeError("injected"))))
    client = ScriptedClient([
        message(("search", {"query": "", "limit": 99})),
        message(("unknown", {})), message(("crash", {})),
        message(("search", {"query": "corrected"})), message(text="recovered"),
    ])
    engine = engine_for(harness, client, registry)
    events = [e async for e in engine.submit_message("run")]
    ends = [e for e in events if isinstance(e, ToolExecutionCompleted)]
    assert [e.is_error for e in ends] == [True, True, True, False]
    assert engine.messages[-1].text == "recovered"
    assert paired([m.model_dump() for m in engine.messages])
    assert len(client.requests) == 5


@pytest.mark.parametrize("harness", ["legacy", "openharness"])
async def test_fault_partial_parallel_failure_has_both_results(harness):
    registry = ToolRegistry()
    registry.register(DomainTool("ok", "", Arguments, lambda _: {"value": 7}))
    registry.register(DomainTool("bad", "", Arguments,
                                lambda _: (_ for _ in ()).throw(RuntimeError("injected"))))
    client = ScriptedClient([message(("ok", {}), ("bad", {})), message(text="one failed")])
    engine = engine_for(harness, client, registry)
    _ = [e async for e in engine.submit_message("go")]
    results = client.requests[1].messages[-1].content
    assert len(results) == 2 and sum(r.is_error for r in results) == 1
    assert paired([m.model_dump() for m in engine.messages])


@pytest.mark.parametrize("harness", ["legacy", "openharness"])
async def test_fault_round_limit_stops_without_final_answer(harness):
    registry = ToolRegistry()
    registry.register(DomainTool("ok", "", Arguments, lambda _: {}))
    client = ScriptedClient([message(("ok", {})) for _ in range(10)])
    engine = engine_for(harness, client, registry, max_turns=2)
    events, raised_limit = [], False
    try:
        async for event in engine.submit_message("loop"):
            events.append(event)
    except MaxTurnsExceeded:
        raised_limit = True
    assert len(client.requests) == 2
    assert raised_limit or any(isinstance(e, ErrorEvent) for e in events)
    assert not engine.messages[-1].text


@pytest.mark.parametrize("harness", ["legacy", "openharness"])
async def test_restore_missing_result_removes_orphan_before_model(harness):
    client = ScriptedClient([message(text="restored")])
    engine = engine_for(harness, client, ToolRegistry())
    engine.load_messages([ConversationMessage.from_user_text("old"), message(("missing", {}))])
    _ = [e async for e in engine.submit_message("continue")]
    assert not any(m.tool_uses for m in client.requests[0].messages)


async def test_legacy_checkpoint_persists_each_loop_node(tmp_path):
    registry = ToolRegistry()
    registry.register(DomainTool("ok", "", Arguments, lambda _: {"value": 3}))
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoint.db")) as saver:
        engine = engine_for("legacy", ScriptedClient([
            message(("ok", {})), message(text="done")]), registry,
            checkpointer=saver, thread_id="trial")
        _ = [e async for e in engine.submit_message("run")]
        config = {"configurable": {"thread_id": "trial"}}
        history = [s async for s in engine.graph.aget_state_history(config)]
        assert any(s.next == ("tools",) for s in history)
        after_tool = next(s for s in history if s.next == ("model",)
                          and s.values.get("messages", [])[-1]["content"][0]["type"] == "tool_result")
        assert after_tool.values["turns"] == 1


@pytest.mark.parametrize("harness", ["legacy", "openharness"])
async def test_native_loop_repeats_identical_writes_without_application_idempotency(harness):
    effects = []
    registry = ToolRegistry()
    registry.register(DomainTool("write", "", Arguments, lambda _: effects.append(1), False))
    engine = engine_for(harness, ScriptedClient([
        message(("write", {})), message(("write", {})), message(text="done")]), registry)
    _ = [e async for e in engine.submit_message("write")]
    # An observed limitation, not a desired product guarantee.
    assert len(effects) == 2


async def test_common_transport_retries_429_and_captures_complete_usage():
    requests = []
    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, json={"error": {"message": "injected", "type": "rate_limit"}})
        chunks = [
            {"id": "test", "object": "chat.completion.chunk", "created": 0, "model": "fixture",
             "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}]},
            {"id": "test", "object": "chat.completion.chunk", "created": 0, "model": "fixture",
             "choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 2,
                                      "total_tokens": 22,
                                      "prompt_tokens_details": {"cached_tokens": 10}}},
        ]
        return httpx.Response(200, text="".join(f"data: {json.dumps(x)}\n\n" for x in chunks)
                              + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"})
    sdk = AsyncOpenAI(api_key="fake", base_url="https://fixture.invalid/v1", max_retries=0,
                      http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    client = ObservedClient("fake", "https://fixture.invalid", sdk=sdk, retry_delay=0)
    try:
        from openharness.api.client import ApiMessageRequest
        _ = [e async for e in client.stream_message(ApiMessageRequest(
            model="fixture", messages=[ConversationMessage.from_user_text("go")]))]
        stats = client.summary()
        assert stats["api_attempts"] == 2 and stats["total_tokens"] is None
        assert stats["observed_total_tokens"] == 22
        assert stats["cached_tokens"] is None and stats["observed_cached_tokens"] == 10
        assert stats["reasoning_tokens"] is None and not stats["usage_complete"]
        assert client.calls[0]["http_status"] == 429
        assert json.loads(requests[-1].content)["stream_options"]["include_usage"]
    finally:
        await client.close()


@pytest.mark.parametrize("harness", ["legacy", "openharness"])
async def test_empty_response_is_not_a_success(harness):
    engine = engine_for(harness, ScriptedClient([message()]), ToolRegistry())
    events = [e async for e in engine.submit_message("empty")]
    assert any(isinstance(e, ErrorEvent) for e in events)
    assert not any(m.role == "assistant" for m in engine.messages)


@pytest.mark.parametrize("harness", ["legacy", "openharness"])
async def test_cancel_during_tool_wait_does_not_continue_model(harness):
    started, finished = asyncio.Event(), asyncio.Event()
    async def wait(_):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()
    registry = ToolRegistry()
    registry.register(DomainTool("wait", "", Arguments, wait))
    client = ScriptedClient([message(("wait", {})), message(text="should not execute")])
    engine = engine_for(harness, client, registry)
    async def collect():
        return [e async for e in engine.submit_message("go")]
    task = asyncio.create_task(collect())
    await asyncio.wait_for(started.wait(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set() and len(client.requests) == 1


async def test_common_transport_timeout_has_bounded_retries_and_unknown_usage():
    requests = []
    def handler(request):
        requests.append(request)
        raise httpx.ReadTimeout("injected", request=request)
    sdk = AsyncOpenAI(api_key="fake", base_url="https://fixture.invalid/v1", max_retries=0,
                      http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    client = ObservedClient("fake", "https://fixture.invalid", sdk=sdk, retry_delay=0)
    try:
        from openharness.api.client import ApiMessageRequest
        with pytest.raises(RuntimeError):
            _ = [e async for e in client.stream_message(ApiMessageRequest(
                model="fixture", messages=[ConversationMessage.from_user_text("go")]))]
        assert len(requests) == 3
        assert client.summary()["total_tokens"] is None
    finally:
        await client.close()
