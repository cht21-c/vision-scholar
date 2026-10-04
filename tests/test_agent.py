import asyncio
from dataclasses import replace

import pytest
from openharness.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

from backend.agent import AgentService
from backend.mock_client import ScriptedClient
from backend.store import ConflictError
from backend.tools import inspect_code


def call(name, arguments):
    return ConversationMessage(role="assistant", content=[ToolUseBlock(name=name, input=arguments)])


def answer(text):
    return ConversationMessage(role="assistant", content=[TextBlock(text=text)])


async def run_service(context, responses, prompt="查看 patchify 源码", mode="code"):
    config, store, library, index = context
    client = ScriptedClient(responses)
    service = AgentService(config, store, library, index, lambda: client)
    await service.start()
    session = store.create_session()
    run = store.create_run(session["id"], prompt, [], mode)
    service.launch(run["id"])
    await service.tasks[run["id"]]
    return service, client, store.run(run["id"]), session


async def test_real_harness_recovers_schema_error_and_pairs_tool_results(context):
    citation = inspect_code("vision_ops.py", "patchify")["citation"]
    service, client, run, session = await run_service(context, [
        call("inspect_code", {"path": "vision_ops.py", "unknown_field": True}),
        call("inspect_code", {"path": "vision_ops.py", "symbol": "patchify"}),
        answer(f"图像按空间位置拆分成非重叠块。{citation}"),
    ])
    try:
        _, store, _, _ = context
        assert run["status"] == "completed"
        assert len(client.requests) == 3
        error_result = client.requests[1].messages[-1].content[0]
        assert isinstance(error_result, ToolResultBlock)
        assert error_result.is_error
        assert store.session(session["id"])["history"]
        events = store.events(run["id"])
        assert any(e["type"] == "tool_end" and e["data"]["is_error"] for e in events)
        assert events[-1]["type"] == "terminal"
        assert run["verification"]["status"] == "verified"
        checkpoint = await service.graph.aget_state({"configurable": {"thread_id": run["id"]}})
        assert checkpoint.values["verification"]["status"] == "verified"
        assert checkpoint.next == ()
    finally:
        await service.close()


async def test_fabricated_citation_cannot_be_published(context):
    service, _, run, session = await run_service(context, [
        call("search_papers", {"query": "residual mapping"}),
        answer("Invented claim [fixture:p88:c1]"),
    ], prompt="解释残差连接", mode="paper")
    try:
        assert run["status"] == "failed"
        assert run["answer"] == ""
        assert run["verification"]["status"] == "invalid"
        assert context[1].session(session["id"])["history"] == []
    finally:
        await service.close()


async def test_multi_turn_restoration_and_concurrent_run_guard(context):
    config, store, library, index = context
    citation = inspect_code("vision_ops.py", "patchify")["citation"]
    client = ScriptedClient([
        call("inspect_code", {"symbol": "patchify"}), answer(f"第一次回答 {citation}"),
        call("inspect_code", {"symbol": "patchify"}), answer(f"第二次回答 {citation}"),
    ])
    service = AgentService(config, store, library, index, lambda: client)
    await service.start()
    try:
        session = store.create_session()
        first = store.create_run(session["id"], "查看源码", [], "code")
        with pytest.raises(ConflictError):
            store.create_run(session["id"], "并发问题", [], "code")
        service.launch(first["id"])
        await service.tasks[first["id"]]
        second = store.create_run(session["id"], "再解释这段代码", [], "code")
        service.launch(second["id"])
        await service.tasks[second["id"]]
        assert store.run(second["id"])["status"] == "completed"
        assert any("第一次回答" in m.text for m in client.requests[2].messages)
        assert len(store.session(session["id"])["history"]) == 8
    finally:
        await service.close()


async def test_cancellation_is_terminal_and_replayable(context):
    config, store, library, index = context
    client = ScriptedClient([answer("unreachable")], delay=10)
    service = AgentService(config, store, library, index, lambda: client)
    await service.start()
    try:
        session = store.create_session()
        run = store.create_run(session["id"], "查看代码", [], "code")
        service.launch(run["id"])
        await asyncio.sleep(0.05)
        result = await service.cancel(run["id"])
        assert result["status"] == "cancelled"
        events = store.events(run["id"])
        assert events[-1]["type"] == "terminal"
        assert store.events(run["id"], events[-2]["seq"]) == [events[-1]]
        assert store.session(session["id"])["history"] == []
        # Cancellation releases the single-run constraint.
        assert store.create_run(session["id"], "后续任务", [], "code")["status"] == "queued"
    finally:
        await service.close()


async def test_timeout_and_model_failure(context):
    config, store, library, index = context
    config = replace(config, run_timeout=1)
    service = AgentService(config, store, library, index,
                           lambda: ScriptedClient([answer("late")], delay=10))
    await service.start()
    try:
        session = store.create_session()
        run = store.create_run(session["id"], "查看代码", [], "code")
        service.launch(run["id"])
        await service.tasks[run["id"]]
        assert store.run(run["id"])["status"] == "failed"
        assert "超过" in store.run(run["id"])["error"]
    finally:
        await service.close()
    # Empty script simulates a provider failure through the real engine loop.
    service, _, result, _ = await run_service(context, [])
    try:
        assert result["status"] == "failed"
        assert "Script exhausted" in result["error"]
    finally:
        await service.close()


async def test_restart_marks_unfinished_jobs_and_retains_completed_history(context):
    config, store, library, index = context
    session = store.create_session()
    store.save_history(session["id"], [answer("已完成的历史").model_dump()])
    run = store.create_run(session["id"], "重启前开始的任务", [], "paper")
    store.update_run(run["id"], status="running")
    service = AgentService(config, store, library, index)
    await service.start()
    try:
        assert store.run(run["id"])["status"] == "interrupted"
        assert store.session(session["id"])["history"][0]["content"][0]["text"] == "已完成的历史"
        assert store.events(run["id"])[-1]["data"]["status"] == "interrupted"
    finally:
        await service.close()
