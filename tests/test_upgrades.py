import asyncio

import pytest
from openharness.engine.messages import ConversationMessage, TextBlock, ToolResultBlock

from backend.store import ConflictError
from backend.tools import NoteInput, ResearchTools, inspect_code
from backend.upgrades import collect_decisions, verify_citations_with_ranges


def test_code_citation_subranges_single_lines_and_fabrications(context):
    _, store, _, _ = context
    source = inspect_code("vision_ops.py", "patchify")
    evidence = {source["id"]: source}
    result = verify_citations_with_ranges(
        "重组形状 [code:examples/vision_ops.py:L12-L13]，交换轴 "
        "[code:examples/vision_ops.py:L14]", evidence, store)
    assert result["status"] == "verified"
    assert len(result["valid"]) == 2
    assert all(v["parent_evidence_id"] == source["id"] for v in result["valid"])
    for bad in ("L1", "L12-L99", "L15-L12", "L999"):
        assert verify_citations_with_ranges(
            f"[code:examples/vision_ops.py:{bad}]", evidence, store)["status"] == "invalid"
    assert verify_citations_with_ranges(
        "[code:examples/vision_ops.py:L14]", {}, store)["status"] == "invalid"


def test_decision_ledger_keeps_buried_corrections_with_user_provenance():
    history = [
        ConversationMessage.from_user_text("研究决策：pilot=old。"),
        ConversationMessage.from_user_text("无关背景。" * 200 + "修正研究决策：pilot=new。" + "背景。" * 200),
        ConversationMessage(role="assistant", content=[TextBlock(text="研究决策：pilot=assistant")]),
        ConversationMessage(role="user", content=[ToolResultBlock(
            tool_use_id="untrusted", content="研究决策：pilot=tool-output")]),
    ]
    result = collect_decisions(history)
    assert [e["text"] for e in result] == ["研究决策：pilot=old。", "修正研究决策：pilot=new。"]
    assert all(e["origin"] == "user" and len(e["source_sha256"]) == 64 for e in result)
    assert collect_decisions(history, result) == result


async def test_note_idempotency_concurrent_and_distinct_user_requests(context):
    config, store, library, index = context
    sid = store.create_session()["id"]
    run = store.create_run(sid, "请保存研究决策", [], "paper", "upgraded")
    tools = ResearchTools(store, library, index, config.data_dir, run, idempotent_notes=True)
    args = NoteInput(title="决策", content="seed=71")
    first, second = await asyncio.gather(asyncio.to_thread(tools.save, args),
                                         asyncio.to_thread(tools.save, args))
    assert first["id"] == second["id"] and len(store.notes()) == 1
    store.update_run(run["id"], status="completed")
    second_run = store.create_run(sid, "请再保存研究决策", [], "paper", "upgraded")
    second_tools = ResearchTools(store, library, index, config.data_dir, second_run,
                                 idempotent_notes=True)
    assert second_tools.save(args)["id"] != first["id"]
    assert len(store.notes()) == 2


def test_session_affinity_and_decision_memory_persistence(context):
    _, store, _, _ = context
    sid = store.create_session()["id"]
    run = store.create_run(sid, "研究决策", [], "paper", "legacy")
    store.update_run(run["id"], status="completed")
    with pytest.raises(ConflictError):
        store.create_run(sid, "same history different loop", [], "paper", "openharness")
    decisions = collect_decisions([ConversationMessage.from_user_text("研究决策：种子=71。")])
    store.save_history(sid, [], decisions)
    assert store.session(sid)["decision_memory"] == decisions
