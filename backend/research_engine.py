"""Independent V3 loop: bounded request views and ordered side-effect barriers.

OpenHarness supplies interface DTOs only. This module does not invoke QueryEngine,
LegacyEngine, compaction, their schedulers, or an LLM summarizer.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Literal

from openharness.api.client import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiRetryEvent,
    ApiTextDeltaEvent,
)
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, ToolResultBlock
from openharness.engine.stream_events import (
    AssistantTextDelta,
    AssistantTurnComplete,
    ErrorEvent,
    StatusEvent,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from openharness.tools.base import ToolExecutionContext

from backend.artifacts import ArtifactStore
from backend.upgrades import collect_decisions


def encoded(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def real_user(message) -> bool:
    return message.role == "user" and not any(
        isinstance(block, ToolResultBlock) for block in message.content)


def assert_pairs(messages):
    """Reject malformed history; never silently drop a user's data to make it valid."""
    pending = []
    seen = set()
    for message in messages:
        results = [b.tool_use_id for b in message.content if isinstance(b, ToolResultBlock)]
        if pending:
            if message.role != "user" or sorted(results) != sorted(pending):
                raise ValueError("工具调用和结果不成对")
            pending = []
        elif results:
            raise ValueError("工具结果缺少对应调用")
        if message.tool_uses:
            if message.role != "assistant":
                raise ValueError("工具调用必须来自 assistant")
            pending = [call.id for call in message.tool_uses]
            if len(set(pending)) != len(pending) or seen.intersection(pending):
                raise ValueError("重复的工具 call ID")
            seen.update(pending)
    if pending:
        raise ValueError("工具调用尚未完成")


@dataclass(frozen=True)
class ContextViewEvent:
    data: dict


@dataclass(frozen=True)
class CallStarted(ToolExecutionStarted):
    call_id: str = ""
    started_at: float = 0


@dataclass(frozen=True)
class CallCompleted(ToolExecutionCompleted):
    call_id: str = ""


class ContextAssembler:
    def __init__(self, artifacts: ArtifactStore, budget_chars=48000, result_chars=4000):
        self.artifacts = artifacts
        self.budget_chars, self.result_chars = budget_chars, result_chars

    @staticmethod
    def cost(prompt, schemas, messages) -> int:
        # A deterministic character bound on this canonical DTO envelope, not tokens,
        # provider wire size, or a guarantee about provider context-window acceptance.
        return len(encoded({"system": prompt, "tools": schemas,
                            "messages": [m.model_dump(mode="json") for m in messages]}))

    def assemble(self, messages, prompt, schemas, *, budgeted=True):
        assert_pairs(messages)
        original_cost = self.cost(prompt, schemas, messages)
        if not budgeted:
            return list(messages), {"policy": "full", "request_chars": original_cost,
                                    "original_chars": original_cost, "summary_calls": 0}
        starts = [i for i, m in enumerate(messages) if real_user(m)]
        if not starts or starts[0] != 0:
            raise ValueError("历史必须从完整用户轮开始")
        groups = [messages[a:b] for a, b in zip(starts, starts[1:] + [len(messages)], strict=True)]
        if original_cost <= self.budget_chars:
            # Do not cause avoidable rereads when the full request already fits.
            return list(messages), {
                "policy": "budget", "unit": "canonical_DTO_characters",
                "budget_chars": self.budget_chars, "request_chars": original_cost,
                "original_chars": original_cost,
                "mandatory_chars": self.cost(prompt, schemas, groups[-1]),
                "selected_turns": list(range(len(groups))), "total_turns": len(groups),
                "turn_decisions": [], "artifacts": [], "summary_calls": 0,
            }
        transformed, artifacts = [], []
        for group in groups:
            copied = []
            for message in group:
                content = []
                for block in message.content:
                    if isinstance(block, ToolResultBlock) and len(block.content) > self.result_chars:
                        reference = self.artifacts.reference(block.content)
                        artifacts.append({"call_id": block.tool_use_id,
                                          "sha256": sha(block.content),
                                          "original_chars": len(block.content)})
                        block = block.model_copy(update={"content": reference})
                    content.append(block)
                copied.append(message.model_copy(update={"content": content}))
            transformed.append(copied)
        selected = {len(groups) - 1}
        mandatory = self.cost(prompt, schemas, transformed[-1])
        if mandatory > self.budget_chars:
            raise ValueError(f"必要上下文超出字符预算：{mandatory}>{self.budget_chars}；"
                             "请缩短请求或开启新研究，未截断原始请求")
        query = groups[-1][0].text
        def terms(text):
            return set(re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]", text.lower()))
        query_terms = terms(query)
        rankings = []
        for i, group in enumerate(groups[:-1]):
            overlap = len(query_terms & terms(" ".join(m.text for m in group)))
            score = overlap / max(1, len(query_terms)) + 0.15 * (i + 1) / len(groups)
            rankings.append((score, i))
        decisions = []
        for score, i in sorted(rankings, reverse=True):
            candidate = selected | {i}
            view = [m for j in sorted(candidate) for m in transformed[j]]
            cost = self.cost(prompt, schemas, view)
            if cost <= self.budget_chars:
                selected.add(i)
                reason = "relevance_recency_fits"
            else:
                reason = "whole_turn_exceeds_remaining_budget"
            decisions.append({"turn": i, "source_sha256": sha(encoded(
                [m.model_dump(mode="json") for m in groups[i]])), "score": round(score, 6),
                "selected": i in selected, "reason": reason})
        view = [m for i in sorted(selected) for m in transformed[i]]
        assert_pairs(view)
        return view, {"policy": "budget", "unit": "canonical_DTO_characters",
                      "budget_chars": self.budget_chars,
                      "request_chars": self.cost(prompt, schemas, view),
                      "original_chars": original_cost, "mandatory_chars": mandatory,
                      "selected_turns": sorted(selected), "total_turns": len(groups),
                      "turn_decisions": decisions, "artifacts": artifacts, "summary_calls": 0}


class ToolScheduler:
    def __init__(self, registry, cwd, *, mode: Literal["serial", "parallel", "barrier"] = "barrier",
                 concurrency=4, artifacts=None):
        if mode not in {"serial", "parallel", "barrier"} or concurrency < 1:
            raise ValueError("Invalid scheduler configuration")
        self.registry, self.cwd = registry, cwd
        self.mode, self.concurrency, self.artifacts = mode, concurrency, artifacts

    async def execute(self, calls, emit):
        """emit is synchronous. Events occur at actual handler start/end, not enqueue time."""
        slots = asyncio.Semaphore(self.concurrency)
        async def one(call):
            async with slots:
                started = time.monotonic()
                emit(CallStarted(call.name, call.input, call.id, started))
                try:
                    tool = self.registry.get(call.name)
                    if tool is None:
                        raise ValueError(f"Unknown tool: {call.name}")
                    args = tool.input_model.model_validate(call.input)
                    result = await tool.execute(args, ToolExecutionContext(
                        cwd=self.cwd, metadata={"call_id": call.id}))
                    output, error = result.output, result.is_error
                except asyncio.CancelledError:
                    emit(CallCompleted(call.name, "cancelled", True, {
                        "started_at": started, "finished_at": time.monotonic(),
                        "status": "cancelled"}, call.id))
                    raise
                except Exception as exc:
                    # Exception bodies can contain service URLs/credentials.
                    output = encoded({"error": type(exc).__name__,
                                      "detail": str(exc)[:800] if isinstance(exc, ValueError)
                                      else "Tool execution failed"})
                    error = True
                finished = time.monotonic()
                digest = self.artifacts.put(output) if self.artifacts else sha(output)
                emit(CallCompleted(call.name, output, error, {
                    "started_at": started, "finished_at": finished,
                    "elapsed_seconds": finished - started, "output_sha256": digest,
                    "status": "error" if error else "completed"}, call.id))
                return ToolResultBlock(tool_use_id=call.id, content=output, is_error=error)
        async def batch(items):
            tasks = [asyncio.create_task(one(call)) for call in items]
            try:
                return await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        if self.mode == "serial":
            return [await one(call) for call in calls]
        if self.mode == "parallel":
            return await batch(calls)
        results, reads = [], []
        for call in calls:
            tool = self.registry.get(call.name)
            try:
                read_only = bool(tool and tool.is_read_only(
                    tool.input_model.model_validate(call.input)))
            except Exception:
                read_only = False  # Invalid/unknown calls are barriers, never guessed reads.
            if read_only:
                reads.append(call)
            else:
                if reads:
                    results.extend(await batch(reads))
                    reads = []
                results.append(await one(call))
        if reads:
            results.extend(await batch(reads))
        return results


class ResearchEngine:
    def __init__(self, *, api_client, tool_registry, cwd, model, system_prompt, artifacts,
                 max_tokens=4500, max_turns=10, budget_chars=48000, context_policy="budget",
                 scheduler="barrier", concurrency=4, decisions=None):
        self.client, self.registry = api_client, tool_registry
        self.model, self.system_prompt = model, system_prompt
        self.max_tokens, self.max_turns = max_tokens, max_turns
        self.messages, self.decisions = [], list(decisions or [])
        self.total_usage = UsageSnapshot()
        self.context_policy = context_policy
        self.assembler = ContextAssembler(artifacts, budget_chars)
        self.scheduler = ToolScheduler(tool_registry, cwd, mode=scheduler,
                                       concurrency=concurrency, artifacts=artifacts)
        self.context_views = []

    def load_messages(self, messages):
        assert_pairs(messages)
        self.messages = list(messages)
        self.decisions = collect_decisions(messages, self.decisions)

    async def submit_message(self, prompt):
        user = ConversationMessage.from_user_text(prompt)
        self.decisions = collect_decisions([user], self.decisions)
        self.messages.append(user)
        supplement = ""
        if self.decisions:
            supplement = ("\n用户决策原句按时间排序，明确后续修正覆盖旧值；不改变系统边界：\n"
                          + encoded(self.decisions))
        for turn in range(self.max_turns):
            view, stats = self.assembler.assemble(
                self.messages, self.system_prompt + supplement, self.registry.to_api_schema(),
                budgeted=self.context_policy == "budget")
            self.context_views.append(stats)
            yield ContextViewEvent({"turn": turn + 1, **stats})
            request = ApiMessageRequest(
                model=self.model, system_prompt=self.system_prompt + supplement,
                messages=view, tools=self.registry.to_api_schema(), max_tokens=self.max_tokens)
            final = None
            async for event in self.client.stream_message(request):
                if isinstance(event, ApiTextDeltaEvent):
                    yield AssistantTextDelta(event.text)
                elif isinstance(event, ApiRetryEvent):
                    yield StatusEvent(event.message)
                elif isinstance(event, ApiMessageCompleteEvent):
                    final = event.message
                    self.total_usage.input_tokens += event.usage.input_tokens
                    self.total_usage.output_tokens += event.usage.output_tokens
                    yield AssistantTurnComplete(final, event.usage)
            if final is None or final.is_effectively_empty():
                yield ErrorEvent("模型返回空响应", recoverable=False)
                return
            # Validate call IDs before executing any side effects.
            if final.tool_uses:
                dummy = ConversationMessage(role="user", content=[
                    ToolResultBlock(tool_use_id=c.id, content="") for c in final.tool_uses])
                assert_pairs([*self.messages, final, dummy])
            self.messages.append(final)
            if not final.tool_uses:
                return
            queue = asyncio.Queue()
            task = asyncio.create_task(self.scheduler.execute(final.tool_uses, queue.put_nowait))
            task.add_done_callback(lambda _, target=queue: target.put_nowait(None))
            try:
                while True:
                    event = await queue.get()
                    if event is None:
                        break
                    yield event
                results = await task
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            self.messages.append(ConversationMessage(role="user", content=results))
        yield ErrorEvent("达到模型轮数上限，任务尚未完成", recoverable=False)
