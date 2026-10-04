"""Reconstructed LangGraph model/tools loop. Does not call QueryEngine.

The OpenHarness message/event dataclasses and tool schemas are shared interface
types, not execution logic. Full history is the baseline context policy.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from openharness.api.client import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiRetryEvent,
    ApiTextDeltaEvent,
)
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import (
    ConversationMessage,
    ToolResultBlock,
    sanitize_conversation_messages,
)
from openharness.engine.stream_events import (
    AssistantTextDelta,
    AssistantTurnComplete,
    ErrorEvent,
    StatusEvent,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from openharness.tools.base import ToolExecutionContext


class LoopState(TypedDict):
    messages: list[dict]
    turns: int
    next: str


class LegacyEngine:
    def __init__(self, *, api_client, tool_registry, cwd, model, system_prompt,
                 max_tokens=4500, max_turns=10, checkpointer=None, thread_id="legacy",
                 window_turns: int | None = None, **_):
        self.client, self.registry = api_client, tool_registry
        self.cwd, self.model, self.system_prompt = cwd, model, system_prompt
        self.max_tokens, self.max_turns = max_tokens, max_turns
        self.messages: list[ConversationMessage] = []
        self.total_usage = UsageSnapshot()
        self.thread_id, self.window_turns = thread_id, window_turns
        graph = StateGraph(LoopState)
        graph.add_node("model", self._model)
        graph.add_node("tools", self._tools)
        graph.add_edge(START, "model")
        graph.add_conditional_edges("model", lambda s: s["next"], {"tools": "tools", "end": END})
        graph.add_edge("tools", "model")
        self.graph = graph.compile(checkpointer=checkpointer)

    def load_messages(self, messages):
        self.messages = sanitize_conversation_messages(messages)

    def _request_history(self, messages):
        if self.window_turns is None:
            return messages
        # Cut at real user turns only. Never split assistant/tool-result pairs.
        starts = [i for i, m in enumerate(messages)
                  if m.role == "user" and m.text and not any(
                      isinstance(b, ToolResultBlock) for b in m.content)]
        if len(starts) > self.window_turns:
            return messages[starts[-self.window_turns]:]
        return messages

    async def _model(self, state):
        emit = get_stream_writer()
        self.messages = [ConversationMessage.model_validate(m) for m in state["messages"]]
        if state["turns"] >= self.max_turns:
            emit(ErrorEvent("达到模型轮数上限，任务尚未完成", recoverable=False))
            return {"next": "end"}
        request = ApiMessageRequest(
            model=self.model, system_prompt=self.system_prompt,
            messages=self._request_history(self.messages), max_tokens=self.max_tokens,
            tools=self.registry.to_api_schema(),
        )
        final = None
        async for event in self.client.stream_message(request):
            if isinstance(event, ApiTextDeltaEvent):
                emit(AssistantTextDelta(event.text))
            elif isinstance(event, ApiRetryEvent):
                emit(StatusEvent(event.message))
            elif isinstance(event, ApiMessageCompleteEvent):
                final = event.message
                self.total_usage.input_tokens += event.usage.input_tokens
                self.total_usage.output_tokens += event.usage.output_tokens
                emit(AssistantTurnComplete(final, event.usage))
        if final is None or final.is_effectively_empty():
            emit(ErrorEvent("模型返回空响应", recoverable=False))
            return {"next": "end"}
        self.messages.append(final)
        return {"messages": [m.model_dump(mode="json") for m in self.messages],
                "turns": state["turns"] + 1, "next": "tools" if final.tool_uses else "end"}

    async def _tools(self, state):
        emit = get_stream_writer()
        self.messages = [ConversationMessage.model_validate(m) for m in state["messages"]]
        results = []
        # A conventional serial node, explicitly a baseline implementation choice.
        for call in self.messages[-1].tool_uses:
            emit(ToolExecutionStarted(call.name, call.input))
            try:
                tool = self.registry.get(call.name)
                if tool is None:
                    raise ValueError(f"Unknown tool: {call.name}")
                arguments = tool.input_model.model_validate(call.input)
                result = await tool.execute(arguments, ToolExecutionContext(cwd=self.cwd))
                output, error = result.output, result.is_error
            except Exception as exc:
                output, error = f"{type(exc).__name__}: {exc}", True
            results.append(ToolResultBlock(tool_use_id=call.id, content=output, is_error=error))
            emit(ToolExecutionCompleted(call.name, output, error))
        self.messages.append(ConversationMessage(role="user", content=results))
        return {"messages": [m.model_dump(mode="json") for m in self.messages]}

    async def submit_message(self, prompt):
        self.messages.append(ConversationMessage.from_user_text(prompt))
        state = {"messages": [m.model_dump(mode="json") for m in self.messages],
                 "turns": 0, "next": "model"}
        async for event in self.graph.astream(
            state, {"configurable": {"thread_id": self.thread_id},
                    "recursion_limit": 2 * self.max_turns + 4}, stream_mode="custom",
        ):
            yield event
