"""Explicit scripted provider for engineering tests, never a model-quality substitute."""
from __future__ import annotations

import asyncio
import copy
import json

from openharness.api.client import ApiMessageCompleteEvent, ApiTextDeltaEvent
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


class ScriptedClient:
    def __init__(self, responses: list[ConversationMessage], delay: float = 0):
        self.responses = list(responses)
        self.requests = []
        self.delay = delay

    async def stream_message(self, request):
        self.requests.append(copy.deepcopy(request))
        await asyncio.sleep(self.delay)
        if not self.responses:
            raise RuntimeError("Script exhausted")
        message = self.responses.pop(0)
        if message.text:
            yield ApiTextDeltaEvent(message.text)
        yield ApiMessageCompleteEvent(message, UsageSnapshot(),
                                      "tool_use" if message.tool_uses else "end_turn")

    async def close(self):
        pass


class MockClient:
    async def stream_message(self, request):
        await asyncio.sleep(0.15)
        results = [b for b in request.messages[-1].content if isinstance(b, ToolResultBlock)]
        if not results:
            available = {t["name"] for t in request.tools}
            if "run_experiment" in available:
                tool = ToolUseBlock(name="run_experiment", input={"kind": "attention", "seed": 42})
            elif "inspect_code" in available:
                tool = ToolUseBlock(name="inspect_code", input={"symbol": "patchify"})
            else:
                tool = ToolUseBlock(name="search_papers",
                                    input={"query": "residual identity shortcut mapping", "limit": 3})
            message = ConversationMessage(role="assistant", content=[tool])
        else:
            payload = json.loads(results[0].content) if not results[0].is_error else {}
            items = payload.get("results", [])
            citation = items[0].get("citation", "") if items else payload.get("citation", "")
            text = ("**脚本 Mock · 仅验证工具链路**\n\n工具已返回实际数据。"
                    f"{citation}\n\n此输出由固定脚本生成，不能用来判断模型回答质量。")
            message = ConversationMessage(role="assistant", content=[TextBlock(text=text)])
            yield ApiTextDeltaEvent(text)
        yield ApiMessageCompleteEvent(message, UsageSnapshot(),
                                      "tool_use" if message.tool_uses else "end_turn")

    async def close(self):
        pass
