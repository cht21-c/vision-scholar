"""Shared OpenAI transport: explicit sampling, streaming usage and per-attempt traces.

Reuses wire-format DTO conversion only; no agent loop or context policy lives here.
"""
from __future__ import annotations

import asyncio
import copy
import json
import time
from typing import Any
from uuid import uuid4

from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from openharness.api.client import ApiMessageCompleteEvent, ApiRetryEvent, ApiTextDeltaEvent
from openharness.api.openai_client import (
    _convert_messages_to_openai,
    _convert_tools_to_openai,
    _strip_think_blocks,
    _token_limit_param_for_model,
)
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, TextBlock, ToolUseBlock


class ObservedClient:
    def __init__(self, api_key: str, base_url: str, *, timeout: float = 90,
                 thinking_disabled: bool = False, sdk=None, retry_delay: float = 0.5):
        self.sdk = sdk or AsyncOpenAI(api_key=api_key, base_url=base_url,
                                      timeout=timeout, max_retries=0)
        self.thinking_disabled = thinking_disabled
        self.retry_delay = retry_delay
        self.calls: list[dict[str, Any]] = []
        self.started = time.monotonic()
        self.first_visible_seconds: float | None = None

    async def close(self):
        await self.sdk.close()

    def summary(self) -> dict:
        usage = [c["usage"] for c in self.calls if c.get("usage") is not None]
        def summed(key):
            values = [u.get(key) for u in usage]
            return sum(values) if values and all(v is not None for v in values) else None
        complete = len(usage) == len(self.calls)
        return {
            "api_attempts": len(self.calls), "usage_records": len(usage),
            "input_tokens": summed("input_tokens") if complete else None,
            "output_tokens": summed("output_tokens") if complete else None,
            "cached_tokens": summed("cached_tokens") if complete else None,
            "reasoning_tokens": summed("reasoning_tokens") if complete else None,
            "total_tokens": summed("total_tokens") if complete else None,
            "observed_input_tokens": summed("input_tokens"),
            "observed_output_tokens": summed("output_tokens"),
            "observed_total_tokens": summed("total_tokens"),
            "observed_cached_tokens": summed("cached_tokens"),
            "observed_reasoning_tokens": summed("reasoning_tokens"),
            "usage_complete": complete,
            "first_visible_seconds": self.first_visible_seconds,
            "peak_request_chars": max((c["request_chars"] for c in self.calls), default=0),
            "no_tools_requests": sum(not c["request"].get("tools") for c in self.calls),
            "actual_models": sorted({c["actual_model"] for c in self.calls
                                     if c.get("actual_model")}),
        }

    async def stream_message(self, request):
        params = {
            "model": request.model,
            "messages": _convert_messages_to_openai(request.messages, request.system_prompt),
            "stream": True, "stream_options": {"include_usage": True}, "temperature": 0,
            **_token_limit_param_for_model(request.model, request.max_tokens),
        }
        if request.tools:
            params["tools"] = _convert_tools_to_openai(request.tools)
        if self.thinking_disabled:
            params["extra_body"] = {"thinking": {"type": "disabled"}}
        for attempt in range(3):
            started = time.monotonic()
            record = {
                "attempt": attempt + 1, "request": copy.deepcopy(params),
                "request_chars": len(json.dumps(params, ensure_ascii=False)),
                "message_count": len(params["messages"]), "usage": None,
                "actual_model": None, "first_delta_seconds": None,
            }
            self.calls.append(record)
            text, think_buffer, tool_calls, finish = "", "", {}, None
            visible_sent = False
            stream = None
            try:
                stream = await self.sdk.chat.completions.create(**params)
                async for chunk in stream:
                    record["actual_model"] = chunk.model
                    if chunk.usage:
                        raw = chunk.usage.model_dump()
                        pd = raw.get("prompt_tokens_details") or {}
                        cd = raw.get("completion_tokens_details") or {}
                        record["usage"] = {
                            "input_tokens": raw.get("prompt_tokens"),
                            "output_tokens": raw.get("completion_tokens"),
                            "total_tokens": raw.get("total_tokens"),
                            "cached_tokens": pd.get("cached_tokens"),
                            "reasoning_tokens": cd.get("reasoning_tokens"), "raw": raw,
                        }
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    finish = choice.finish_reason or finish
                    delta = choice.delta
                    if delta.content:
                        think_buffer += delta.content
                        visible, think_buffer = _strip_think_blocks(think_buffer)
                        if visible:
                            text += visible
                            visible_sent = True
                            if record["first_delta_seconds"] is None:
                                record["first_delta_seconds"] = time.monotonic() - started
                            if self.first_visible_seconds is None:
                                self.first_visible_seconds = time.monotonic() - self.started
                            yield ApiTextDeltaEvent(visible)
                    for part in delta.tool_calls or []:
                        entry = tool_calls.setdefault(part.index, {
                            "id": "", "name": "", "arguments": "",
                        })
                        if part.id:
                            entry["id"] = part.id
                        if part.function:
                            if part.function.name:
                                entry["name"] = part.function.name
                            entry["arguments"] += part.function.arguments or ""
                content = [TextBlock(text=text)] if text else []
                for _, tool in sorted(tool_calls.items()):
                    if not tool["name"]:
                        continue
                    try:
                        args = json.loads(tool["arguments"])
                        if not isinstance(args, dict):
                            raise ValueError("Tool arguments must be an object")
                    except (ValueError, TypeError):
                        # This field deliberately fails strict schemas; never silently default.
                        args = {"__invalid_json__": tool["arguments"]}
                    content.append(ToolUseBlock(id=tool["id"] or f"call_{uuid4().hex}",
                                                name=tool["name"], input=args))
                if not content:
                    raise ValueError("模型返回空响应")
                message = ConversationMessage(role="assistant", content=content)
                record.update(response=message.model_dump(mode="json"), finish_reason=finish,
                              elapsed_seconds=time.monotonic() - started)
                usage = record["usage"] or {}
                yield ApiMessageCompleteEvent(
                    message, UsageSnapshot(input_tokens=usage.get("input_tokens") or 0,
                                           output_tokens=usage.get("output_tokens") or 0), finish,
                )
                return
            except asyncio.CancelledError:
                record.update(error="cancelled", elapsed_seconds=time.monotonic() - started)
                raise
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                # Do not persist exception bodies/URLs/headers, which can contain credentials.
                record.update(error=type(exc).__name__, http_status=status,
                              elapsed_seconds=time.monotonic() - started)
                retryable = (isinstance(exc, (APIConnectionError, TimeoutError))
                             or isinstance(exc, APIStatusError)
                             and status in {429, 500, 502, 503, 504})
                if not retryable or visible_sent or attempt == 2:
                    raise RuntimeError(f"模型调用失败：{type(exc).__name__}，状态 {status}") from exc
                delay = self.retry_delay * (2 ** attempt)
                yield ApiRetryEvent("模型暂时不可用，正在重试", attempt + 1, 3, delay)
                await asyncio.sleep(delay)
            finally:
                if stream is not None:
                    await stream.close()
