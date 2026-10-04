"""Explicit loop selection. Upstream QueryEngine is used without modification."""
from openharness.config.settings import PermissionSettings
from openharness.engine.query_engine import QueryEngine
from openharness.permissions.checker import PermissionChecker

from backend.config import ROOT
from backend.legacy_agent import LegacyEngine


def build_engine(harness, client, registry, prompt, model, *, max_turns=10, max_tokens=4500,
                 context_window=100000, compact_threshold=85000, checkpointer=None,
                 thread_id="legacy", window_turns=None, decisions=None, artifacts=None,
                 budget_chars=48000, scheduler="barrier", concurrency=4):
    common = dict(api_client=client, tool_registry=registry, cwd=ROOT / "examples",
                  model=model, system_prompt=prompt, max_tokens=max_tokens, max_turns=max_turns)
    if harness in {"research", "research_full"}:
        from backend.research_engine import ResearchEngine
        if artifacts is None:
            raise ValueError("ResearchEngine requires a session-scoped artifact store")
        return ResearchEngine(**common, artifacts=artifacts, decisions=decisions,
                              budget_chars=budget_chars, scheduler=scheduler,
                              concurrency=concurrency,
                              context_policy="full" if harness == "research_full" else "budget")
    if harness == "legacy":
        return LegacyEngine(**common, checkpointer=checkpointer, thread_id=thread_id,
                            window_turns=window_turns)
    if harness in {"openharness", "upgraded", "upgraded_memory"}:
        engine = QueryEngine(**common, permission_checker=PermissionChecker(PermissionSettings(
            allowed_tools=[tool.name for tool in registry.list_tools()],
        )), context_window_tokens=context_window,
            auto_compact_threshold_tokens=compact_threshold)
        if harness in {"upgraded", "upgraded_memory"}:
            from backend.upgrades import DecisionProtectedEngine
            return DecisionProtectedEngine(engine, decisions)
        return engine
    raise ValueError(f"运行版本尚未实现：{harness}")
