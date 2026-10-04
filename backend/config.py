from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    provider: str = "offline"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    run_timeout: int = 180
    max_turns: int = 10
    thinking_disabled: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        provider = os.environ.get("VS_PROVIDER", "")
        base_url = os.environ.get("VS_BASE_URL", "")
        api_key = os.environ.get("VS_API_KEY", "")
        model = os.environ.get("VS_MODEL", "")
        gateway = os.environ.get("ANTHROPIC_BASE_URL", "")
        if not provider and gateway and urlsplit(gateway).hostname in {
            "localhost", "127.0.0.1", "::1"
        }:
            provider = "anthropic"
            base_url = gateway
            api_key = "local-runtime"
            model = model or "modelhub/gpt-5.5-2026-04-24"
        elif not provider and os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
            base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            api_key = os.environ["OPENAI_API_KEY"]
            model = model or "gpt-4.1-mini"
        settings = cls(
            data_dir=Path(os.environ.get("VS_DATA_DIR", str(ROOT / "data"))).resolve(),
            provider=provider or "offline",
            model=model,
            base_url=base_url,
            api_key=api_key,
            run_timeout=int(os.environ.get("VS_RUN_TIMEOUT", "180")),
            max_turns=int(os.environ.get("VS_MAX_TURNS", "10")),
            thinking_disabled=os.environ.get("VS_THINKING_DISABLED", "0") == "1",
        )
        settings.prepare()
        return settings

    def prepare(self) -> None:
        for name in ("papers", "reports", "experiments", "harness"):
            (self.data_dir / name).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("OPENHARNESS_CONFIG_DIR", str(self.data_dir / "harness"))
        os.environ.setdefault("OPENHARNESS_DATA_DIR", str(self.data_dir / "harness" / "data"))

    def public(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "available": self.provider in {"openai", "anthropic", "mock"},
            "mock": self.provider == "mock",
            "runtime": "Vision Scholar ResearchEngine / OpenHarness comparison",
            "harnesses": ["legacy", "openharness", "upgraded", "research"],
            "retrievers": ["lsa", "neural"],
            "max_turns": self.max_turns,
        }
