"""Local comparison demo using the privately stored experiment credentials."""
import os

import uvicorn
from dotenv import dotenv_values

from backend.config import ROOT

if __name__ == "__main__":
    private = dotenv_values(ROOT / "data" / "private" / "benchmark.env")
    os.environ.update(
        VS_PROVIDER="openai", VS_API_KEY=private["BENCH_API_KEY"],
        VS_BASE_URL=private["BENCH_BASE_URL"], VS_MODEL=private["BENCH_MODEL"],
        VS_THINKING_DISABLED="1",
    )
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8765, workers=1)
