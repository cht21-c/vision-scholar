"""Run the default real CPU study without a model or credentials."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from backend.store import Store
from backend.studies import StudyConfig, StudyService


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/reproduced-study"))
    args = parser.parse_args()
    service = StudyService(Store(args.data_dir / "study.db"), args.data_dir / "studies")
    plan = service.plan(StudyConfig())
    print("Plan:", plan["id"], flush=True)
    result = await service.run(plan["id"])
    print(json.dumps({"status": result["status"], "reused": result["reused"],
                      "summary": result["result"]["summary"],
                      "evidence_directory": str(args.data_dir / "studies" / plan["id"])},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
