"""Explicit public export: traces and prompts, never credentials/databases/PDFs."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def main():
    private = dotenv_values(ROOT / "data/private/benchmark.env")
    replacements = [(str(ROOT), "<project>")]
    replacements += [(v, f"<private-{k.lower()}>") for k, v in private.items() if v]

    def scrub(value):
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()
                    if k.lower() not in {"api_key", "authorization", "access_token", "headers"}}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        if isinstance(value, str):
            for old, new in replacements:
                value = value.replace(old, new)
            return value
        return value

    provenance = []
    # Every recorded development and heldout attempt is exported, including failures.
    for src in sorted((ROOT / "data/v3/trials").glob("*.json")):
        raw = json.loads(src.read_text())
        fields = ("id", "arm", "repetition", "created_at", "system_prompt", "tool_schemas",
                  "events", "answer", "errors", "verification", "evidence", "usage",
                  "calls", "context_views", "elapsed_seconds", "checks", "passed", "sources")
        obj = {k: raw[k] for k in fields if k in raw}
        obj["case"] = {k: v for k, v in raw["case"].items() if k != "history"}
        obj["export_note"] = (
            "Private transport model identifier and local paths redacted. Duplicate final history "
            "omitted; actual request/response traces retained in calls. Synthetic starting history "
            "is reproducible with scripts.benchmark_research.context_cases."
        )
        dst = OUT / "v3/trials" / src.name
        write(dst, scrub(obj))
        provenance.append({"source": src.relative_to(ROOT).as_posix(), "source_sha256": digest(src),
                           "export": dst.relative_to(ROOT).as_posix(), "export_sha256": digest(dst)})

    for filename in ("scheduler.json", "ui-study-verification.json"):
        write(OUT / "v3" / filename, scrub(json.loads((ROOT / "data/v3" / filename).read_text())))
    verified = json.loads((ROOT / "data/v3/ui-study-verification.json").read_text())
    study = ROOT / "data/studies" / verified["plan_id"]
    for name in ["plan.json", "result.json"] + [
        f"{kind}-{seed}.{extension}" for seed in (17, 43, 89)
        for kind, extension in (("split", "json"), ("predictions", "json"), ("corruptions", "npz"))
    ]:
        src = study / name
        dst = OUT / "study" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Exact bytes preserve plan, feature and evidence hashes.
        shutil.copyfile(src, dst)

    for src in sorted((ROOT / "data/v3/demo").glob("*")):
        if src.suffix not in {".png", ".md", ".json"}:
            continue
        dst = OUT / "demo" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".json":
            write(dst, scrub(json.loads(src.read_text())))
        elif src.suffix == ".md":
            dst.write_text(scrub(src.read_text()))
        else:
            shutil.copyfile(src, dst)

    # Prior published comparison is kept distinct from the newly frozen experiment.
    for src in sorted((ROOT / "data/comparison").glob("*summary.json")):
        write(OUT / "v2" / src.name, scrub(json.loads(src.read_text())))
    for src in sorted((ROOT / "data/comparison/trials").glob("*.json")):
        raw = json.loads(src.read_text())
        fields = ("id", "group", "case_id", "harness", "repetition", "created_at", "case",
                  "system_prompt", "evaluation_basis", "native_usage", "checks", "passed",
                  "calls", "usage", "elapsed_seconds")
        obj = {k: raw[k] for k in fields if k in raw}
        obj["turns"] = [{k: v for k, v in turn.items() if k != "history"}
                        for turn in raw["turns"]]
        write(OUT / "v2/trials" / src.name, scrub(obj))
    for name in ("faults.json", "citation-gate-ablation.json", "retrieval.json",
                 "agent-evidence-audit.json"):
        write(OUT / "v2" / name, scrub(json.loads((ROOT / "data/comparison" / name).read_text())))
    shutil.copyfile(ROOT / "data/comparison/figures/comparison.png", OUT / "v2/comparison.png")
    freeze = json.loads((ROOT / "benchmarks/v3/implementation-freeze.json").read_text())
    changed = [p for p, h in freeze["sources"].items() if digest(ROOT / p) != h]
    if changed:
        raise ValueError(f"Frozen behavioral sources changed: {changed}")
    write(OUT / "v3/export-manifest.json", {
        "policy": "Whitelisted scientific artifacts; redacted model transport identifiers/paths",
        "frozen_sources_match": True, "trials": provenance,
    })
    # Check text outputs against private values without printing them.
    for path in OUT.rglob("*"):
        if path.suffix not in {".json", ".md"}:
            continue
        text = path.read_text()
        if str(ROOT) in text or any(v and v in text for v in private.values()):
            raise ValueError(f"Private value in public export: {path.relative_to(OUT)}")
    print(f"Exported {len(provenance)} trials and exact study artifacts to evidence/")


if __name__ == "__main__":
    main()
