import asyncio
import json

import numpy as np
import pytest
from sklearn.datasets import load_digits

from backend.store import ConflictError, Store
from backend.studies import StudyConfig, StudyService, canonical, sha
from backend.tools import ResearchTools, StudyReference


def test_short_reference_is_exact_run_scoped_and_does_not_repair_unknown_ids(context):
    config, store, library, index = context
    session = store.create_session()
    run = {"id": "test", "session_id": session["id"], "prompt": "创建方案", "paper_ids": []}
    tools = ResearchTools(store, library, index, config.data_dir, run)
    plan = tools.plan_study(StudyConfig())
    assert plan["plan_ref"] == "study-1"
    assert tools.resolve_study(StudyReference(plan_id="study-1")) == plan["plan_id"]
    assert tools.plan_study(StudyConfig())["plan_ref"] == "study-1"
    for value in ("study-2", plan["plan_id"][:20]):
        with pytest.raises(ValueError):
            tools.resolve_study(StudyReference(plan_id=value))
    other = ResearchTools(store, library, index, config.data_dir, {**run, "id": "other"})
    with pytest.raises(ValueError):
        other.resolve_study(StudyReference(plan_id="study-1"))


@pytest.mark.parametrize("config", [
    {"seeds": [1, 1, 2]}, {"seeds": [1, 2]}, {"seeds": [-1, 2, 3]},
    {"c_candidates": [float("nan")]}, {"c_candidates": [0]}, {"occlusion_size": 9},
])
def test_study_bounds(config):
    with pytest.raises(ValueError):
        StudyConfig(**config)


async def test_real_study_recompute_splits_metrics_pairing_and_reuse(tmp_path):
    service = StudyService(Store(tmp_path / "db"), tmp_path / "studies")
    row = service.plan(StudyConfig(seeds=(11, 13, 19), c_candidates=(0.1, 1.0)))
    assert service.plan(StudyConfig(seeds=(11, 13, 19), c_candidates=(0.1, 1.0)))["id"] == row["id"]
    assert row["status"] == "planned" and row["result"] is None
    finished = await service.run(row["id"])
    assert finished["status"] == "completed" and not finished["reused"]
    reused = await service.run(row["id"])
    assert reused["reused"] and reused["result"] == finished["result"]
    result = finished["result"]
    dataset = load_digits()
    for seed in (11, 13, 19):
        split_path = service.evidence_path(row["id"], f"split-{seed}.json")
        split = json.loads(split_path.read_text())
        assert len(set(split["train"] + split["validation"] + split["test"])) == len(dataset.target)
        assert sum(len(indices) for indices in split.values()) == len(dataset.target)
        prediction = json.loads(service.evidence_path(row["id"], f"predictions-{seed}.json").read_text())
        assert prediction["test_indices"] == split["test"]
        assert prediction["target"] == dataset.target[split["test"]].tolist()
        metrics = [m for m in result["per_seed"] if m["seed"] == seed]
        corruptions = np.load(service.evidence_path(row["id"], f"corruptions-{seed}.npz"))
        for metric in metrics:
            pred = prediction["predictions"][metric["method"]][metric["condition"]]
            assert np.mean(np.array(pred) == prediction["target"]) == metric["accuracy"]
            assert metric["split_sha256"] == sha(canonical(split))
            other = next(m for m in metrics if m["condition"] == metric["condition"]
                         and m["method"] != metric["method"])
            assert metric["test_features_sha256"] == other["test_features_sha256"]
            features = (dataset.data[split["test"]] if metric["condition"] == "clean"
                        else corruptions[metric["condition"]])
            assert sha(features.tobytes()) == metric["test_features_sha256"]
        for selection in (s for s in result["hyperparameter_selection"] if s["seed"] == seed):
            assert selection["fit_samples"] == selection["scaler_fit_samples"] == len(split["train"])
            expected = sorted(selection["candidates"], key=lambda c: (-c["validation_accuracy"], c["C"]))[0]
            assert expected["C"] == selection["selected_C"]
    assert len(result["summary"]) == 6 and len(result["paired_comparisons"]) == 9
    target = service.evidence_path(row["id"], "split-11.json")
    target.write_text("{}")
    with pytest.raises(ValueError, match="工件"):
        service.read(row["id"])


async def test_study_claim_integrity_and_interrupted_state(tmp_path):
    store = Store(tmp_path / "db")
    service = StudyService(store, tmp_path / "studies")
    first = service.plan(StudyConfig(seeds=(11, 13, 19)))
    second = service.plan(StudyConfig(seeds=(23, 29, 31)))
    task = asyncio.create_task(service.run(first["id"]))
    for _ in range(100):
        if service.read(first["id"])["status"] == "running":
            break
        await asyncio.sleep(0.01)
    with pytest.raises(ConflictError):
        await service.run(first["id"])
    with pytest.raises(ConflictError):
        await service.run(second["id"])
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert service.read(first["id"])["status"] == "interrupted"
    with store.connection() as con:
        con.execute("UPDATE studies SET plan='{}' WHERE id=?", (second["id"],))
    with pytest.raises(ValueError, match="哈希"):
        await service.run(second["id"])
