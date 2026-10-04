"""Independently recompute reported study metrics and intervals from saved predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import f1_score

from backend.studies import canonical, sha


def verify(directory: Path) -> dict:
    plan = json.loads((directory / "plan.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    assert sha(canonical(plan)) == result["plan_id"], "plan hash"
    dataset = load_digits()
    assert sha(dataset.data.tobytes() + dataset.target.tobytes()) == plan["fingerprint"]["dataset_sha256"]
    checked = 0
    for seed in plan["config"]["seeds"]:
        split = json.loads((directory / f"split-{seed}.json").read_text())
        merged = split["train"] + split["validation"] + split["test"]
        assert sorted(merged) == list(range(len(dataset.target))), "disjoint partition"
        prediction = json.loads((directory / f"predictions-{seed}.json").read_text())
        assert prediction["test_indices"] == split["test"], "test pairing"
        truth = dataset.target[split["test"]]
        assert prediction["target"] == truth.tolist(), "target labels"
        with np.load(directory / f"corruptions-{seed}.npz") as corruptions:
            for row in result["per_seed"]:
                if row["seed"] != seed:
                    continue
                pred = np.array(prediction["predictions"][row["method"]][row["condition"]])
                assert np.isclose(row["accuracy"], (pred == truth).mean(), atol=1e-12)
                assert np.isclose(row["macro_f1"], f1_score(truth, pred, average="macro"), atol=1e-12)
                assert row["split_sha256"] == sha(canonical(split))
                features = (dataset.data[split["test"]] if row["condition"] == "clean"
                            else corruptions[row["condition"]])
                assert row["test_features_sha256"] == sha(features.tobytes())
                checked += 1
        for row in result["paired_comparisons"]:
            if row["seed"] != seed:
                continue
            predictions = prediction["predictions"]
            pca = np.array(predictions["pca_svm"][row["condition"]])
            pixels = np.array(predictions["pixels_svm"][row["condition"]])
            delta = (pca == truth).astype(float) - (pixels == truth).astype(float)
            draws = np.random.default_rng(row["bootstrap_seed"]).integers(
                0, len(truth), (row["bootstrap_samples"], len(truth)))
            interval = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
            assert np.isclose(row["pca_minus_pixels"], delta.mean(), atol=1e-12)
            assert np.allclose(interval, [row["paired_bootstrap_95_low"],
                                          row["paired_bootstrap_95_high"]], atol=1e-12)
    for row in result["summary"]:
        values = [r["accuracy"] for r in result["per_seed"]
                  if (r["method"], r["condition"]) == (row["method"], row["condition"])]
        assert np.isclose(row["mean_accuracy"], np.mean(values), atol=1e-12)
        assert np.isclose(row["std_accuracy"], np.std(values, ddof=1), atol=1e-12)
    return {"verified": True, "plan_id": result["plan_id"], "per_seed_metrics_recomputed": checked,
            "scope": "Saved predictions, split labels, feature hashes, means/std and paired CIs. "
                     "Re-run reproduce_study to independently retrain the models."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    print(json.dumps(verify(parser.parse_args().directory), indent=2))
