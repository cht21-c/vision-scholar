"""Fixed CPU study, executed in a killable child process; no generated code."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from backend.studies import StudyConfig, atomic_json, canonical, fingerprint, sha


def compute(plan: dict, directory: Path) -> dict:
    started = time.monotonic()
    config = StudyConfig.model_validate(plan["config"])
    if plan["fingerprint"] != fingerprint() or sha(canonical(plan)) != directory.name:
        raise ValueError("Plan fingerprint mismatch")
    data = load_digits()
    x, y = data.data, data.target
    rows, comparisons, selections = [], [], []
    with threadpool_limits(limits=1):
        for seed in config.seeds:
            train_val, test = train_test_split(
                np.arange(len(y)), test_size=plan["test_fraction"],
                stratify=y, random_state=seed)
            train, validation = train_test_split(
                train_val, test_size=plan["validation_fraction_of_remaining"],
                stratify=y[train_val], random_state=seed)
            assert not (set(train) & set(validation) or set(train) & set(test)
                        or set(validation) & set(test))
            rng = np.random.default_rng(seed + 1000003)
            noise = np.clip(x[test] + rng.normal(0, config.noise_sigma, x[test].shape), 0, 16)
            occlusion = x[test].copy().reshape(-1, 8, 8)
            positions = rng.integers(0, 9 - config.occlusion_size, size=(len(test), 2))
            for image, (r, c) in zip(occlusion, positions, strict=True):
                image[r:r + config.occlusion_size, c:c + config.occlusion_size] = 0
            conditions = {"clean": x[test], "noise": noise,
                          "occlusion": occlusion.reshape(-1, 64)}
            split = {"train": train.tolist(), "validation": validation.tolist(),
                     "test": test.tolist()}
            atomic_json(directory / f"split-{seed}.json", split)
            np.savez_compressed(directory / f"corruptions-{seed}.npz",
                                noise=noise, occlusion=conditions["occlusion"], positions=positions)
            predictions = {}
            for method in plan["methods"]:
                candidates = []
                models = []
                for c in sorted(config.c_candidates):
                    steps = [StandardScaler()]
                    if method == "pca_svm":
                        steps.append(PCA(n_components=config.pca_components, svd_solver="full"))
                    steps.append(SVC(C=c, kernel="rbf", gamma="scale"))
                    model = make_pipeline(*steps)
                    model.fit(x[train], y[train])
                    score = float(accuracy_score(y[validation], model.predict(x[validation])))
                    candidates.append({"C": c, "validation_accuracy": score})
                    models.append(model)
                # First maximum in ascending C order is the predeclared tie rule.
                best = max(range(len(candidates)), key=lambda i: candidates[i]["validation_accuracy"])
                model = models[best]
                selections.append({"seed": seed, "method": method, "candidates": candidates,
                                   "selected_C": candidates[best]["C"],
                                   "fit_samples": len(train),
                                   "scaler_fit_samples": int(model[0].n_samples_seen_)})
                predictions[method] = {}
                for condition, features in conditions.items():
                    pred = model.predict(features)
                    predictions[method][condition] = pred
                    rows.append({"seed": seed, "method": method, "condition": condition,
                                 "accuracy": float(accuracy_score(y[test], pred)),
                                 "macro_f1": float(f1_score(y[test], pred, average="macro")),
                                 "selected_C": candidates[best]["C"],
                                 "n_train": len(train), "n_validation": len(validation),
                                 "n_test": len(test), "split_sha256": sha(canonical(split)),
                                 "test_features_sha256": sha(features.tobytes())})
            prediction_file = {"seed": seed, "test_indices": test.tolist(), "target": y[test].tolist(),
                               "predictions": {m: {c: p.tolist() for c, p in values.items()}
                                               for m, values in predictions.items()}}
            atomic_json(directory / f"predictions-{seed}.json", prediction_file)
            for condition in conditions:
                delta = ((predictions["pca_svm"][condition] == y[test]).astype(float)
                         - (predictions["pixels_svm"][condition] == y[test]).astype(float))
                boot_rng = np.random.default_rng(seed + 2000003)
                draws = boot_rng.integers(0, len(test), (plan["bootstrap_samples"], len(test)))
                interval = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
                comparisons.append({"seed": seed, "condition": condition,
                                    "pca_minus_pixels": float(delta.mean()),
                                    "paired_bootstrap_95_low": float(interval[0]),
                                    "paired_bootstrap_95_high": float(interval[1]),
                                    "bootstrap_samples": plan["bootstrap_samples"],
                                    "bootstrap_seed": seed + 2000003})
    summary = []
    for condition in plan["conditions"]:
        for method in plan["methods"]:
            accuracies = [r["accuracy"] for r in rows
                          if r["condition"] == condition and r["method"] == method]
            summary.append({"condition": condition, "method": method, "seeds": len(accuracies),
                            "mean_accuracy": float(np.mean(accuracies)),
                            "std_accuracy": float(np.std(accuracies, ddof=1))})
    result = {"plan_id": directory.name, "metrics_origin": "Python sklearn predictions",
              "summary": summary, "per_seed": rows, "paired_comparisons": comparisons,
              "hyperparameter_selection": selections,
              "fingerprint": plan["fingerprint"], "elapsed_seconds": time.monotonic() - started,
              "limitations": plan["limitations"],
              "status": "completed", "fit_policy": plan["selection"]}
    atomic_json(directory / "result.json", result)
    return result


if __name__ == "__main__":
    import json
    target = Path(sys.argv[1])
    compute(json.loads((target / "plan.json").read_text()), target)
