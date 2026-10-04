"""Registered CPU experiments; no generated code or shell execution."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
import sklearn
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from backend.config import ROOT
from backend.store import Store, dump
from examples.vision_ops import patchify, scaled_dot_product_attention


class ExperimentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["digits", "attention"] = "digits"
    seed: int = Field(default=42, ge=0, le=100000)
    pca_components: int = Field(default=24, ge=2, le=48)
    image_size: int = Field(default=8, ge=4, le=32)
    patch_size: int = Field(default=2, ge=1, le=8)

    @model_validator(mode="after")
    def shape_check(self):
        if self.kind == "attention":
            if self.image_size % self.patch_size:
                raise ValueError("图像尺寸必须能被 patch 大小整除")
            if (self.image_size // self.patch_size) ** 2 > 256:
                raise ValueError("数学演示最多 256 个 token")
        return self


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_digits(config: ExperimentInput) -> dict:
    dataset = load_digits()
    x, y = dataset.data, dataset.target
    indices = np.arange(len(y))
    train_val, test = train_test_split(
        indices, test_size=0.2, stratify=y, random_state=config.seed,
    )
    train, validation = train_test_split(
        train_val, test_size=0.25, stratify=y[train_val], random_state=config.seed,
    )
    trials, results = [], []
    for variant in ("pixels", "pca"):
        best = None
        for c in (0.5, 2.0, 8.0):
            steps = [StandardScaler()]
            if variant == "pca":
                steps.append(PCA(n_components=config.pca_components, svd_solver="full"))
            steps.append(SVC(C=c, kernel="rbf", gamma="scale"))
            pipeline = make_pipeline(*steps)
            pipeline.fit(x[train], y[train])
            val_accuracy = float(accuracy_score(y[validation], pipeline.predict(x[validation])))
            trials.append({"variant": variant, "C": c, "validation_accuracy": val_accuracy})
            if best is None or val_accuracy > best[0]:
                best = (val_accuracy, c, pipeline)
        # The test partition is accessed only after hyperparameter selection on validation.
        val_accuracy, chosen_c, fitted = best
        prediction = fitted.predict(x[test])
        results.append({
            "variant": variant, "selected_C": chosen_c, "validation_accuracy": val_accuracy,
            "test_accuracy": float(accuracy_score(y[test], prediction)),
            "test_macro_f1": float(f1_score(y[test], prediction, average="macro")),
            "confusion_matrix": confusion_matrix(y[test], prediction, labels=range(10)).tolist(),
        })
    # Separate real SGD trace for log-diagnosis demonstrations (not the SVC training log).
    scaler = StandardScaler().fit(x[train])
    train_x, val_x = scaler.transform(x[train]), scaler.transform(x[validation])
    learner = SGDClassifier(loss="log_loss", alpha=0.001, random_state=config.seed,
                            learning_rate="constant", eta0=0.01)
    rng = np.random.default_rng(config.seed)
    trace = []
    for epoch in range(1, 17):
        order = rng.permutation(len(train))
        learner.partial_fit(train_x[order], y[train][order], classes=np.arange(10))
        trace.append({
            "epoch": epoch,
            "train_loss": float(log_loss(y[train], learner.predict_proba(train_x),
                                        labels=range(10))),
            "val_loss": float(log_loss(y[validation], learner.predict_proba(val_x),
                                      labels=range(10))),
            "train_accuracy": float(accuracy_score(y[train], learner.predict(train_x))),
            "val_accuracy": float(accuracy_score(y[validation], learner.predict(val_x))),
        })
    return {
        "dataset": "scikit-learn load_digits (UCI optical recognition, real 8×8 digits)",
        "samples": len(y), "features": x.shape[1], "classes": 10,
        "split": {"train": len(train), "validation": len(validation), "test": len(test)},
        "split_sha256": digest(np.concatenate([train, [-1], validation, [-1], test]).tobytes()),
        "dataset_sha256": digest(x.tobytes() + y.tobytes()),
        "selection": "每种预先声明的特征方案只用 validation 选 C；独立 test 各评一次。Scaler/PCA 仅拟合 train。",
        "trials": trials, "results": results,
        "training_log": trace, "training_log_source": "独立 SGDClassifier 16 个真实 epoch，非 SVC 日志",
        "limitation": "教学基线实验，不是 ResNet/ViT 的论文规模复现；单次拆分不能证明统计显著性。",
    }


def run_attention(config: ExperimentInput) -> dict:
    rng = np.random.default_rng(config.seed)
    image = rng.random((config.image_size, config.image_size, 1))
    patches = patchify(image, config.patch_size)
    projection = rng.normal(size=(patches.shape[1], 8))
    embeddings = patches @ projection
    output, weights = scaled_dot_product_attention(embeddings, embeddings, embeddings)
    return {
        "dataset": "Seeded random matrix · 数学实验",
        "image_shape": list(image.shape), "patch_shape": list(patches.shape),
        "token_count": patches.shape[0], "embedding_shape": list(embeddings.shape),
        "output_shape": list(output.shape),
        "row_sums_min": float(weights.sum(axis=1).min()),
        "row_sums_max": float(weights.sum(axis=1).max()),
        "attention_preview": weights[:16, :16].round(6).tolist(),
        "limitation": "随机权重的 attention 数学演示，没有训练、不含 CLS token，不表示图像理解能力。",
    }


def execute_experiment(config: ExperimentInput, store: Store, output_dir: Path) -> dict:
    start = time.monotonic()
    with threadpool_limits(limits=1):
        result = run_digits(config) if config.kind == "digits" else run_attention(config)
    result.update({
        "seed": config.seed, "elapsed_seconds": round(time.monotonic() - start, 3),
        "code_sha256": digest(Path(__file__).read_bytes()
                              + (ROOT / "examples" / "vision_ops.py").read_bytes()),
        "versions": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "metrics_origin": "Python computation",
    })
    experiment = store.add_experiment(config.kind, config.model_dump(), result)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{experiment['id']}.json").write_text(dump(experiment), encoding="utf-8")
    return experiment


class Epoch(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    epoch: int = Field(ge=0)
    train_loss: float = Field(ge=0)
    val_loss: float = Field(ge=0)
    train_accuracy: float = Field(ge=0, le=1)
    val_accuracy: float = Field(ge=0, le=1)


def analyze_log(content: str) -> dict:
    if len(content) > 2_000_000:
        raise ValueError("日志超过 2 MB")
    try:
        if content.lstrip().startswith(("[", "{")):
            data = json.loads(content)
            if isinstance(data, dict):
                data = data.get("training_log", data.get("result", {}).get("training_log"))
        else:
            data = list(csv.DictReader(io.StringIO(content)))
        if not isinstance(data, list) or not 2 <= len(data) <= 10000:
            raise ValueError("需要 2–10000 条 epoch 记录")
        records = [Epoch.model_validate(row) for row in data]
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("日志需包含 epoch、train_loss、val_loss、train_accuracy、val_accuracy；"
                         "accuracy 使用 0–1 的有限数值。") from exc
    epochs = [r.epoch for r in records]
    if epochs != sorted(set(epochs)):
        raise ValueError("epoch 必须严格递增，不能重复")
    best_accuracy = max(records, key=lambda r: r.val_accuracy)
    best_loss = min(records, key=lambda r: r.val_loss)
    last = records[-1]
    gap = last.train_accuracy - last.val_accuracy
    rising_loss = last.val_loss > best_loss.val_loss * 1.15
    signals = []
    if gap > 0.08 and rising_loss:
        signals.append("出现过拟合信号：末轮泛化差距 > 0.08，且验证损失比最低值高 > 15%。")
    if last.train_loss > records[0].train_loss:
        signals.append("训练损失高于首轮，建议检查学习率、输入归一化和标签。")
    if not signals:
        signals.append("未触发当前规则的明显过拟合/发散信号；仍需结合数据拆分与多次实验判断。")
    return {
        "epochs": len(records), "best_accuracy_epoch": best_accuracy.epoch,
        "best_val_accuracy": best_accuracy.val_accuracy, "best_loss_epoch": best_loss.epoch,
        "best_val_loss": best_loss.val_loss, "last_generalization_gap": gap,
        "train_loss_change": last.train_loss - records[0].train_loss,
        "val_loss_change": last.val_loss - records[0].val_loss,
        "signals": signals, "records": [r.model_dump() for r in records],
        "source_sha256": digest(content.encode()),
        "metrics_origin": "Python computation; input log is user-provided, not independently audited",
    }
