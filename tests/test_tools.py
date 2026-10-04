import io
import json

import httpx
import numpy as np
import pytest
from pydantic import ValidationError
from pypdf import PdfWriter

from backend.experiments import ExperimentInput, analyze_log, execute_experiment
from backend.papers import (
    MAX_PDF_BYTES,
    fetch_arxiv,
    parse_pdf,
    validate_arxiv_id,
    validate_arxiv_url,
)
from backend.retrieval import verify_citations
from backend.tools import CodeInput, LogInput, NoteInput, ResearchTools, inspect_code
from examples.vision_ops import patchify, scaled_dot_product_attention


def test_pdf_ingestion_deduplicates_and_retains_page_evidence(context, pdf_bytes):
    _, store, library, index = context
    duplicate = library.ingest(pdf_bytes, paper_id="different-id")
    assert duplicate["id"] == "fixture"
    assert len(store.papers()) == 1
    assert store.paper("fixture")["page_count"] == 2
    assert "Identity shortcut" in library.page("fixture", 1)["text"]
    result = index.search("identity residual shortcut", ["fixture"], 1)[0]
    assert result["page"] == 1
    assert result["id"] == "fixture:p1:c1"
    assert result["text"] in library.page("fixture", 1)["text"]
    assert index.search("xyzzznonsenseword") == []


@pytest.mark.parametrize("payload", [b"not a pdf", b"%PDF-broken", b"x" * (MAX_PDF_BYTES + 1)])
def test_invalid_pdf(payload):
    with pytest.raises(ValueError):
        parse_pdf(payload)


def test_blank_and_encrypted_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(ValueError, match="OCR"):
        parse_pdf(buffer.getvalue())
    writer.encrypt("test")
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(ValueError, match="加密"):
        parse_pdf(buffer.getvalue())


@pytest.mark.parametrize("value", ["../file", "https://arxiv.org/pdf/2010.11929", "abc", "1234.12"])
def test_invalid_arxiv_id(value):
    with pytest.raises(ValueError):
        validate_arxiv_id(value)


@pytest.mark.parametrize("url", [
    "http://arxiv.org/pdf/2010.11929", "https://arxiv.org.evil.test/",
    "https://127.0.0.1/", "https://arxiv.org:8443/", "https://user:pass@arxiv.org/",
])
def test_external_download_scope(url):
    with pytest.raises(ValueError):
        validate_arxiv_url(url)


async def test_arxiv_redirect_revalidated(monkeypatch):
    original = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(
        302, headers={"location": "https://127.0.0.1/private"},
    ))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        **kwargs, transport=transport,
    ))
    with pytest.raises(ValueError, match="HTTPS"):
        await fetch_arxiv("https://arxiv.org/pdf/2010.11929")


async def test_arxiv_network_failure(monkeypatch):
    original = httpx.AsyncClient

    def fail(request):
        raise httpx.ConnectError("offline", request=request)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(
        **kwargs, transport=httpx.MockTransport(fail),
    ))
    with pytest.raises(httpx.ConnectError):
        await fetch_arxiv("https://arxiv.org/pdf/2010.11929")


def test_page_bounds_and_citation_provenance(context):
    _, store, library, index = context
    with pytest.raises(ValueError):
        library.page("fixture", 3)
    with pytest.raises(KeyError):
        library.page("missing", 1)
    hit = index.search("residual identity", limit=1)[0]
    evidence = {hit["id"]: hit}
    valid = verify_citations("See [fixture:p1:c1]", evidence, store)
    assert valid["status"] == "verified"
    assert len(valid["valid"]) == 1
    missing = verify_citations("[fixture:p99:c1]", evidence, store)
    assert missing["status"] == "invalid"
    unread = verify_citations("[fixture:p2:c1]", evidence, store)
    assert unread["invalid"][0]["reason"] == "该片段未在本轮读取"
    assert verify_citations("No evidence available", {}, store)["status"] == "no_citations"


def test_code_scope_and_symbol_bounds(tmp_path):
    result = inspect_code("vision_ops.py", "patchify")
    assert "def patchify" in result["text"]
    assert result["start"] < result["end"]
    assert f":L{result['start']}-L{result['end']}" in result["citation"]
    with pytest.raises(ValueError):
        inspect_code("../backend/config.py")
    with pytest.raises(ValueError):
        inspect_code("/etc/passwd")
    with pytest.raises(ValueError, match="符号不存在"):
        inspect_code("vision_ops.py", "imaginary")
    root = tmp_path / "examples"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("secret = 1")
    (root / "link.py").symlink_to(outside)
    with pytest.raises(ValueError):
        inspect_code("link.py", root=root)


def test_schema_rejects_unregistered_parameters():
    with pytest.raises(ValidationError):
        ExperimentInput(kind="shell", command="whoami")
    with pytest.raises(ValidationError):
        ExperimentInput(kind="attention", image_size=7, patch_size=2)
    with pytest.raises(ValidationError):
        ExperimentInput(kind="attention", image_size=32, patch_size=1)
    with pytest.raises(ValidationError):
        CodeInput(path="vision_ops.py", exec=True)


def log_records():
    return [
        {"epoch": 1, "train_loss": 0.8, "val_loss": 0.9,
         "train_accuracy": 0.7, "val_accuracy": 0.68},
        {"epoch": 2, "train_loss": 0.4, "val_loss": 0.5,
         "train_accuracy": 0.85, "val_accuracy": 0.82},
        {"epoch": 3, "train_loss": 0.1, "val_loss": 0.7,
         "train_accuracy": 0.98, "val_accuracy": 0.78},
    ]


def test_log_metrics_and_overfitting_signal():
    result = analyze_log(json.dumps(log_records()))
    assert result["best_accuracy_epoch"] == 2
    assert result["best_loss_epoch"] == 2
    assert result["last_generalization_gap"] == pytest.approx(0.2)
    assert "过拟合" in result["signals"][0]
    csv = "epoch,train_loss,val_loss,train_accuracy,val_accuracy\n1,0.8,0.9,0.7,0.68\n2,0.4,0.5,0.85,0.82"
    assert analyze_log(csv)["best_accuracy_epoch"] == 2


@pytest.mark.parametrize("content", [
    "[]", "{}", "hello", '[{"epoch":1}]',
    '[{"epoch":1,"train_loss":NaN,"val_loss":0.5,"train_accuracy":0.9,"val_accuracy":0.8}]',
])
def test_invalid_log(content):
    with pytest.raises(ValueError):
        analyze_log(content)


def test_reject_duplicate_epochs_and_nonfinite_metrics():
    rows = log_records()
    rows[1]["epoch"] = 1
    with pytest.raises(ValueError, match="严格递增"):
        analyze_log(json.dumps(rows))
    rows = log_records()
    rows[1]["val_accuracy"] = float("nan")
    with pytest.raises(ValueError):
        analyze_log(json.dumps(rows))


def test_real_experiment_reproducibility(context):
    config, store, _, _ = context
    args = ExperimentInput(seed=42, pca_components=24)
    first = execute_experiment(args, store, config.data_dir / "experiments")
    second = execute_experiment(args, store, config.data_dir / "experiments")
    a, b = first["result"], second["result"]
    assert a["split"] == {"train": 1077, "validation": 360, "test": 360}
    assert a["dataset_sha256"] == b["dataset_sha256"]
    assert a["split_sha256"] == b["split_sha256"]
    assert a["results"] == b["results"]
    for result in a["results"]:
        assert np.sum(result["confusion_matrix"]) == a["split"]["test"]
        assert result["test_accuracy"] >= 0.9
    assert len(a["training_log"]) == 16
    assert first["id"] != second["id"]


def test_stored_experiment_log_retains_provenance(context):
    config, store, library, index = context
    experiment = execute_experiment(ExperimentInput(), store, config.data_dir / "experiments")
    session = store.create_session()
    run = store.create_run(session["id"], "分析实验日志", [], "experiment")
    tools = ResearchTools(store, library, index, config.data_dir, run)
    computed = tools.log(LogInput(experiment_id=experiment["id"]))
    assert computed["experiment_id"] == experiment["id"]
    assert computed["seed"] == 42
    assert computed["split"] == {"train": 1077, "validation": 360, "test": 360}
    assert computed["code_sha256"] == experiment["result"]["code_sha256"]
    assert "stored experiment log" in computed["metrics_origin"]
    assert "非 SVC 日志" in computed["log_source"]
    assert "user-provided" in tools.log(LogInput(content=json.dumps(log_records())))["metrics_origin"]


def test_patch_order_and_attention_math():
    image = np.arange(16).reshape(4, 4, 1)
    patches = patchify(image, 2)
    np.testing.assert_array_equal(patches[0], [0, 1, 4, 5])
    np.testing.assert_array_equal(patches[3], [10, 11, 14, 15])
    q = np.zeros((4, 2))
    values = np.arange(8).reshape(4, 2)
    output, weights = scaled_dot_product_attention(q, q, values)
    np.testing.assert_allclose(weights, 0.25)
    np.testing.assert_allclose(output, np.tile(values.mean(axis=0), (4, 1)))


async def test_specialist_registry_and_notes_gating(context):
    config, store, library, index = context
    session = store.create_session()
    run = store.create_run(session["id"], "解释残差，不要保存", ["fixture"], "paper")
    tools = ResearchTools(store, library, index, config.data_dir, run)
    names = {tool.name for tool in tools.registry("paper").list_tools()}
    assert "search_papers" in names
    assert "run_experiment" not in names
    assert "bash" not in names
    with pytest.raises(ValueError):
        tools.save(NoteInput(title="decision", content="don't save this"))
    with pytest.raises(ValueError):
        await tools.experiment(ExperimentInput())
    assert store.notes() == []
