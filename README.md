# Vision Scholar

An evidence-grounded computer vision research workspace with an **independent agent execution loop**, bounded context views, ordered tool scheduling, and reproducible CPU studies.

[中文说明](README.zh-CN.md) · [Engineering contributions](CONTRIBUTIONS.md) · [Measured results](docs/research-evaluation.md) · [Full prompts](docs/research-prompts.md) · [Demo](docs/demo.md)

![Real study interface](evidence/demo/desktop-study.png)

Ask questions about papers, inspect real source lines, create an immutable experiment plan, run it, and trace the answer back to paper pages or saved predictions. React/TypeScript provides the workspace; FastAPI and LangGraph route tasks; V3 `ResearchEngine` owns the inner model/tool loop.

## What is independently implemented?

- **Budgeted request views:** retain complete local history, select whole user turns, preserve explicit user corrections, and externalize large tool results only when the request exceeds its budget. Each view records its choices and content hashes.
- **Ordered side effects:** concurrent read-only tools, bounded concurrency, and sequential write barriers preserve model-declared call order. Each call carries its ID and execution interval.
- **Verifiable artifacts:** session-scoped SHA256 artifacts can be paginated or searched; corrupt or cross-session references are rejected.
- **Research lifecycle:** immutable plan → killable CPU worker → paired multi-seed predictions → independently recomputed metrics → downloadable evidence. Completed identical plans reuse their result.

These are original engineering implementations, **not claims of novel algorithms**. OpenHarness supplies shared DTOs, transport and tool interfaces. V1/V2 use its loop; V3 does not call or inherit that loop. V0 is a reconstructed LangGraph baseline. See the [attribution matrix](CONTRIBUTIONS.md).

## Evidence, including failures

One fixed model, temperature 0, thinking disabled. All development attempts and heldout failures are retained.

| Frozen experiment | V2 | V3 full history | V3 budget |
|---|---:|---:|---:|
| Context exact-marker checks | 32/32 | 32/32 | 32/32 |
| Mean total provider tokens | 7,427 | 25,705 | 6,577 |
| Normal-task engineering checks | 14/16 | — | 14/16 |

V3 used **74.4% fewer total tokens than its full-history ablation** in these synthetic context tasks, and 11.4% fewer than V2. This includes recovery costs: the full-history model sometimes attempted unnecessary invalid artifact reads. It is not a universal compression rate or a demonstrated increase in factual accuracy.

The strict normal-task failures omitted a full plan ID; that grader requirement was not explicit in the corresponding prompt. Semantic review also found corrupted quotations and a mistaken confidence-interval claim. Source validation does not establish factual entailment.

Scheduler replay: serial **40/40**, direct parallel **35/40**, read-parallel/write-barrier **40/40**. Delays were artificial. The real digits study compares two SVM feature pipelines over three seeds and three test conditions; both reach **98.15% clean accuracy**, but noise accuracy falls near chance. Negative results are published.

Read the [full report](docs/research-evaluation.md), [Chinese workbook](evidence/v3/原创机制与实验证据.xlsx), [164 model traces](evidence/v3/trials/), [semantic audit](evidence/v3/semantic-audit.json), and [prediction evidence](evidence/study/). These are project-specific engineering checks, not human gold or an external benchmark.

Clean-directory validation passed: **84 Python tests**, frontend build, independent metric recomputation, and retraining with identical prediction/split/corruption bytes. Two new desktop/mobile research UI tests passed against a real model. [Validation record](evidence/v3/delivery-verification.json); GitHub CI is provided but has not yet run remotely.

## Run locally

Requirements: Git, Python 3.11+, [uv](https://docs.astral.sh/uv/), Node 20.19+ or 22.12+. Validated locally with Python 3.12 and Node 24 on macOS.

```bash
./setup.sh
./start.sh
# Open http://127.0.0.1:8765
```

Dependencies are locked, including OpenHarness commit `9b2efd795c6aa09f88b0c257d269a9e518da6ae7`. No sibling checkout is required. First installation requires network access. The npm lock currently references the public npmmirror registry.

Import the four paper presets from the library (ResNet, ViT, CLIP, DETR). PDFs download from arXiv and stay local. The study works offline without paper downloads or model credentials.

For an actual agent, copy `.env.example` to `.env`, select `VS_PROVIDER=openai` or `anthropic`, and set your own model, base URL and key. Then restart. `offline` keeps the library and CPU experiments usable; explicit `mock` runs a labeled scripted demo. Credentials and local databases are ignored by Git.

## Reproduce without an API key

```bash
uv run --extra dev pytest -q
uv run python -m scripts.verify_study evidence/study
uv run python -m scripts.reproduce_study
uv run python -m scripts.benchmark_scheduler
```

Report regeneration requires `openpyxl` and `matplotlib` (available in the `benchmark` extra):

```bash
uv sync --locked --extra dev --extra benchmark
uv run --extra benchmark python -m scripts.report_research
```

For real model comparisons, see [experiment commands](docs/reproduction.md). They use your own paid API configuration and require the paper corpus. The normal tests and scientific verification do not call an external model.

## Project map

| Module | Responsibility |
|---|---|
| `backend/research_engine.py` | Independent loop, context assembly, scheduler |
| `backend/artifacts.py` | Session-scoped evidence artifacts |
| `backend/studies.py`, `study_worker.py` | Plans, process lifecycle, real experiments |
| `backend/agent.py`, `harnesses.py` | LangGraph routing, harness selection, app events |
| `backend/papers.py`, `retrieval.py`, `tools.py` | Paper ingestion, hybrid search, domain tools |
| `frontend/src/` | Workspace, citations, research UI, persisted sessions |
| `benchmarks/v3/` | Protocol and implementation freeze |
| `evidence/` | Public results, traces, predictions, screenshots |
| `scripts/verify_study.py` | Independent metric and interval recomputation |

The lightweight retriever uses BM25, LSA and reciprocal-rank fusion. Optional neural retrieval uses Qdrant local, BGE embeddings and a cross-encoder; the small existing regression does not establish a neural advantage. [V0/V1/V2 comparison](docs/harness-comparison.md) is retained as historical evidence and uses a different protocol.

## Limits

Local, single-user, single-worker application. No public authentication, multi-user permissions, GPU scheduling or training checkpoint recovery. Code inspection is restricted to the teaching `examples/` folder. PDF ingestion handles text PDFs, not OCR.

The context budget counts canonical DTO JSON characters, not provider tokens or wire bytes. Read-only declarations are trusted. Arbitrary thread-based side effects cannot be cancelled or made exactly-once; the study subprocess can be terminated. Citation existence checks cannot guarantee truthful answers. The digits experiments are small teaching studies, not large-paper reproductions.

## License and attribution

Project code: [MIT](LICENSE). [HKUDS/OpenHarness](https://github.com/HKUDS/OpenHarness): MIT, pinned dependency. Papers and datasets retain their original rights; no paper PDFs or downloaded model weights are shipped. See [third-party notices](THIRD_PARTY_NOTICES.md).
