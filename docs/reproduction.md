# Reproduction guide

## Offline engineering and scientific verification

Install with `./setup.sh`. It resolves the Git-pinned upstream dependency and locked packages without requiring an adjacent OpenHarness checkout. After installation:

```bash
uv run --extra dev pytest -q
uv run python -m scripts.verify_study evidence/study
uv run python -m scripts.reproduce_study --data-dir data/my-study
uv run python -m scripts.benchmark_scheduler
```

`verify_study` uses saved predictions and labels to recompute accuracy, macro F1, seed means/standard deviations and paired confidence intervals. It checks split coverage, pairing, data/feature hashes and plan identity. `reproduce_study` retrains models in a separate directory. Feed its printed evidence directory to `verify_study` to verify that new run.

The plan records Python/numpy/sklearn versions and code/data hashes; a different runtime can produce a different plan ID. Preserve the supplied evidence unchanged. This small study does not assert bitwise reproducibility of every numerical result across hardware/BLAS versions.

## Real model experiments

These commands incur model API usage. Run the server once, import the four paper presets in the library, then stop the server or leave it idle. The benchmark copies the paper corpus into isolated trial databases; it does not share application notes.

Create a local, ignored file `data/private/benchmark.env`:

```dotenv
BENCH_API_KEY=your_key
BENCH_BASE_URL=https://your-compatible-provider.example/v1
BENCH_MODEL=your_model_or_endpoint
```

Use a model endpoint supporting OpenAI tool calls. The archived run used `doubao-seed-2-0-lite-260428`, temperature 0 and thinking disabled. A different model/provider is a new experiment, not a reproduction of identical model behavior.

```bash
uv sync --locked --extra dev --extra benchmark
# Data-dependent development checks
uv run --extra benchmark python -m scripts.benchmark_research --suite tasks --split dev --repeats 1

# Original frozen heldout protocol; the published cases are now visible to you
uv run --extra benchmark python -m scripts.benchmark_research --suite tasks --split heldout --repeats 2
uv run --extra benchmark python -m scripts.benchmark_research --suite context --split heldout --repeats 2 --arms upgraded research_full research
```

Each batch uses concurrency 4 and prints progress/ETA. Resume the same command after interruption: completed trial files are retained. A changed manifest is rejected; use a new `--batch` label for genuinely separate experiments, retaining previous failures. Do not call repeated tuning on these published cases an unseen heldout test.

Frozen task prompts, seeds and assertions are in `benchmarks/v3/protocol.json`; implementation hashes are in `implementation-freeze.json`. The full synthetic-history generator was frozen after development and before heldout execution, not before all implementation. Normal tasks and context trials deliberately use the same domain tools across their compared arms.

## Rebuild the public report

This uses the repository's existing public data and no API:

```bash
uv run --extra benchmark python -m scripts.report_research
```

Outputs: `docs/research-evaluation.md`, `docs/research-prompts.md`, `evidence/v3/summary.json`, Excel, PNG and SVG. The workbook has Chinese headers, blue header fill/white text, frozen first rows and filters.

`scripts.export_public_evidence` is the maintainer export step for the archived workspace. It requires the local source results, UI-study verification and private config used by that run to redact exact private values. It is not needed to read or reproduce the already-public report. Exported traces retain actual requests/responses; private transport model identifiers and absolute local paths are replaced. A manifest records original and export hashes. No database, paper PDF, downloaded neural weights, API key or personal resume is included.

## Browser demonstration

Start `./start.sh` with your model configured and paper presets imported.

```bash
cd frontend
npx playwright install chromium
npm run test:e2e -- research.spec.ts
```

The research test creates the default plan, executes or reuses it, compares six table rows to backend results, downloads predictions and checks SHA256, calls the real model from the UI, verifies `read_study` without `run_study`, and checks Markdown export. It runs in desktop and mobile viewports. Outputs go to ignored `data/v3/demo/`.

The older `workspace.spec.ts` suite additionally needs archived first-round evaluation reports; it is not part of offline CI. CI does not call paid APIs or assume preexisting local reports.

## Runtime constraints

Use one application worker per data directory. Do not start another application process against the same SQLite databases. The API listens on loopback by default; this repository is not configured as an authenticated internet service.

Context cost is a character count over a canonical DTO envelope containing the system prompt, tool schemas and messages. Provider usage is recorded separately. The benchmark's total-token comparison includes all observed attempts and artifact rereads; pricing and cache hit rates vary by provider.
