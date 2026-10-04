# Engineering contribution and attribution

Originality here means independently designed and implemented engineering components. It does not mean academic novelty, an algorithmic invention, or zero dependency reuse.

| Component | Project contribution | Reuse | Evidence |
|---|---|---|---|
| V0 baseline | Reconstructed state loop in LangGraph | LangGraph, shared tools/transport | `backend/legacy_agent.py`; historical comparison |
| V1 integration | Domain integration and app lifecycle | OpenHarness QueryEngine loop | `backend/harnesses.py` |
| V2 upgrades | Quoted user decisions with source hashes, citation subranges, per-run note deduplication | OpenHarness loop and compaction | `backend/upgrades.py`; retained failure traces |
| V3 loop | Independent request/response/tool iteration, cancellation propagation | OpenHarness messages/events/request DTOs and API clients | `backend/research_engine.py`; frozen source SHA |
| V3 context | Whole-user-turn grouping, mandatory current input/decisions, budget checks, request-view audit | V2 decision extractor; deterministic relevance heuristics | Full/budget ablation, randomized markers, tamper/pair tests |
| V3 scheduler | Bounded read parallelism, ordered write barriers, call-ID timing | Tool read-only declaration and schema validation | 120 identical-call replays across three modes |
| Tool artifacts | Session boundary, content addressing, integrity checks, bounded rereads | SHA256 and atomic filesystem replacement | Cross-session/tamper tests; artifact-lookup trials |
| Study lifecycle | Immutable plans, DB claim, subprocess worker, status recovery, result reuse | SQLite, subprocess primitives | Lifecycle tests; real browser flow |
| Scientific evidence | Train/validation/test partition, paired corruptions, per-sample records, independent metric verifier | sklearn digits, SVM/PCA, standard paired bootstrap | `evidence/study/`, 18 recomputed metrics |
| Frontend and backend | Routing, SSE UI, citation reader, study panel, export | React, FastAPI, LangGraph, plotting libraries | Desktop/mobile UI tests and screenshots |

`ResearchEngine` neither inherits `QueryEngine`/`LegacyEngine` nor delegates its execution loop, scheduler or context assembly to them. It reuses shared interface types and transport instead of claiming those interfaces were newly invented. Domain tools and prompts are held constant in the V2/V3 task comparison.

## Problems that drove the changes

1. Earlier compression could lose user corrections. V2 introduced a quoted decision ledger; V3 retained it while making each bounded request view inspectable.
2. Unrestricted tool parallelism could read before a preceding write finished. V3 gave the ordered tool-call list explicit write-barrier semantics.
3. A plausible experiment answer was insufficient evidence. The study workflow binds immutable parameters to saved splits, predictions and hashes, and computes tables without relying on model-generated numbers.
4. Development exposed long hash transcription errors. Run-local short references map exactly to immutable study IDs, without fuzzy guessing.
5. Eagerly externalizing every large tool result increased avoidable rereads. V3 keeps inline evidence when the full request already fits.

The [development log](docs/v3-development.md) preserves failed attempts. The [frozen evaluation](docs/research-evaluation.md) includes negative and neutral results, including no demonstrated normal-task success gain. Public evidence hashes identify the evaluated code, not an untracked “latest” version.

## Deliberate limits

Decision extraction remains heuristic; arbitrary natural-language preferences may be missed. Read-only metadata is trusted. No general prompt-injection defense, exactly-once guarantee, GPU orchestration, or semantic-entailment verifier is claimed. The statistical methods are standard, and digits results do not establish large-scale CV performance.

For dependency and dataset rights see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Personal resume drafts are excluded from the public repository.
