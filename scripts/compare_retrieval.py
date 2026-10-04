"""Separate retrieval experiment; not attributed to either execution harness."""
import hashlib
import statistics
import time

from backend.neural_retrieval import NeuralIndex
from backend.retrieval import HybridIndex
from backend.store import Store
from scripts.compare_harnesses import OUTPUT, write_json
from scripts.evaluate import RETRIEVAL


def main():
    store = Store(OUTPUT / "corpus.db")
    neural = NeuralIndex(store, OUTPUT.parent / "neural")
    indexes = {"lsa": HybridIndex(store), "neural": neural}
    records = []
    try:
        for index in indexes.values():
            index.search("warmup", ["resnet"])  # Exclude model loading/index building.
        for repetition in range(3):
            for case_id, paper, query, pages in RETRIEVAL:
                order = list(indexes) if repetition % 2 else list(indexes)[::-1]
                for name in order:
                    start = time.monotonic()
                    hits = indexes[name].search(query, [paper], 6)
                    elapsed = time.monotonic() - start
                    rank = next((i for i, h in enumerate(hits, 1) if h["page"] in pages), None)
                    records.append({
                        "id": case_id, "repetition": repetition + 1, "retriever": name,
                        "query": query, "paper_id": paper, "reference_pages": pages,
                        "hits": hits, "first_relevant_rank": rank,
                        "hit_at_6": rank is not None, "rr_at_6": 1 / rank if rank else 0,
                        "elapsed_seconds": elapsed,
                    })
                    print(f"{name} {case_id} r{repetition+1} rank={rank} {elapsed:.3f}s", flush=True)
        summary = {}
        for name in indexes:
            rows = [r for r in records if r["retriever"] == name]
            summary[name] = {
                "queries": len(RETRIEVAL), "repetitions": 3,
                "hit_at_6": statistics.mean(r["hit_at_6"] for r in rows),
                "mrr_at_6": statistics.mean(r["rr_at_6"] for r in rows),
                "p50_seconds": statistics.median(r["elapsed_seconds"] for r in rows),
            }
        write_json(OUTPUT / "retrieval.json", {
            "basis": "8固定英文查询、限定论文、预设物理页范围；非段落级人评或问答准确率。",
            "rows": records, "summary": summary,
            "weights": [{"path": str(p.relative_to(OUTPUT.parent)),
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                        for p in (OUTPUT.parent / "neural" / "models").rglob("*.onnx")],
        })
        print(summary)
    finally:
        neural.close()


if __name__ == "__main__":
    main()
