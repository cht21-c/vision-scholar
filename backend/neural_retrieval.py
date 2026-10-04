"""Optional resume-faithful Qdrant + neural dense/BM25 + cross-encoder retrieval."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from backend.retrieval import HybridIndex, tokens

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class NeuralIndex(HybridIndex):
    method = "Qdrant + BGE-small-en-v1.5 + BM25 + RRF + MiniLM cross-encoder"

    def __init__(self, store, directory: Path):
        super().__init__(store)
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.embedding = self.reranker = self.qdrant = None
        self._neural_signature = None

    def close(self):
        if self.qdrant is not None:
            self.qdrant.close()

    def _refresh_neural(self):
        from fastembed import TextEmbedding
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        from qdrant_client import QdrantClient, models
        super()._refresh()
        if self._neural_signature == self._signature:
            return
        if self.embedding is None:
            cache = str(self.directory / "models")
            self.embedding = TextEmbedding(EMBED_MODEL, cache_dir=cache, threads=2)
            self.reranker = TextCrossEncoder(RERANK_MODEL, cache_dir=cache, threads=2)
            self.qdrant = QdrantClient(path=str(self.directory / "qdrant"))
        signature = hashlib.sha256(json.dumps(
            [EMBED_MODEL, self._signature], sort_keys=True).encode()).hexdigest()
        manifest_path = self.directory / "index.json"
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if (previous.get("signature") != signature
                or not self.qdrant.collection_exists("paper_chunks")):
            if self.qdrant.collection_exists("paper_chunks"):
                self.qdrant.delete_collection("paper_chunks")
            self.qdrant.create_collection("paper_chunks", vectors_config=models.VectorParams(
                size=384, distance=models.Distance.COSINE))
            vectors = list(self.embedding.embed([d["text"] for d in self.docs], batch_size=32))
            if vectors:
                self.qdrant.upsert("paper_chunks", points=[
                    models.PointStruct(id=str(uuid5(NAMESPACE_URL, d["id"])),
                                       vector=v.tolist(),
                                       payload={"chunk_id": d["id"], "paper_id": d["paper_id"]})
                    for d, v in zip(self.docs, vectors, strict=True)
                ])
            manifest_path.write_text(json.dumps({
                "signature": signature, "embedding": EMBED_MODEL, "reranker": RERANK_MODEL,
                "documents": len(self.docs), "dimensions": 384,
                "truncation_note": "Both English models truncate at their tokenizer limits; "
                                   "PDF chunks are not guaranteed to fit. Tested separately.",
            }, indent=2))
        self._neural_signature = self._signature

    def search(self, query: str, paper_ids: list[str] | None = None, limit: int = 6):
        from qdrant_client import models
        if not query.strip() or not 1 <= limit <= 12:
            raise ValueError("搜索词不能为空，结果数量应为1–12")
        with self._lock:
            self._refresh_neural()
            if not self.docs:
                return []
            q = next(self.embedding.query_embed(query))
            query_filter = models.Filter(must=[models.FieldCondition(
                key="paper_id", match=models.MatchAny(any=paper_ids))]) if paper_ids else None
            dense = self.qdrant.query_points(
                "paper_chunks", query=q.tolist(), query_filter=query_filter, limit=24,
            ).points
            terms = set(tokens(query))
            scores = np.zeros(len(self.docs))
            for term in terms:
                frequency = np.array([c.get(term, 0) for c in self.counters], dtype=float)
                denominator = frequency + 1.5 * (0.25 + 0.75 * self.lengths / self.avg_length)
                scores += self.idf.get(term, 0) * frequency * 2.5 / np.maximum(denominator, 1e-9)
            eligible = [i for i, d in enumerate(self.docs)
                        if not paper_ids or d["paper_id"] in paper_ids]
            lexical = sorted(eligible, key=lambda i: -scores[i])[:24]
            ranking, dense_scores = {}, {}
            for rank, point in enumerate(dense, 1):
                cid = point.payload["chunk_id"]
                ranking[cid] = 1 / (60 + rank)
                dense_scores[cid] = point.score
            for rank, i in enumerate(lexical, 1):
                cid = self.docs[i]["id"]
                ranking[cid] = ranking.get(cid, 0) + 1 / (60 + rank)
            by_id = {d["id"]: (i, d) for i, d in enumerate(self.docs)}
            candidates = [by_id[cid] for cid in sorted(ranking, key=lambda c: -ranking[c])[:24]]
            reranked = list(self.reranker.rerank(query, [d["text"] for _, d in candidates],
                                                batch_size=16))
            results = []
            for rank in np.argsort(reranked)[::-1]:
                i, doc = candidates[rank]
                if sum(d["paper_id"] == doc["paper_id"] and d["page"] == doc["page"]
                       for d in results) >= 2:
                    continue
                results.append({**doc, "citation": f"[{doc['id']}]",
                                "scores": {"bm25": float(scores[i]),
                                           "dense": dense_scores.get(doc["id"]),
                                           "rrf": ranking[doc["id"]],
                                           "cross_encoder": float(reranked[rank])}})
                if len(results) == limit:
                    break
            return results
