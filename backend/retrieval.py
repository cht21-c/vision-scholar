"""Local, inspectable hybrid search: BM25 + LSA + RRF + lexical reranking."""
from __future__ import annotations

import re
import threading
from collections import Counter

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from backend.store import Store

ALIASES = {
    "残差": "residual shortcut identity mapping",
    "恒等": "identity mapping",
    "退化": "degradation optimization",
    "梯度": "gradient",
    "图像块": "patch embedding",
    "位置编码": "position embedding",
    "注意力": "attention",
    "对比学习": "contrastive learning",
    "零样本": "zero shot",
    "损失": "loss objective",
    "匹配": "bipartite matching",
    "目标检测": "object detection",
    "监督": "supervision",
}


def tokens(text: str) -> list[str]:
    text = text.lower()
    for phrase, english in ALIASES.items():
        if phrase in text:
            text += " " + english
    return [t for t in re.findall(r"[a-z][a-z0-9_-]*|\d+(?:\.\d+)?", text)
            if t not in ENGLISH_STOP_WORDS and len(t) > 1]


class HybridIndex:
    def __init__(self, store: Store):
        self.store = store
        self._signature = None
        self._lock = threading.RLock()
        self.docs: list[dict] = []

    def invalidate(self) -> None:
        with self._lock:
            self._signature = None

    def _refresh(self) -> None:
        signature = tuple((p["id"], p["sha256"]) for p in self.store.papers())
        if signature == self._signature:
            return
        self.docs = self.store.chunks()
        self.counters = [Counter(tokens(d["text"])) for d in self.docs]
        self.lengths = np.array([sum(c.values()) for c in self.counters], dtype=float)
        self.avg_length = float(self.lengths.mean()) if len(self.docs) else 1.0
        df = Counter(t for c in self.counters for t in c)
        self.idf = {t: np.log(1 + (len(self.docs) - n + 0.5) / (n + 0.5))
                    for t, n in df.items()}
        self.vectorizer = self.svd = self.dense = None
        if self.docs:
            self.vectorizer = TfidfVectorizer(
                tokenizer=tokens, token_pattern=None, lowercase=False,
                max_features=18000, sublinear_tf=True,
            )
            matrix = self.vectorizer.fit_transform(d["text"] for d in self.docs)
            dimensions = min(96, matrix.shape[0] - 1, matrix.shape[1] - 1)
            with threadpool_limits(limits=1):
                if dimensions >= 2:
                    self.svd = TruncatedSVD(n_components=dimensions, random_state=42)
                    self.dense = normalize(self.svd.fit_transform(matrix))
                else:
                    self.dense = normalize(matrix.toarray())
        self._signature = signature

    def search(self, query: str, paper_ids: list[str] | None = None,
               limit: int = 6) -> list[dict]:
        if not query.strip() or not 1 <= limit <= 12:
            raise ValueError("搜索词不能为空，结果数量应为 1–12")
        with self._lock, threadpool_limits(limits=1):
            self._refresh()
            if not self.docs:
                return []
            terms = set(tokens(query))
            bm25 = np.zeros(len(self.docs))
            for term in terms:
                frequency = np.array([c.get(term, 0) for c in self.counters], dtype=float)
                denominator = frequency + 1.5 * (0.25 + 0.75 * self.lengths / self.avg_length)
                bm25 += self.idf.get(term, 0) * frequency * 2.5 / np.maximum(denominator, 1e-9)
            vector = self.vectorizer.transform([query])
            dense_query = normalize(
                self.svd.transform(vector) if self.svd else vector.toarray()
            )
            cosine = (self.dense @ dense_query.T).ravel()
            eligible = [i for i, d in enumerate(self.docs)
                        if (not paper_ids or d["paper_id"] in paper_ids)
                        and (bm25[i] > 0 or cosine[i] > 0.08)]
            if not eligible:
                return []
            ranks = {}
            for scores in (bm25, cosine):
                for rank, i in enumerate(sorted(eligible, key=lambda j: -scores[j]), 1):
                    ranks[i] = ranks.get(i, 0) + 1 / (60 + rank)
            scored = []
            for i in eligible:
                coverage = len(terms & self.counters[i].keys()) / max(len(terms), 1)
                score = ranks[i] + 0.018 * coverage
                scored.append((score, i, coverage))
            scored.sort(reverse=True)
            # Keep exact duplicate/near duplicate chunks from dominating results.
            selected = []
            for score, i, coverage in scored:
                d = self.docs[i]
                same_page = sum(r["paper_id"] == d["paper_id"] and r["page"] == d["page"]
                                for r in selected)
                if same_page >= 2:
                    continue
                selected.append({
                    **d, "citation": f"[{d['id']}]",
                    "scores": {"bm25": round(float(bm25[i]), 4),
                               "lsa_cosine": round(float(cosine[i]), 4),
                               "lexical_coverage": round(coverage, 4),
                               "fused": round(score, 5)},
                })
                if len(selected) == limit:
                    break
            return selected


CITATION_PATTERN = re.compile(r"\[([a-zA-Z0-9_.-]+:p\d+:c\d+)\]")
CODE_PATTERN = re.compile(r"\[(code:[a-zA-Z0-9_./-]+:L\d+-L\d+)\]")


def verify_citations(answer: str, evidence: dict[str, dict], store: Store) -> dict:
    ids = list(dict.fromkeys(CITATION_PATTERN.findall(answer) + CODE_PATTERN.findall(answer)))
    valid, invalid = [], []
    for citation in ids:
        item = evidence.get(citation)
        if citation.startswith("code:"):
            if item:
                valid.append(item)
            else:
                invalid.append({"id": citation, "reason": "该代码范围未在本轮读取"})
            continue
        exists = store.one("SELECT id FROM chunks WHERE id=?", (citation,))
        if not exists:
            invalid.append({"id": citation, "reason": "引用不存在"})
        elif not item:
            invalid.append({"id": citation, "reason": "该片段未在本轮读取"})
        else:
            valid.append(item)
    return {
        "status": "invalid" if invalid else ("verified" if valid else "no_citations"),
        "valid": valid, "invalid": invalid,
        "scope": "核验原文存在及本轮读取记录；不等同于结论的语义正确性。",
    }
