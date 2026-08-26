from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    if not text:
        return ""
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception:
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = []
        for chunk in chunks:
            tokens = segment_vietnamese(chunk.get("text", "")).lower().split()
            self.corpus_tokens.append(tokens)

        if self.corpus_tokens:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents:
            return []

        tokenized_query = segment_vietnamese(query).lower().split()
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        # Lấy các kết quả có điểm > 0, nếu không có thì lấy top điểm cao nhất
        positive_indices = [i for i, s in enumerate(scores) if s > 0]
        if positive_indices:
            top_indices = sorted(positive_indices, key=lambda i: scores[i], reverse=True)[:top_k]
        else:
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for i in top_indices:
            doc = self.documents[i]
            results.append(SearchResult(
                text=doc.get("text", ""),
                score=float(scores[i]),
                metadata=doc.get("metadata", {}),
                method="bm25"
            ))
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2.0)
            client.get_collections()
            self.client = client
        except Exception:
            # Fallback to in-memory Qdrant client when Docker is unavailable
            self.client = QdrantClient(location=":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        if not chunks:
            return

        from qdrant_client.models import Distance, VectorParams, PointStruct

        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )

        texts = [c["text"] for c in chunks]
        vectors = self._get_encoder().encode(texts, batch_size=32, show_progress_bar=False)

        points = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            payload = {**chunk.get("metadata", {}), "text": chunk["text"]}
            points.append(PointStruct(
                id=i,
                vector=vec.tolist(),
                payload=payload
            ))

        self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if not query:
            return []

        query_vector = self._get_encoder().encode(query).tolist()
        try:
            # qdrant-client >= 1.9 query_points()
            response = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=top_k
            )
            pts = response.points
        except Exception:
            # fallback to search() if query_points is unsupported
            pts = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=top_k
            )

        results = []
        for pt in pts:
            payload = pt.payload or {}
            text = payload.get("text", "")
            meta = {k: v for k, v in payload.items() if k != "text"}
            results.append(SearchResult(
                text=text,
                score=float(pt.score),
                metadata=meta,
                method="dense"
            ))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                            top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank + 1)."""
    rrf_scores = {}  # text -> {"score": float, "metadata": dict}

    for result_list in results_list:
        for rank, result in enumerate(result_list):
            doc_text = result.text
            if doc_text not in rrf_scores:
                rrf_scores[doc_text] = {
                    "score": 0.0,
                    "metadata": result.metadata
                }
            rrf_scores[doc_text]["score"] += 1.0 / (k + rank + 1)

    sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1]["score"], reverse=True)[:top_k]

    return [
        SearchResult(
            text=text,
            score=float(data["score"]),
            metadata=data["metadata"],
            method="hybrid"
        )
        for text, data in sorted_docs
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
