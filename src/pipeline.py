from __future__ import annotations

"""Production RAG Pipeline — Ghép M1+M2+M3+M4+M5."""

import os, sys, time

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, get_llm_client, LLM_MODEL


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    latencies = {}

    # Step 1: Load & Chunk (M1)
    t0 = time.perf_counter()
    print("\n[1/4] Chunking documents (Hierarchical M1)...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    latencies["chunking_sec"] = round(time.perf_counter() - t0, 3)
    print(f"  [OK] {len(all_chunks)} chunks from {len(docs)} documents ({latencies['chunking_sec']:.2f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.perf_counter()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5 combined mode)...", flush=True)
    enriched = enrich_chunks(all_chunks, methods=["combined"])
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        latencies["enrichment_sec"] = round(time.perf_counter() - t0, 3)
        print(f"  [OK] Enriched {len(enriched)} chunks ({latencies['enrichment_sec']:.2f}s)", flush=True)
    else:
        latencies["enrichment_sec"] = 0.0
        print("  [Notice] M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.perf_counter()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense Qdrant)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    latencies["indexing_sec"] = round(time.perf_counter() - t0, 3)
    print(f"  [OK] Indexed ({latencies['indexing_sec']:.2f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.perf_counter()
    print("\n[4/4] Loading reranker (CrossEncoder M3)...", flush=True)
    reranker = CrossEncoderReranker()
    reranker._load_model()
    latencies["reranker_load_sec"] = round(time.perf_counter() - t0, 3)
    print(f"  [OK] Reranker ready ({latencies['reranker_load_sec']:.2f}s)", flush=True)

    return search, reranker, latencies


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    client = get_llm_client()
    if client and contexts:
        try:
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Trả lời chính xác, ngắn gọn và CHỈ dựa trên context được cung cấp. Nếu context có nhiều phiên bản, hãy nêu rõ phiên bản mới nhất đang áp dụng."},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
                ]
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  [Notice] LLM generation error: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker, latencies: dict | None = None):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    t_queries_start = time.perf_counter()
    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)
    avg_query_ms = round(((time.perf_counter() - t_queries_start) / len(test_set)) * 1000, 2)

    t0 = time.perf_counter()
    print(f"\n[Eval] Running RAGAS Evaluation (4 metrics x {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    eval_time = round(time.perf_counter() - t0, 3)
    print(f"  [OK] RAGAS done ({eval_time:.2f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        status = "[PASS]" if s >= 0.75 else "[OK]"
        print(f"  {status} {m:<20}: {s:.4f}")

    # Latency Breakdown Report (Bonus)
    print("\n" + "=" * 60)
    print("LATENCY BREAKDOWN REPORT")
    print("=" * 60)
    if latencies:
        print(f"  1. Chunking Time (M1)       : {latencies.get('chunking_sec', 0):.3f}s")
        print(f"  2. Enrichment Time (M5)     : {latencies.get('enrichment_sec', 0):.3f}s")
        print(f"  3. Indexing Time (M2)       : {latencies.get('indexing_sec', 0):.3f}s")
        print(f"  4. Reranker Load Time (M3)  : {latencies.get('reranker_load_sec', 0):.3f}s")
    print(f"  5. Avg Query Latency (End-to-End): {avg_query_ms:.2f}ms / query")
    print(f"  6. Evaluation Time (M4)     : {eval_time:.3f}s")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker, latencies = build_pipeline()
    evaluate_pipeline(search, reranker, latencies)
    print(f"\nTotal Pipeline Time: {time.time() - start:.1f}s")
