"""
Hybrid search engine: BM25 + FAISS + RRF fusion
"""

import os
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss


DATA_DIR = "data"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RRF_ALPHA = 60  
TOP_K = 10


print("Loading BM25 tokenized docs...")
with open(os.path.join(DATA_DIR, "bm25_tokenized.pkl"), "rb") as f:
    tokenized_docs = pickle.load(f)

print("Loading FAISS index...")
faiss_index = faiss.read_index(os.path.join(DATA_DIR, "faiss_hybrid.index"))

print("Loading original documents...")
with open(os.path.join(DATA_DIR, "original_docs.pkl"), "rb") as f:
    original_docs = pickle.load(f)

print("BM25 + FAISS engine ready.")

bm25 = BM25Okapi(tokenized_docs)

encoder = SentenceTransformer(EMBEDDING_MODEL)

def search(query: str, mode: str = "fusion", top_k: int = TOP_K):
    """
    Hybrid search with three modes:
    - bm25: lexical only
    - semantic: dense vector only
    - fusion: RRF over bm25 + semantic
    """
    if mode not in ("bm25", "semantic", "fusion"):
        raise ValueError(f"Unknown mode {mode}")

    n_docs = len(original_docs)
    bm25_scores = np.zeros(n_docs, dtype=np.float64)
    semantic_scores = np.zeros(n_docs, dtype=np.float64)

    if mode in ("bm25", "fusion"):
        tokenized_query = query.split()
        bm25_scores = bm25.get_scores(tokenized_query)

    if mode in ("semantic", "fusion"):
        q_vec = encoder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_vec)
        D, I = faiss_index.search(q_vec, min(top_k * 5, n_docs)) 
        
        for i, (dist, idx) in enumerate(zip(D[0], I[0])):
            if idx >= 0 and idx < n_docs:  
                semantic_scores[idx] = float(dist)

    if mode == "fusion":
        rrf_scores = np.zeros(n_docs, dtype=np.float64)
        
        bm25_ranks = np.argsort(-bm25_scores)
        for r, idx in enumerate(bm25_ranks[:min(top_k * 5, n_docs)]):
            rrf_scores[idx] += 1.0 / (RRF_ALPHA + r)
        
        semantic_ranks = np.argsort(-semantic_scores)
        for r, idx in enumerate(semantic_ranks[:min(top_k * 5, n_docs)]):
            rrf_scores[idx] += 1.0 / (RRF_ALPHA + r)
        
        final_scores = rrf_scores
    elif mode == "bm25":
        final_scores = bm25_scores
    else:
        final_scores = semantic_scores

    top_indices = np.argsort(-final_scores)[:top_k]

    results = []
    for idx in top_indices:
        idx_int = int(idx)  
        score_float = float(final_scores[idx])  
        
        results.append({
            "id": idx_int,
            "text": str(original_docs[idx_int]),  
            "score": score_float
        })
    
    return results


if __name__ == "__main__":
    while True:
        q = input("Enter query (or 'exit'): ").strip()
        if q.lower() == "exit":
            break
        mode = input("Mode (bm25|semantic|fusion): ").strip() or "fusion"
        hits = search(q, mode=mode)
        for i, h in enumerate(hits, 1):
            print(f"{i}. [{h['score']:.4f}] {h['text'][:200]}...")
        print("-" * 50)