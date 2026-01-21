#!/usr/bin/env python3
"""
Temporary script to tune RRF parameters on 50 validation queries only.
This properly tests k-values and weight combinations before applying to test set.
"""

import os
import sys
from collections import defaultdict
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import CrossEncoder
import torch
import re

# === Configuration ===
QUERIES_PATH = "files/queriesROBUST.txt"
QRELS_PATH = "files/qrels_50_Queries"
VAL_QIDS = [str(i) for i in range(301, 351)]  # 50 validation queries

def load_queries(path):
    """Load queries from file."""
    queries = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                qid, query = parts[0], parts[1]
                queries[qid] = query
    return queries

def load_qrels(path):
    """Load qrels from file."""
    qrels = defaultdict(dict)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                if rel > 0:
                    qrels[qid][docid] = rel
    return qrels

def compute_ap(ranked_list, relevant_docs):
    """Compute Average Precision for a single query."""
    if not relevant_docs:
        return 0.0
    
    hits = 0
    sum_precisions = 0.0
    
    for rank, (docid, _) in enumerate(ranked_list, 1):
        if docid in relevant_docs:
            hits += 1
            sum_precisions += hits / rank
    
    return sum_precisions / len(relevant_docs)

def compute_map(results, qrels):
    """Compute Mean Average Precision."""
    aps = []
    for qid, ranked_list in results.items():
        if qid in qrels:
            ap = compute_ap(ranked_list, qrels[qid])
            aps.append(ap)
    return sum(aps) / len(aps) if aps else 0.0

def extract_text_robust(raw_content):
    """Extract clean text from SGML document."""
    if not raw_content:
        return ""
    
    text_parts = []
    for tag in ['TEXT', 'HEADLINE', 'TITLE', 'LEADPARA', 'SUMMARY']:
        pattern = f'<{tag}>(.*?)</{tag}>'
        matches = re.findall(pattern, raw_content, re.DOTALL | re.IGNORECASE)
        for match in matches:
            clean = re.sub(r'<[^>]+>', ' ', match)
            clean = re.sub(r'\s+', ' ', clean).strip()
            if clean:
                text_parts.append(clean)
    
    if text_parts:
        return ' '.join(text_parts)
    
    clean = re.sub(r'<[^>]+>', ' ', raw_content)
    return re.sub(r'\s+', ' ', clean).strip()

def weighted_rrf(ranked_lists, k=60, weights=None):
    """Weighted Reciprocal Rank Fusion."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    
    fused = {}
    all_qids = set()
    for results in ranked_lists:
        all_qids.update(results.keys())
    
    for qid in all_qids:
        doc_scores = defaultdict(float)
        for weight, results in zip(weights, ranked_lists):
            if qid not in results:
                continue
            for rank, (docid, _) in enumerate(results[qid], 1):
                doc_scores[docid] += weight / (k + rank)
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        fused[qid] = sorted_docs[:1000]
    
    return fused

def main():
    print("=" * 60)
    print("RRF Parameter Tuning on 50 Validation Queries")
    print("=" * 60)
    
    # Load data
    queries = load_queries(QUERIES_PATH)
    qrels = load_qrels(QRELS_PATH)
    val_queries = {qid: queries[qid] for qid in VAL_QIDS if qid in queries}
    
    print(f"Loaded {len(val_queries)} validation queries")
    print(f"Loaded qrels for {len(qrels)} queries")
    
    # Initialize searcher
    print("\nInitializing Pyserini searcher...")
    searcher = LuceneSearcher.from_prebuilt_index('robust04')
    
    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # === Step 1: Run BM25+RM3 on validation queries ===
    print("\n--- Running BM25+RM3 on 50 validation queries ---")
    searcher.set_bm25(0.7, 0.4)
    searcher.set_rm3(50, 5, 0.5)
    
    bm25_results = {}
    for qid in tqdm(VAL_QIDS, desc="BM25+RM3"):
        if qid not in queries:
            continue
        hits = searcher.search(queries[qid], k=1000)
        bm25_results[qid] = [(h.docid, h.score) for h in hits]
    
    searcher.unset_rm3()
    bm25_map = compute_map(bm25_results, qrels)
    print(f"✓ BM25+RM3 Validation MAP: {bm25_map:.4f}")
    
    # === Step 2: Run Neural Reranking on validation queries ===
    print("\n--- Running Neural Reranking on 50 validation queries ---")
    cross_encoder = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512, device=device)
    
    searcher.set_bm25(0.9, 0.4)
    neural_results = {}
    
    for qid in tqdm(VAL_QIDS, desc="Neural Rerank"):
        if qid not in queries:
            continue
        
        query = queries[qid]
        bm25_hits = searcher.search(query, k=1000)
        
        if not bm25_hits:
            neural_results[qid] = []
            continue
        
        # Rerank top 200
        pairs = []
        doc_ids = []
        for hit in bm25_hits[:200]:
            doc = searcher.doc(hit.docid)
            if doc:
                raw = doc.raw() if hasattr(doc, 'raw') else doc.contents()
                if raw:
                    text = extract_text_robust(raw)[:2000]
                    pairs.append([query, text])
                    doc_ids.append(hit.docid)
        
        if pairs:
            scores = cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False)
            reranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
            
            # Add remaining BM25 docs
            reranked_ids = set(doc_ids)
            min_score = min(scores) - 1000 if scores.size > 0 else -1000
            for i, hit in enumerate(bm25_hits[200:]):
                if hit.docid not in reranked_ids:
                    reranked.append((hit.docid, min_score - i))
            
            neural_results[qid] = reranked[:1000]
        else:
            neural_results[qid] = [(h.docid, h.score) for h in bm25_hits[:1000]]
    
    neural_map = compute_map(neural_results, qrels)
    print(f"✓ Neural Reranking Validation MAP: {neural_map:.4f}")
    
    # === Step 3: Tune RRF Parameters ===
    print("\n--- Tuning RRF Parameters ---")
    print("Grid: k ∈ {30, 40, 60, 80}, BM25_weight ∈ {1.0, 1.2, 1.5}, Neural_weight ∈ {0.8, 1.0}")
    
    k_values = [30, 40, 60, 80]
    weight_configs = [
        [1.0, 1.0],   # Equal
        [1.2, 1.0],   # Favor BM25 slightly
        [1.5, 1.0],   # Favor BM25 more
        [1.0, 0.8],   # Penalize Neural slightly
        [1.2, 0.8],   # Favor BM25 + penalize Neural
        [1.5, 0.8],   # Strong BM25 preference
    ]
    
    results_table = []
    best_map = 0
    best_k = 60
    best_weights = [1.0, 1.0]
    
    for k in k_values:
        for weights in weight_configs:
            fused = weighted_rrf([bm25_results, neural_results], k=k, weights=weights)
            map_score = compute_map(fused, qrels)
            results_table.append((k, weights, map_score))
            
            if map_score > best_map:
                best_map = map_score
                best_k = k
                best_weights = weights
    
    # Print results table
    print("\n" + "=" * 60)
    print("RESULTS TABLE")
    print("=" * 60)
    print(f"{'k':<6} {'BM25_w':<8} {'Neural_w':<10} {'MAP':<10}")
    print("-" * 40)
    for k, weights, map_score in sorted(results_table, key=lambda x: -x[2]):
        marker = " ⭐" if map_score == best_map else ""
        print(f"{k:<6} {weights[0]:<8} {weights[1]:<10} {map_score:.4f}{marker}")
    
    print("\n" + "=" * 60)
    print(f"BEST PARAMETERS: k={best_k}, weights={best_weights}")
    print(f"BEST VALIDATION MAP: {best_map:.4f}")
    print("=" * 60)
    
    # Compare to individual methods
    print("\n--- Comparison ---")
    print(f"BM25+RM3 alone:     MAP = {bm25_map:.4f}")
    print(f"Neural alone:       MAP = {neural_map:.4f}")
    print(f"RRF Fusion (best):  MAP = {best_map:.4f}")
    improvement = best_map - max(bm25_map, neural_map)
    print(f"Improvement:        +{improvement:.4f} ({improvement/max(bm25_map, neural_map)*100:.1f}%)")

if __name__ == "__main__":
    main()
