#!/usr/bin/env python3
"""
Quick diagnostic to check if validation queries are inherently harder
or if we're overfitting during parameter tuning.
"""

import os
import sys
from collections import defaultdict
import numpy as np

# Try to import pyserini
try:
    from pyserini.search.lucene import LuceneSearcher
except ImportError:
    print("Please install pyserini: pip install pyserini")
    sys.exit(1)


def load_queries(path):
    """Load queries from file"""
    queries = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    qid, query_text = parts[0], parts[1]
                    queries[qid] = query_text
    return queries


def load_qrels(path):
    """Load relevance judgments"""
    qrels = defaultdict(dict)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                qrels[qid][docid] = rel
    return dict(qrels)


def compute_map(results, qrels):
    """Compute Mean Average Precision"""
    aps = []
    for qid in results:
        if qid not in qrels:
            continue
        
        relevant = {docid for docid, rel in qrels[qid].items() if rel > 0}
        if not relevant:
            continue
        
        hits = 0
        precision_sum = 0.0
        for rank, (docid, _) in enumerate(results[qid], 1):
            if docid in relevant:
                hits += 1
                precision_sum += hits / rank
        
        ap = precision_sum / len(relevant) if relevant else 0.0
        aps.append((qid, ap))
    
    return aps


def main():
    print("=" * 60)
    print("VALIDATION DIAGNOSTIC: Are queries 341-350 just harder?")
    print("=" * 60)
    
    # Load data
    queries = load_queries("files/queriesROBUST.txt")
    qrels = load_qrels("files/qrels_50_Queries")
    
    all_sorted = sorted(queries.keys())
    train_qids = all_sorted[:40]      # 301-340
    val_qids = all_sorted[40:50]       # 341-350
    
    print(f"\nTrain queries: {train_qids[0]}-{train_qids[-1]} ({len(train_qids)} queries)")
    print(f"Validation queries: {val_qids[0]}-{val_qids[-1]} ({len(val_qids)} queries)")
    
    # Initialize searcher
    print("\nInitializing searcher...")
    searcher = LuceneSearcher.from_prebuilt_index('robust04')
    
    # Test 1: Plain BM25 (no tuning, no RM3) - baseline
    print("\n" + "-" * 40)
    print("TEST 1: Plain BM25 (k1=0.9, b=0.4) - Baseline")
    print("-" * 40)
    
    searcher.set_bm25(0.9, 0.4)
    
    train_results = {}
    for qid in train_qids:
        hits = searcher.search(queries[qid], k=1000)
        train_results[qid] = [(hit.docid, hit.score) for hit in hits]
    
    val_results = {}
    for qid in val_qids:
        hits = searcher.search(queries[qid], k=1000)
        val_results[qid] = [(hit.docid, hit.score) for hit in hits]
    
    train_aps = compute_map(train_results, qrels)
    val_aps = compute_map(val_results, qrels)
    
    train_map = np.mean([ap for _, ap in train_aps])
    val_map = np.mean([ap for _, ap in val_aps])
    
    print(f"Train MAP: {train_map:.4f}")
    print(f"Validation MAP: {val_map:.4f}")
    print(f"Difference: {((train_map - val_map) / train_map * 100):.1f}%")
    
    # Per-query breakdown
    print("\nPer-query AP breakdown:")
    print("\nTrain queries (301-340):")
    for qid, ap in sorted(train_aps, key=lambda x: x[1]):
        print(f"  {qid}: {ap:.4f}")
    
    print("\nValidation queries (341-350):")
    for qid, ap in sorted(val_aps, key=lambda x: x[1]):
        print(f"  {qid}: {ap:.4f}")
    
    # Test 2: Try different BM25 params
    print("\n" + "-" * 40)
    print("TEST 2: Different BM25 params (k1=1.2, b=0.75)")
    print("-" * 40)
    
    searcher.set_bm25(1.2, 0.75)
    
    val_results2 = {}
    for qid in val_qids:
        hits = searcher.search(queries[qid], k=1000)
        val_results2[qid] = [(hit.docid, hit.score) for hit in hits]
    
    val_aps2 = compute_map(val_results2, qrels)
    val_map2 = np.mean([ap for _, ap in val_aps2])
    print(f"Validation MAP: {val_map2:.4f}")
    
    # Test 3: Conservative RM3
    print("\n" + "-" * 40)
    print("TEST 3: Conservative RM3 (fb_terms=10, fb_docs=5, weight=0.7)")
    print("-" * 40)
    
    searcher.set_bm25(0.9, 0.4)
    searcher.set_rm3(10, 5, 0.7)  # More conservative: fewer terms, higher original weight
    
    val_results3 = {}
    for qid in val_qids:
        hits = searcher.search(queries[qid], k=1000)
        val_results3[qid] = [(hit.docid, hit.score) for hit in hits]
    searcher.unset_rm3()
    
    val_aps3 = compute_map(val_results3, qrels)
    val_map3 = np.mean([ap for _, ap in val_aps3])
    print(f"Validation MAP: {val_map3:.4f}")
    
    # Conclusion
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    
    if val_map < train_map * 0.85:
        print("⚠ Validation queries (341-350) ARE inherently harder than train (301-340)")
        print("  This is normal query difficulty variance, not necessarily overfitting.")
        print("\nRECOMMENDATION:")
        print("  Use ALL 50 queries for training to maximize data usage.")
        print("  Your expected test MAP will likely be somewhere between train and val MAP.")
    else:
        print("✓ Train and validation have similar difficulty")
        print("  Your current approach is valid.")


if __name__ == "__main__":
    main()
