#!/usr/bin/env python3
"""
Quick win test: Add BM25-plain to RRF fusion and validate improvement.
Fuses: BM25+RM3 (run_1) + Neural (run_2) + BM25-plain (new)
"""

import os
from collections import defaultdict
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher

# === Configuration ===
RUN_1_PATH = "output/run_1.res"
RUN_2_PATH = "output/run_2.res"
OUTPUT_PATH = "output/run_3.res"
QUERIES_PATH = "files/queriesROBUST.txt"
QRELS_PATH = "files/qrels_50_Queries"
VAL_QIDS = [str(i) for i in range(301, 351)]

def load_queries(path):
    queries = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                queries[parts[0]] = parts[1]
    return queries

def load_run(path):
    results = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, docid, rank, score, _ = parts[:6]
                results[qid].append((docid, float(score)))
    for qid in results:
        results[qid] = sorted(results[qid], key=lambda x: -x[1])
    return dict(results)

def load_qrels(path):
    qrels = defaultdict(dict)
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                if rel > 0:
                    qrels[qid][docid] = rel
    return dict(qrels)

def compute_ap(ranked_list, relevant_docs):
    if not relevant_docs:
        return 0.0
    hits, sum_prec = 0, 0.0
    for rank, (docid, _) in enumerate(ranked_list, 1):
        if docid in relevant_docs:
            hits += 1
            sum_prec += hits / rank
    return sum_prec / len(relevant_docs)

def compute_map(results, qrels):
    aps = [compute_ap(results[qid], qrels[qid]) for qid in results if qid in qrels]
    return sum(aps) / len(aps) if aps else 0.0

def weighted_rrf(ranked_lists, k=30, weights=None):
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    fused = {}
    all_qids = set()
    for r in ranked_lists:
        all_qids.update(r.keys())
    for qid in all_qids:
        doc_scores = defaultdict(float)
        for w, r in zip(weights, ranked_lists):
            if qid in r:
                for rank, (docid, _) in enumerate(r[qid], 1):
                    doc_scores[docid] += w / (k + rank)
        fused[qid] = sorted(doc_scores.items(), key=lambda x: -x[1])[:1000]
    return fused

def write_run(results, path, name="run_3"):
    with open(path, 'w') as f:
        for qid in sorted(results.keys(), key=lambda x: int(x) if x.isdigit() else x):
            for rank, (docid, score) in enumerate(results[qid][:1000], 1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {name}\n")
    print(f"Saved to {path}")

def main():
    print("=" * 60)
    print("Quick Win Test: Add BM25-plain to Fusion")
    print("=" * 60)
    
    # Load queries
    queries = load_queries(QUERIES_PATH)
    qrels = load_qrels(QRELS_PATH)
    print(f"Loaded {len(queries)} queries, qrels for {len(qrels)} queries")
    
    # Load existing runs
    run_1 = load_run(RUN_1_PATH)  # BM25+RM3
    run_2 = load_run(RUN_2_PATH)  # Neural
    print(f"Loaded run_1 ({len(run_1)} q), run_2 ({len(run_2)} q)")
    
    # Generate BM25-plain results
    print("\nGenerating BM25-plain results...")
    searcher = LuceneSearcher.from_prebuilt_index('robust04')
    searcher.set_bm25(0.9, 0.4)  # Default params, no RM3
    
    # Get all qids from run_1
    all_qids = list(run_1.keys())
    bm25_plain = {}
    for qid in tqdm(all_qids, desc="BM25-plain"):
        if qid in queries:
            hits = searcher.search(queries[qid], k=1000)
            bm25_plain[qid] = [(h.docid, h.score) for h in hits]
    
    print(f"Generated BM25-plain for {len(bm25_plain)} queries")
    
    # === Validation tuning ===
    print("\n--- Validation: Finding best weights for 3-way fusion ---")
    
    val_run_1 = {qid: run_1[qid] for qid in VAL_QIDS if qid in run_1}
    val_run_2 = {qid: run_2[qid] for qid in VAL_QIDS if qid in run_2}
    val_bm25_plain = {qid: bm25_plain[qid] for qid in VAL_QIDS if qid in bm25_plain}
    
    print(f"Validation queries: run_1={len(val_run_1)}, run_2={len(val_run_2)}, bm25_plain={len(val_bm25_plain)}")
    
    # Test configurations
    configs = [
        # (k, [bm25+rm3, neural, bm25_plain])
        (30, [1.5, 0.8, 1.0]),   # Current best + plain
        (30, [1.5, 0.8, 1.2]),   # More weight on plain
        (30, [1.5, 0.8, 0.8]),   # Less weight on plain
        (30, [1.5, 1.0, 1.0]),   # Equal neural/plain
        (40, [1.5, 0.8, 1.0]),   # Higher k
        (30, [1.2, 0.8, 1.0]),   # Less BM25+RM3
    ]
    
    # Also test 2-way baseline
    print("\n2-way baseline (current):")
    fused_2way = weighted_rrf([val_run_1, val_run_2], k=30, weights=[1.5, 0.8])
    map_2way = compute_map(fused_2way, qrels)
    print(f"  RRF(BM25+RM3, Neural) → MAP: {map_2way:.4f}")
    
    print("\n3-way fusion tests:")
    best_map = 0
    best_config = None
    
    for k, weights in configs:
        fused = weighted_rrf([val_run_1, val_run_2, val_bm25_plain], k=k, weights=weights)
        map_score = compute_map(fused, qrels)
        marker = " ⭐" if map_score > best_map else ""
        print(f"  k={k}, w={weights} → MAP: {map_score:.4f}{marker}")
        if map_score > best_map:
            best_map = map_score
            best_config = (k, weights)
    
    print(f"\n✓ Best 3-way config: k={best_config[0]}, w={best_config[1]} → MAP: {best_map:.4f}")
    improvement = best_map - map_2way
    print(f"   Improvement over 2-way: {improvement:+.4f} ({improvement/map_2way*100:+.1f}%)")
    
    # === Apply to test queries ===
    if improvement > 0:
        print("\n--- Applying best config to all queries ---")
        test_qids = [qid for qid in all_qids if qid not in VAL_QIDS]
        test_run_1 = {qid: run_1[qid] for qid in test_qids}
        test_run_2 = {qid: run_2[qid] for qid in test_qids}
        test_bm25_plain = {qid: bm25_plain[qid] for qid in test_qids}
        
        results_3 = weighted_rrf(
            [test_run_1, test_run_2, test_bm25_plain],
            k=best_config[0],
            weights=best_config[1]
        )
        
        write_run(results_3, OUTPUT_PATH)
        print(f"\n✓ Created {OUTPUT_PATH} with 3-way fusion")
        print("\nRun evaluation:")
        print("  python evaluate_runs.py --qrels files/qrels_robust04_full.txt output/run_3.res")
    else:
        print("\n⚠ 3-way fusion did not improve. Keeping 2-way fusion.")

if __name__ == "__main__":
    main()
