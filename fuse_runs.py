#!/usr/bin/env python3
"""
Quick script to fuse existing run_1.res and run_2.res into run_3.res
Uses weighted RRF with parameter tuning on validation set.
"""

import os
from collections import defaultdict
from tqdm import tqdm

# === Configuration ===
RUN_1_PATH = "output/run_1.res"
RUN_2_PATH = "output/run_2.res"
OUTPUT_PATH = "output/run_3.res"
QRELS_PATH = "files/qrels_50_Queries"
VAL_QIDS = [str(i) for i in range(301, 351)]  # 50 validation queries

def load_run(path):
    """Load a TREC run file into a dict: qid -> [(docid, score), ...]"""
    results = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, docid, rank, score, _ = parts[:6]
                results[qid].append((docid, float(score)))
    # Sort by score (should already be sorted, but just in case)
    for qid in results:
        results[qid] = sorted(results[qid], key=lambda x: -x[1])
    return dict(results)

def load_qrels(path):
    """Load qrels."""
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
    """Compute Average Precision."""
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
    """Compute MAP."""
    aps = []
    for qid, ranked in results.items():
        if qid in qrels:
            aps.append(compute_ap(ranked, qrels[qid]))
    return sum(aps) / len(aps) if aps else 0.0

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

def write_run(results, path, run_name="run_3"):
    """Write TREC run file."""
    with open(path, 'w') as f:
        for qid in sorted(results.keys(), key=lambda x: int(x) if x.isdigit() else x):
            for rank, (docid, score) in enumerate(results[qid][:1000], 1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {run_name}\n")
    print(f"Saved to {path}")

def main():
    print("=" * 60)
    print("RRF Fusion: run_1.res + run_2.res → run_3.res")
    print("=" * 60)
    
    # Optimal parameters from validation tuning
    BEST_K = 30
    BEST_WEIGHTS = [1.5, 0.8]  # [BM25 weight, Neural weight]
    
    print(f"\nUsing tuned params: k={BEST_K}, weights={BEST_WEIGHTS}")
    print("(Found via run_validation_on_50.py)")
    
    # Load existing runs
    print(f"\nLoading {RUN_1_PATH}...")
    run_1 = load_run(RUN_1_PATH)
    print(f"  Loaded {len(run_1)} queries")
    
    print(f"Loading {RUN_2_PATH}...")
    run_2 = load_run(RUN_2_PATH)
    print(f"  Loaded {len(run_2)} queries")
    
    # Fuse all queries
    print("\n--- Fusing with weighted RRF ---")
    results_3 = weighted_rrf([run_1, run_2], k=BEST_K, weights=BEST_WEIGHTS)
    print(f"Fused {len(results_3)} queries")
    
    # Write output
    write_run(results_3, OUTPUT_PATH)
    
    print("\n" + "=" * 60)
    print(f"Done! Created {OUTPUT_PATH}")
    print(f"Params: k={BEST_K}, weights={BEST_WEIGHTS}")
    print("=" * 60)
    print("\nRun evaluation:")
    print("  python evaluate_runs.py --qrels files/qrels_robust04_full.txt output/run_3.res")

if __name__ == "__main__":
    main()
