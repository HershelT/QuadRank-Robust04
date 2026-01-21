#!/usr/bin/env python3
"""
ROBUST04 Run Evaluation Script (Pure Python - No Dependencies)
Evaluates TREC-formatted run files against your qrels file.

Usage:
    python evaluate_runs.py --qrels qrels_50_Queries run_1.res run_2.res run_3.res
"""

import os
import sys
import math
from collections import defaultdict


def load_qrels(qrels_file):
    """Load qrels: returns {qid: {docid: relevance}}"""
    qrels = defaultdict(dict)
    try:
        with open(qrels_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid = parts[0]
                    docid = parts[2]
                    rel = int(parts[3])
                    qrels[qid][docid] = rel
        print(f"Loaded qrels for {len(qrels)} queries")
    except Exception as e:
        print(f"Error loading qrels: {e}")
        return {}
    return dict(qrels)


def load_run(run_file):
    """Load run file: returns {qid: [(docid, score, rank), ...]}"""
    run = defaultdict(list)
    try:
        with open(run_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) >= 5:
                    qid = parts[0]
                    docid = parts[2]
                    try:
                        rank = int(parts[3])
                        score = float(parts[4])
                    except ValueError:
                        continue
                    run[qid].append((docid, score, rank))
        
        # Sort by score descending for each query
        for qid in run:
            run[qid] = sorted(run[qid], key=lambda x: (-x[1], x[2]))
        
        print(f"Loaded run with {len(run)} queries, {sum(len(v) for v in run.values())} total results")
    except Exception as e:
        print(f"Error loading run file: {e}")
        return {}
    return dict(run)


def compute_average_precision(ranked_list, relevant_docs):
    """Compute AP for a single query"""
    relevant_set = {docid for docid, rel in relevant_docs.items() if rel > 0}
    
    if not relevant_set:
        return 0.0
    
    hits = 0
    sum_precisions = 0.0
    
    for rank, (docid, score, _) in enumerate(ranked_list, 1):
        if docid in relevant_set:
            hits += 1
            precision_at_rank = hits / rank
            sum_precisions += precision_at_rank
    
    return sum_precisions / len(relevant_set)


def compute_precision_at_k(ranked_list, relevant_docs, k):
    """Compute P@K"""
    relevant_set = {docid for docid, rel in relevant_docs.items() if rel > 0}
    
    hits = 0
    for docid, score, _ in ranked_list[:k]:
        if docid in relevant_set:
            hits += 1
    
    return hits / k if k > 0 else 0.0


def compute_recall_at_k(ranked_list, relevant_docs, k):
    """Compute Recall@K"""
    relevant_set = {docid for docid, rel in relevant_docs.items() if rel > 0}
    
    if not relevant_set:
        return 0.0
    
    hits = 0
    for docid, score, _ in ranked_list[:k]:
        if docid in relevant_set:
            hits += 1
    
    return hits / len(relevant_set)


def compute_ndcg_at_k(ranked_list, relevant_docs, k):
    """Compute NDCG@K"""
    
    def dcg(relevances):
        dcg_val = 0.0
        for i, rel in enumerate(relevances, 1):
            dcg_val += (2**rel - 1) / math.log2(i + 1)
        return dcg_val
    
    # Get relevance scores in ranked order
    relevances = []
    for docid, score, _ in ranked_list[:k]:
        rel = relevant_docs.get(docid, 0)
        relevances.append(max(0, rel))
    
    # Pad with zeros
    while len(relevances) < k:
        relevances.append(0)
    
    # Ideal ranking
    ideal_rels = sorted([max(0, r) for r in relevant_docs.values()], reverse=True)[:k]
    while len(ideal_rels) < k:
        ideal_rels.append(0)
    
    dcg_score = dcg(relevances)
    idcg_score = dcg(ideal_rels)
    
    if idcg_score == 0:
        return 0.0
    
    return dcg_score / idcg_score


def compute_reciprocal_rank(ranked_list, relevant_docs):
    """Compute Reciprocal Rank (for MRR)"""
    relevant_set = {docid for docid, rel in relevant_docs.items() if rel > 0}
    
    for rank, (docid, score, _) in enumerate(ranked_list, 1):
        if docid in relevant_set:
            return 1.0 / rank
    
    return 0.0


def evaluate_run(run, qrels):
    """Evaluate a run against qrels, return metrics dict"""
    
    # Only evaluate queries that exist in both run and qrels
    common_qids = set(run.keys()) & set(qrels.keys())
    
    if not common_qids:
        print(f"  WARNING: No common queries between run and qrels!")
        print(f"  Run queries (sample): {list(run.keys())[:5]}")
        print(f"  Qrels queries (sample): {list(qrels.keys())[:5]}")
        return None
    
    # Collect per-query metrics
    ap_scores = []
    p5_scores = []
    p10_scores = []
    p20_scores = []
    recall100_scores = []
    recall1000_scores = []
    ndcg10_scores = []
    ndcg20_scores = []
    rr_scores = []
    
    num_rel = 0
    num_rel_ret = 0
    
    for qid in common_qids:
        ranked_list = run[qid]
        relevant_docs = qrels[qid]
        
        # Count relevant docs
        rel_for_query = sum(1 for r in relevant_docs.values() if r > 0)
        num_rel += rel_for_query
        
        # Count relevant retrieved
        rel_set = {d for d, r in relevant_docs.items() if r > 0}
        for docid, _, _ in ranked_list:
            if docid in rel_set:
                num_rel_ret += 1
        
        # Compute metrics
        ap_scores.append(compute_average_precision(ranked_list, relevant_docs))
        p5_scores.append(compute_precision_at_k(ranked_list, relevant_docs, 5))
        p10_scores.append(compute_precision_at_k(ranked_list, relevant_docs, 10))
        p20_scores.append(compute_precision_at_k(ranked_list, relevant_docs, 20))
        recall100_scores.append(compute_recall_at_k(ranked_list, relevant_docs, 100))
        recall1000_scores.append(compute_recall_at_k(ranked_list, relevant_docs, 1000))
        ndcg10_scores.append(compute_ndcg_at_k(ranked_list, relevant_docs, 10))
        ndcg20_scores.append(compute_ndcg_at_k(ranked_list, relevant_docs, 20))
        rr_scores.append(compute_reciprocal_rank(ranked_list, relevant_docs))
    
    n = len(common_qids)
    
    return {
        'num_q': n,
        'num_rel': num_rel,
        'num_rel_ret': num_rel_ret,
        'MAP': sum(ap_scores) / n,
        'P@5': sum(p5_scores) / n,
        'P@10': sum(p10_scores) / n,
        'P@20': sum(p20_scores) / n,
        'Recall@100': sum(recall100_scores) / n,
        'Recall@1000': sum(recall1000_scores) / n,
        'NDCG@10': sum(ndcg10_scores) / n,
        'NDCG@20': sum(ndcg20_scores) / n,
        'MRR': sum(rr_scores) / n,
    }


def print_metrics(run_name, metrics):
    """Pretty print metrics"""
    if metrics is None:
        print("  No metrics computed (no overlapping queries)")
        return
    
    print(f"  Queries evaluated: {metrics['num_q']}")
    print(f"  Relevant docs:     {metrics['num_rel']}")
    print(f"  Relevant retrieved:{metrics['num_rel_ret']}")
    print()
    print(f"  MAP:          {metrics['MAP']:.4f}")
    print(f"  MRR:          {metrics['MRR']:.4f}")
    print(f"  P@5:          {metrics['P@5']:.4f}")
    print(f"  P@10:         {metrics['P@10']:.4f}")
    print(f"  P@20:         {metrics['P@20']:.4f}")
    print(f"  NDCG@10:      {metrics['NDCG@10']:.4f}")
    print(f"  NDCG@20:      {metrics['NDCG@20']:.4f}")
    print(f"  Recall@100:   {metrics['Recall@100']:.4f}")
    print(f"  Recall@1000:  {metrics['Recall@1000']:.4f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate TREC run files')
    parser.add_argument('run_files', nargs='*', help='Run files to evaluate')
    parser.add_argument('--qrels', type=str, required=True,
                        help='Path to qrels file (e.g., qrels_50_Queries)')
    
    args = parser.parse_args()
    
    # Find run files
    run_files = args.run_files
    if not run_files:
        # Look for .res files in current directory
        run_files = sorted([f for f in os.listdir('.') if f.endswith('.res')])
        if not run_files:
            print("No run files specified and no .res files found in current directory")
            print("Usage: python evaluate_runs.py --qrels qrels_50_Queries run_1.res run_2.res run_3.res")
            sys.exit(1)
    
    # Load qrels
    qrels_file = args.qrels
    if not os.path.exists(qrels_file):
        print(f"Error: qrels file not found: {qrels_file}")
        sys.exit(1)
    
    print(f"\nLoading qrels from: {qrels_file}")
    qrels = load_qrels(qrels_file)
    
    if not qrels:
        print("Failed to load qrels!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("ROBUST04 EVALUATION RESULTS")
    print("=" * 60)
    
    all_results = []
    
    for run_file in run_files:
        if not os.path.exists(run_file):
            print(f"\nFile not found: {run_file}")
            continue
        
        print(f"\n{run_file}")
        print("-" * 50)
        
        run = load_run(run_file)
        if not run:
            print("  Failed to load run file")
            continue
        
        metrics = evaluate_run(run, qrels)
        print_metrics(run_file, metrics)
        
        if metrics:
            all_results.append((run_file, metrics))
    
    # Summary table
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("SUMMARY (sorted by MAP)")
        print("=" * 60)
        
        sorted_results = sorted(all_results, key=lambda x: x[1]['MAP'], reverse=True)
        
        print(f"\n{'Run':<25} {'MAP':>8} {'P@10':>8} {'P@20':>8} {'NDCG@20':>8}")
        print("-" * 65)
        
        for run_file, metrics in sorted_results:
            name = os.path.basename(run_file)[:24]
            print(f"{name:<25} {metrics['MAP']:>8.4f} {metrics['P@10']:>8.4f} {metrics['P@20']:>8.4f} {metrics['NDCG@20']:>8.4f}")
        
        best_run, best_metrics = sorted_results[0]
        print(f"\n*** BEST: {os.path.basename(best_run)} with MAP = {best_metrics['MAP']:.4f} ***")


if __name__ == "__main__":
    main()