#!/usr/bin/env python3
"""
Fast Fusion Script
==================
Skips redundant Neural Reranking by loading existing run_2.res
Computes only the new/fast components:
1. Run 1b: BM25 + Query2Doc + RM3 (Uses cache)
2. Run 1c: BM25-plain
3. Run 3: 4-Way Fusion
"""

import os
import argparse
from robust04_ranking_solution import ROBUST04Retriever
from collections import defaultdict

def load_trec_run(path):
    """Load TREC run file"""
    results = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, docid, rank, score, _ = parts
                results[qid].append((docid, float(score)))
    return dict(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', required=True)
    parser.add_argument('--qrels', default=None)
    parser.add_argument('--output', default='output')
    args = parser.parse_args()

    # Init retriever
    retriever = ROBUST04Retriever(args.queries, args.qrels, args.output)
    
    print("\n" + "="*60)
    print("FAST FUSION (Skipping Neural Re-run)")
    print("="*60)

    # 1. Load or Run BM25+RM3 (Run 1)
    run1_path = os.path.join(args.output, "run_1.res")
    if os.path.exists(run1_path):
        print(f"✓ Loading existing Run 1 from {run1_path}")
        results_1 = load_trec_run(run1_path)
    else:
        print("⚡ Computing Run 1 (BM25+RM3)...")
        results_1 = retriever.run_bm25_rm3()
        retriever._write_trec_run(results_1, "run_1", run1_path)

    # 2. Run Query2Doc (Run 1b) - This is what we updated!
    print("\n⚡ Computing Run 1b (Query2Doc)...")
    results_1b = retriever.run_bm25_query2doc(
        k1=0.7, b=0.4, fb_terms=70, fb_docs=10, original_weight=0.3
    )
    retriever._write_trec_run(results_1b, "run_1b", os.path.join(args.output, "run_1b.res"))

    # 3. Run BM25-plain (Run 1c)
    print("\n⚡ Computing Run 1c (BM25-plain)...")
    retriever.searcher.set_bm25(0.9, 0.4)
    results_1c = {}
    for qid in retriever.test_qids:
        hits = retriever.searcher.search(retriever.queries[qid], k=1000)
        results_1c[qid] = [(hit.docid, hit.score) for hit in hits]
    retriever._write_trec_run(results_1c, "run_1c", os.path.join(args.output, "run_1c.res"))

    # 4. Load Neural (Run 2)
    run2_path = os.path.join(args.output, "run_2.res")
    if os.path.exists(run2_path):
        print(f"\n✓ Loading existing Run 2 from {run2_path}")
        results_2 = load_trec_run(run2_path)
    else:
        print("❌ Error: run_2.res not found! Cannot skip neural reranking.")
        return

    # 5. Perform 4-Way Fusion
    print("\n⚡ Performing 4-Way RRF Fusion...")
    
    def get_query_weights(query: str) -> list:
        word_count = len(query.split())
        if word_count <= 3:
            return [1.5, 1.3, 1.2, 0.7]
        elif word_count <= 5:
            return [1.3, 1.2, 1.0, 1.0]
        else:
            return [1.0, 1.0, 0.8, 1.5]

    results_3 = {}
    for qid in retriever.test_qids:
        query = retriever.queries[qid]
        weights = get_query_weights(query)
        
        rankings = [
            {qid: results_1.get(qid, [])},
            {qid: results_1b.get(qid, [])},
            {qid: results_1c.get(qid, [])},
            {qid: results_2.get(qid, [])}
        ]
        
        fused = retriever.weighted_reciprocal_rank_fusion(rankings, k=30, weights=weights)
        results_3[qid] = fused.get(qid, [])

    output_3 = os.path.join(args.output, "run_3.res")
    retriever._write_trec_run(results_3, "run_3", output_3)
    
    print("\n✓ FAST FUSION COMPLETE!")
    print(f"  Saved to {output_3}")

if __name__ == "__main__":
    main()
