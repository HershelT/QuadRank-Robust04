#!/usr/bin/env python3
"""
Test Query2Doc LLM Expansion
============================
Quick test to verify Gemini API integration and caching works.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from robust04_ranking_solution import ROBUST04Retriever

def main():
    print("=" * 60)
    print("Testing Query2Doc LLM Expansion")
    print("=" * 60)
    
    # Initialize retriever (minimal)
    retriever = ROBUST04Retriever(
        queries_path="files/queriesROBUST.txt",
        qrels_path="files/qrels_50_Queries",
        output_dir="output"
    )
    
    # Test on first 3 queries
    test_qids = retriever.val_qids[:3]
    print(f"\nTesting on {len(test_qids)} queries: {test_qids}")
    
    # Expand queries
    expanded = retriever.expand_queries_with_llm(test_qids)
    
    # Show results
    print("\n" + "=" * 60)
    print("EXPANSION RESULTS")
    print("=" * 60)
    
    for qid in test_qids:
        query = retriever.queries[qid]
        exp = expanded[qid]
        print(f"\n[{qid}] Original: {query}")
        print(f"    Expanded: {exp[:200]}...")

    print("\n" + "=" * 60)
    print("✓ Query2Doc test completed!")
    print("  Check output/query_expansions.json for cached results")
    print("=" * 60)

if __name__ == "__main__":
    main()
