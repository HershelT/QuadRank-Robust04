#!/usr/bin/env python3
"""
Precompute Query Expansions using LLM (Query2Doc)
==================================================

This script generates pseudo-documents for all queries using Gemini API
and saves them to a cache file. Run this BEFORE the main pipeline.

Usage:
    python precompute_expansions.py --queries files/queriesROBUST.txt --output output

The cache will be saved to: output/query_expansions.json

You can then run the main pipeline which will use the cached expansions:
    python robust04_ranking_solution.py --queries files/queriesROBUST.txt --output output
"""

import os
import sys
import json
import hashlib
import argparse
import time
from typing import Dict

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Make sure GEMINI_API_KEY is set.")

# ============================================================
# CONFIGURATION
# ============================================================

# Query2Doc prompt template (based on EMNLP 2023 paper)
QUERY2DOC_PROMPT = """Write a short passage (100-150 words) that would be relevant to answer this question. The passage should contain facts and information that directly address the query.

Query: {query}

Relevant passage:"""

# Cache file name (relative to output directory)
CACHE_FILENAME = "query_expansions.json"

# ============================================================
# LLM FUNCTIONS
# ============================================================

def init_gemini_client():
    """Initialize Gemini API client"""
    try:
        import google.generativeai as genai
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            print("❌ ERROR: GEMINI_API_KEY not found in environment")
            print("   Set it in .env file or export GEMINI_API_KEY=your_key")
            sys.exit(1)
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        print("✓ Gemini API initialized (gemini-2.0-flash)")
        return model
    except ImportError:
        print("❌ ERROR: google-generativeai not installed")
        print("   Run: pip install google-generativeai")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize Gemini API: {e}")
        sys.exit(1)


def load_cache(cache_path: str) -> Dict[str, str]:
    """Load existing cache from disk"""
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict[str, str], cache_path: str):
    """Save cache to disk"""
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def generate_expansion(model, query: str, max_retries: int = 3) -> str:
    """Generate a single query expansion with retries"""
    for attempt in range(max_retries):
        try:
            prompt = QUERY2DOC_PROMPT.format(query=query)
            response = model.generate_content(prompt)
            expansion = response.text.strip()
            
            # Limit expansion length
            if len(expansion) > 500:
                expansion = expansion[:500]
            
            return expansion
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  ⚠ Retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"  ❌ Failed after {max_retries} attempts: {e}")
                return ""
    return ""


def load_queries(queries_path: str) -> Dict[str, str]:
    """Load queries from file"""
    queries = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                qid, query = parts[0], parts[1]
                queries[qid] = query
    return queries


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Precompute Query2Doc Expansions')
    parser.add_argument('--queries', type=str, required=True,
                        help='Path to queriesROBUST.txt')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory for cache file')
    parser.add_argument('--force', action='store_true',
                        help='Force regenerate all expansions (ignore cache)')
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output, exist_ok=True)
    cache_path = os.path.join(args.output, CACHE_FILENAME)
    
    print("="*60)
    print("QUERY2DOC EXPANSION PRECOMPUTATION")
    print("="*60)
    print(f"Queries file: {args.queries}")
    print(f"Cache file: {cache_path}")
    
    # Load queries
    queries = load_queries(args.queries)
    print(f"Loaded {len(queries)} queries")
    
    # Load existing cache
    if args.force:
        cache = {}
        print("⚠ Force mode: ignoring existing cache")
    else:
        cache = load_cache(cache_path)
        print(f"📁 Existing cache has {len(cache)} entries")
    
    # Find queries that need expansion
    to_expand = []
    for qid, query in queries.items():
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key not in cache:
            to_expand.append((qid, query, cache_key))
    
    if not to_expand:
        print("\n✓ All queries already have cached expansions!")
        print(f"  Cache file: {cache_path}")
        return
    
    print(f"\n🔄 Need to generate {len(to_expand)} expansions")
    
    # Initialize Gemini
    model = init_gemini_client()
    
    # Generate expansions
    print("\n--- Generating Expansions ---")
    for i, (qid, query, cache_key) in enumerate(to_expand):
        print(f"[{i+1}/{len(to_expand)}] Query {qid}: {query[:50]}...")
        
        expansion = generate_expansion(model, query)
        if expansion:
            cache[cache_key] = expansion
            # Save after each to avoid losing progress
            save_cache(cache, cache_path)
            print(f"  ✓ Generated ({len(expansion)} chars)")
        else:
            print(f"  ⚠ Empty expansion")
        
        # Small delay to avoid rate limiting
        if i < len(to_expand) - 1:
            time.sleep(0.5)
    
    # Summary
    print("\n" + "="*60)
    print("COMPLETION SUMMARY")
    print("="*60)
    print(f"✓ Cache file: {cache_path}")
    print(f"✓ Total cached expansions: {len(cache)}")
    print("\nYou can now run the main pipeline:")
    print(f"  python robust04_ranking_solution.py --queries {args.queries} --output {args.output}")


if __name__ == "__main__":
    main()
