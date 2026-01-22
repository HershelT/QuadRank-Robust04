#!/usr/bin/env python3
"""
Precompute Query Expansions using LiteLLM (Universal)
=====================================================

This script generates pseudo-documents for all queries using ANY LLM supported by LiteLLM
(Gemini, Ollama, OpenRouter, OpenAI, etc.) and saves them to a cache file.

Usage:
    # Use default model (from .env or code default)
    python precompute_expansions.py --queries files/queriesROBUST.txt

    # Use Ollama
    python precompute_expansions.py --queries files/queriesROBUST.txt --model ollama/llama3

    # Use OpenRouter
    python precompute_expansions.py --queries files/queriesROBUST.txt --model openrouter/meta-llama/llama-3.1-405b

The cache will be saved to: output/query_expansions.json
"""

import os
import sys
import json
import hashlib
import argparse
import time
from typing import Dict, Optional

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables might not load.")

# Import LiteLLM
try:
    import litellm
    from litellm import completion
    # Suppress verbose LiteLLM logging
    litellm.verbose_logger = False
    os.environ["LITELLM_LOG"] = "ERROR"
except ImportError:
    print("❌ ERROR: litellm not installed")
    print("   Run: pip install litellm")
    sys.exit(1)

# ============================================================
# CONFIGURATION
# ============================================================

# Default model if not specified in CLI or .env
DEFAULT_MODEL = "gemini/gemini-flash-latest"

# Query2Doc prompt template
QUERY2DOC_PROMPT = """Write a short passage (100-150 words) that would be relevant to answer this question. The passage should contain facts and information that directly address the query.

Query: {query}

Relevant passage:"""

CACHE_FILENAME = "query_expansions.json"

# ============================================================
# UTILS
# ============================================================

def load_cache(cache_path: str) -> Dict[str, str]:
    """Load existing cache from disk"""
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"⚠ Warning: Corrupt cache file {cache_path}, starting fresh.")
                return {}
    return {}

def save_cache(cache: Dict[str, str], cache_path: str):
    """Save cache to disk"""
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def load_queries(queries_path: str) -> Dict[str, str]:
    """Load queries from file"""
    queries = {}
    if not os.path.exists(queries_path):
        print(f"❌ ERROR: Queries file not found: {queries_path}")
        sys.exit(1)
        
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
# LLM GENERATION
# ============================================================

def generate_expansion(model_name: str, query: str, max_retries: int = 3) -> str:
    """Generate a single query expansion using LiteLLM"""
    messages = [
        {"role": "user", "content": QUERY2DOC_PROMPT.format(query=query)}
    ]

    for attempt in range(max_retries):
        try:
            response = completion(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=250
            )
            
            # Extract text content
            if hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    return content.strip()
            
            return ""

        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                print(f"  ⚠ Retry {attempt + 1}/{max_retries}: {error_msg[:100]}...")
                # Exponential backoff
                time.sleep(2 ** attempt)
            else:
                print(f"  ❌ Failed after {max_retries} attempts: {error_msg}")
                return ""
    return ""

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Precompute Query2Doc Expansions (LiteLLM)')
    parser.add_argument('--queries', type=str, required=True,
                        help='Path to queriesROBUST.txt')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory for cache file')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (e.g., ollama/llama3, gemini/gemini-1.5-flash). '
                             'Overrides LLM_MODEL env var.')
    parser.add_argument('--force', action='store_true',
                        help='Force regenerate all expansions (ignore cache)')
    
    args = parser.parse_args()
    
    # Determine model: CLI > ENV > Default
    env_model = os.environ.get('LLM_MODEL')
    model_name = args.model or env_model or DEFAULT_MODEL
    
    # Setup
    os.makedirs(args.output, exist_ok=True)
    cache_path = os.path.join(args.output, CACHE_FILENAME)
    
    print("="*60)
    print("QUERY2DOC EXPANSION PRECOMPUTATION (Universal)")
    print("="*60)
    print(f"Model: {model_name}")
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
    
    # Find queries that need expansion (Hash-based deduplication)
    to_expand = []
    seen_hashes = set()
    
    for qid, query in queries.items():
        # Use MD5 hash of query text as key
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        
        # Check if already processed
        if query_hash not in cache and query_hash not in seen_hashes:
            to_expand.append((query_hash, query))
            seen_hashes.add(query_hash)
    
    if not to_expand:
        print("\n✓ All queries already have cached expansions!")
        return
    
    print(f"\n🔄 Need to generate {len(to_expand)} expansions")
    print("\n--- Generating Expansions ---")
    
    # Loop
    count = 0
    for i, (query_hash, query) in enumerate(to_expand):
        print(f"[{i+1}/{len(to_expand)}] {model_name} | {query[:40]}...")
        
        expansion = generate_expansion(model_name, query)
        
        if expansion:
            cache[query_hash] = expansion
            count += 1
            print(f"  ✓ Generated ({len(expansion)} chars)")
            
            # Save frequently (every 5) to avoid data loss
            if count % 5 == 0:
                save_cache(cache, cache_path)
        else:
            print(f"  ⚠ Empty expansion generated")
        
        # Small delay to be polite to APIs
        if i < len(to_expand) - 1:
            time.sleep(0.5)
            
    # Final save
    save_cache(cache, cache_path)
    
    print("\n" + "="*60)
    print("COMPLETION SUMMARY")
    print("="*60)
    print(f"✓ Cache saved to: {cache_path}")
    print(f"✓ Total cached expansions: {len(cache)}")
    print("\nYou can now run the main pipeline:")
    print(f"  python robust04_ranking_solution.py --queries {args.queries} --output {args.output}")

if __name__ == "__main__":
    main()
