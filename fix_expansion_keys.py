#!/usr/bin/env python3
"""
Fix query_expansions.json keys to use correct MD5 hashes.
Maps expansions by line order from queriesROBUST.txt.
"""

import json
import hashlib

# Load queries
queries = []
with open('files/queriesROBUST.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split('\t')
            if len(parts) >= 2:
                qid, query_text = parts[0], parts[1]
                queries.append((qid, query_text))

print(f"Loaded {len(queries)} queries from queriesROBUST.txt")

# Load existing expansions
with open('output/query_expansions.json', 'r', encoding='utf-8') as f:
    old_cache = json.load(f)

print(f"Loaded {len(old_cache)} expansions from query_expansions.json")

# Get old values in order
old_values = list(old_cache.values())

# Check if counts match
if len(queries) != len(old_values):
    print(f"⚠ WARNING: Query count ({len(queries)}) != Expansion count ({len(old_values)})")
    print(f"  Will match by order up to min count: {min(len(queries), len(old_values))}")

# Create new cache with simple QID keys
new_cache = {}
for i, (qid, query_text) in enumerate(queries):
    if i < len(old_values):
        # Use simple QID as key
        new_cache[qid] = old_values[i]
        if i < 5:
            print(f"  [{qid}] '{query_text[:40]}...' → {qid}")   

# Save fixed cache
with open('output/query_expansions.json', 'w', encoding='utf-8') as f:
    json.dump(new_cache, f, indent=2, ensure_ascii=False)

print(f"\n✓ Fixed {len(new_cache)} expansions with QID keys")
print("  Saved to output/query_expansions.json")
