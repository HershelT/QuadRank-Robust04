#!/usr/bin/env python3
"""
Migrate Query Cache: QID Keys -> MD5 Hash Keys
==============================================
Needed for V2 architecture which deduplicates query text.
"""
import json
import hashlib
import os

def load_queries(path):
    queries = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) >= 2:
                qid, text = parts[0], parts[1]
                queries[qid] = text
    return queries

def main():
    print("Migrating cache keys from QID to MD5 Hash...")
    
    # 1. Load Queries (QID -> Text)
    queries = load_queries('files/queriesROBUST.txt')
    print(f"Loaded {len(queries)} queries")
    
    # 2. Load Existing Cache (QID -> Expansion)
    with open('output/query_expansions.json', 'r', encoding='utf-8') as f:
        old_cache = json.load(f)
    print(f"Loaded {len(old_cache)} existing cached items")
    
    # 3. Create New Cache (Hash -> Expansion)
    new_cache = {}
    stats = {'converted': 0, 'duplicates': 0, 'missing_query': 0}
    
    for key, expansion in old_cache.items():
        # Key assumed to be QID
        if key in queries:
            query_text = queries[key]
            query_hash = hashlib.md5(query_text.encode('utf-8')).hexdigest()
            
            if query_hash in new_cache:
                stats['duplicates'] += 1
            else:
                new_cache[query_hash] = expansion
                stats['converted'] += 1
        else:
            # Maybe it's already a hash? (Length 32 hex)
            if len(key) == 32 and all(c in '0123456789abcdef' for c in key):
                new_cache[key] = expansion
                stats['converted'] += 1
            else:
                print(f"Warning: Cache key {key} not found in queries file")
                stats['missing_query'] += 1
    
    # 4. Save
    with open('output/query_expansions.json', 'w', encoding='utf-8') as f:
        json.dump(new_cache, f, indent=2, ensure_ascii=False)
        
    print("\nMigration Complete:")
    print(f"  - Unique Hash Entries: {len(new_cache)}")
    print(f"  - Duplicates merged: {stats['duplicates']}")
    print(f"  - Missing/Skipped: {stats['missing_query']}")
    print("Saved to output/query_expansions.json")

if __name__ == "__main__":
    main()
