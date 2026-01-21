#!/usr/bin/env python3
"""
ROBUST04 Text Retrieval Ranking Competition Solution - V2 UPGRADED
===================================================================

IMPROVEMENTS OVER V1:
1. monoT5 reranker (SOTA for document ranking)
2. Deeper reranking (250 documents)
3. Query-dependent fusion weights
4. Multi-model ensemble option
5. Improved RRF with optimized parameters

Three Methods:
1. BM25 + RM3 (Query Expansion) - Strong baseline from class material
2. Neural Reranking with Cross-Encoder - Beyond class material (advanced)  
3. RRF Fusion - Hybrid approach combining multiple rankers

Hardware: RTX 5070 (8GB VRAM), 64GB RAM, Intel Ultra 9
Target: Maximum MAP on ROBUST04 (199 test queries)

Author: Generated for Text Retrieval Course Competition
Date: January 2026
"""

import os
import sys
import time
import re
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import json

# Try to import, install if needed
try:
    from pyserini.search.lucene import LuceneSearcher
    from pyserini.index.lucene import IndexReader
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    from sentence_transformers import CrossEncoder
    from tqdm import tqdm
    import numpy as np
except ImportError:
    print("Installing required packages...")
    import subprocess
    packages = [
        'pyserini', 'torch', 'transformers>=4.51.0',
        'sentence-transformers>=2.7.0', 'tqdm', 'numpy', 'FlagEmbedding'
    ]
    for pkg in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])
    from pyserini.search.lucene import LuceneSearcher
    from pyserini.index.lucene import IndexReader
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    from sentence_transformers import CrossEncoder
    from tqdm import tqdm
    import numpy as np


class ROBUST04RetrieverV2:
    """
    ROBUST04 Retrieval System V2 - Enhanced with multiple improvements
    """
    
    # Available reranker models
    RERANKER_MODELS = {
        # monoT5 models (BEST for document ranking)
        'monot5-base': 'castorini/monoT5-base-msmarco',
        'monot5-base-10k': 'castorini/monoT5-base-msmarco-10k',  # Recommended
        'monot5-large': 'castorini/monoT5-large-msmarco',
        
        # BGE models (good but not as good as monoT5 for ROBUST04)
        'bge-v2-m3': 'BAAI/bge-reranker-v2-m3',
        'bge-large': 'BAAI/bge-reranker-large',
        'bge-base': 'BAAI/bge-reranker-base',
        
        # Legacy fallback
        'minilm': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
    }
    
    def __init__(self, queries_path: str, qrels_path: str = None, output_dir: str = "./output"):
        """Initialize the retriever"""
        self.queries_path = queries_path
        self.qrels_path = qrels_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load queries
        self.queries = self._load_queries(queries_path)
        print(f"Loaded {len(self.queries)} queries")
        
        # Split: all 50 labeled for training, 199 for test
        all_sorted = sorted(self.queries.keys())
        self.train_qids = all_sorted[:50]
        self.val_qids = all_sorted[:50]  # Same for reporting
        self.test_qids = all_sorted[50:]
        print(f"Train: {len(self.train_qids)} queries (301-350), Test: {len(self.test_qids)} queries")
        
        # Load qrels
        self.qrels = None
        if qrels_path and os.path.exists(qrels_path):
            self.qrels = self._load_qrels(qrels_path)
            print(f"Loaded qrels for {len(self.qrels)} queries")
        
        # Initialize searcher
        print("Initializing Pyserini searcher with prebuilt robust04 index...")
        self.searcher = LuceneSearcher.from_prebuilt_index('robust04')
        print("Searcher initialized!")
        
        # Device setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Model caches
        self.monot5_model = None
        self.monot5_tokenizer = None
        self.cross_encoder = None
    
    def _load_queries(self, path: str) -> Dict[str, str]:
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
                    else:
                        parts = line.split(None, 1)
                        if len(parts) >= 2:
                            qid, query_text = parts[0], parts[1]
                            queries[qid] = query_text
        return queries
    
    def _load_qrels(self, path: str) -> Dict[str, Dict[str, int]]:
        """Load relevance judgments"""
        qrels = defaultdict(dict)
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                    qrels[qid][docid] = rel
        return dict(qrels)
    
    def _write_trec_run(self, results: Dict[str, List[Tuple[str, float]]], 
                        run_name: str, output_path: str):
        """Write results in TREC format"""
        with open(output_path, 'w') as f:
            for qid in sorted(results.keys(), key=lambda x: int(x) if x.isdigit() else x):
                for rank, (docid, score) in enumerate(results[qid][:1000], 1):
                    f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {run_name}\n")
        print(f"Saved results to {output_path}")
    
    def compute_map(self, results: Dict[str, List[Tuple[str, float]]], 
                    qrels: Dict[str, Dict[str, int]] = None) -> float:
        """Compute Mean Average Precision"""
        if qrels is None:
            qrels = self.qrels
        if qrels is None:
            return 0.0
        
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
            aps.append(ap)
        
        return np.mean(aps) if aps else 0.0

    # ============================================================
    # TEXT EXTRACTION (Critical for SGML-formatted ROBUST04 docs)
    # ============================================================
    
    def _extract_text_robust(self, raw_content: str) -> str:
        """Extract clean text from ROBUST04 SGML-formatted documents."""
        if not raw_content:
            return ""
        
        text = raw_content
        text = text.replace('\x00', '')
        text = re.sub(r'<!--.*?-->', ' ', text, flags=re.DOTALL)
        
        # Extract title
        title = ""
        title_patterns = [
            r'<HEAD[^>]*>(.*?)</HEAD>',
            r'<TI[^>]*>(.*?)</TI>',
            r'<HL[^>]*>(.*?)</HL>',
            r'<HEADLINE[^>]*>(.*?)</HEADLINE>',
            r'<HEADER[^>]*>(.*?)</HEADER>',
        ]
        for pattern in title_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                break
        
        # Extract body
        body = ""
        body_patterns = [
            r'<TEXT[^>]*>(.*?)</TEXT>',
            r'<LP[^>]*>(.*?)</LP>',
            r'<LEADPARA[^>]*>(.*?)</LEADPARA>',
            r'<BODY[^>]*>(.*?)</BODY>',
        ]
        for pattern in body_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                body = match.group(1).strip()
                break
        
        # Fallback
        if not body:
            clean_text = re.sub(r'<[^>]+>', ' ', text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            if len(clean_text) > 100:
                body = clean_text[100:]
                if not title:
                    title = clean_text[:100]
            else:
                body = clean_text
        
        def clean_text_segment(s: str) -> str:
            s = re.sub(r'<[^>]+>', ' ', s)
            s = re.sub(r'\s+', ' ', s)
            return s.strip()
        
        title = clean_text_segment(title)
        body = clean_text_segment(body)
        
        combined = f"{title}. {body}" if title else body
        if len(combined) > 50:
            space_ratio = combined.count(' ') / len(combined)
            if space_ratio > 0.4:
                combined = re.sub(r'(?<=\w) (?=\w)', '', combined)
        
        return combined

    # ============================================================
    # QUERY-DEPENDENT FUSION WEIGHTS (NEW in V2)
    # ============================================================
    
    def get_query_dependent_weights(self, query: str) -> Tuple[float, float]:
        """
        Determine fusion weights based on query characteristics.
        
        Short queries (1-2 words): BM25 better → higher BM25 weight
        Long queries (5+ words): Neural better → more balanced
        Entity/specific queries: BM25 better
        Conceptual queries: Neural better
        
        Returns: (bm25_weight, neural_weight)
        """
        words = query.lower().split()
        num_words = len(words)
        
        # Check for entity-like patterns (capitalized words, numbers, specific terms)
        has_numbers = any(char.isdigit() for char in query)
        
        # Check for conceptual/abstract terms
        conceptual_terms = ['impact', 'effect', 'influence', 'relationship', 'cause', 
                          'result', 'consequence', 'how', 'why', 'what']
        is_conceptual = any(term in words for term in conceptual_terms)
        
        if num_words <= 2:
            # Very short queries - BM25 is usually better
            return (1.8, 0.6)
        elif num_words >= 6:
            # Long queries - Neural can understand context better
            if is_conceptual:
                return (1.2, 1.0)
            else:
                return (1.4, 0.9)
        elif has_numbers:
            # Queries with numbers (dates, quantities) - BM25 better
            return (1.7, 0.7)
        elif is_conceptual:
            # Conceptual queries - Neural better
            return (1.3, 1.0)
        else:
            # Default balanced
            return (1.5, 0.8)

    # ============================================================
    # monoT5 RERANKER (NEW in V2 - SOTA for document ranking)
    # ============================================================
    
    def _load_monot5(self, model_name: str = 'castorini/monoT5-base-msmarco-10k'):
        """Load monoT5 model for reranking"""
        if self.monot5_model is None:
            print(f"Loading monoT5 model: {model_name}")
            self.monot5_tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.monot5_model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.monot5_model.to(self.device)
            self.monot5_model.eval()
            print("✓ monoT5 loaded successfully")
        return self.monot5_model, self.monot5_tokenizer
    
    def _monot5_score(self, query: str, documents: List[str], batch_size: int = 16) -> List[float]:
        """
        Score query-document pairs using monoT5.
        
        monoT5 uses the prompt: "Query: {q} Document: {d} Relevant:"
        and outputs probability of "true" vs "false" token.
        """
        model, tokenizer = self._load_monot5()
        
        scores = []
        true_token_id = tokenizer.encode('true', add_special_tokens=False)[0]
        false_token_id = tokenizer.encode('false', add_special_tokens=False)[0]
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            
            # Prepare inputs
            inputs = []
            for doc in batch_docs:
                # Truncate document to fit in context
                doc_truncated = doc[:1500]  # ~400 tokens
                prompt = f"Query: {query} Document: {doc_truncated} Relevant:"
                inputs.append(prompt)
            
            # Tokenize
            encoded = tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                # Get logits for the first generated token
                logits = outputs.scores[0]  # Shape: (batch_size, vocab_size)
                
                # Get probabilities for true/false
                true_logits = logits[:, true_token_id]
                false_logits = logits[:, false_token_id]
                
                # Softmax to get probability
                probs = torch.softmax(torch.stack([false_logits, true_logits], dim=1), dim=1)
                true_probs = probs[:, 1].cpu().numpy()
                
                scores.extend(true_probs.tolist())
        
        return scores
    
    # ============================================================
    # METHOD 1: BM25 + RM3 (Query Expansion)
    # ============================================================
    
    def run_bm25_rm3(self, k1: float = 0.7, b: float = 0.4,
                     fb_terms: int = 50, fb_docs: int = 5, 
                     original_weight: float = 0.5,
                     hits: int = 1000) -> Dict[str, List[Tuple[str, float]]]:
        """Run BM25 with RM3 query expansion"""
        print(f"\n{'='*60}")
        print("METHOD 1: BM25 + RM3 Query Expansion")
        print(f"{'='*60}")
        print(f"Parameters: k1={k1}, b={b}, fb_terms={fb_terms}, "
              f"fb_docs={fb_docs}, original_weight={original_weight}")
        
        # Validate on training set first
        if self.qrels:
            print("\n--- Validating on training queries FIRST ---")
            self.searcher.set_bm25(k1, b)
            self.searcher.set_rm3(fb_terms, fb_docs, original_weight)
            
            train_results = {}
            for qid in self.train_qids:
                hits_list = self.searcher.search(self.queries[qid], k=hits)
                train_results[qid] = [(hit.docid, hit.score) for hit in hits_list]
            
            self.searcher.unset_rm3()
            train_map = self.compute_map(train_results)
            print(f"✓ Train MAP ({len(self.train_qids)} queries): {train_map:.4f}")
            print("--- Now running on 199 test queries ---\n")
        
        # Run on test queries
        self.searcher.set_bm25(k1, b)
        self.searcher.set_rm3(fb_terms, fb_docs, original_weight)
        
        results = {}
        for qid in tqdm(self.test_qids, desc="BM25+RM3"):
            query = self.queries[qid]
            hits_list = self.searcher.search(query, k=hits)
            results[qid] = [(hit.docid, hit.score) for hit in hits_list]
        
        self.searcher.unset_rm3()
        return results
    
    # ============================================================
    # METHOD 2: Neural Reranking (V2 - with monoT5 option)
    # ============================================================
    
    def run_neural_reranking(self, model_name: str = 'auto',
                             initial_hits: int = 250,  # Increased from 200
                             final_hits: int = 1000,
                             batch_size: int = 32,
                             use_monot5: bool = True) -> Dict[str, List[Tuple[str, float]]]:
        """
        Two-stage retrieval with neural reranking.
        
        V2 Improvements:
        - monoT5 option (better for document ranking)
        - Deeper reranking (250 docs instead of 150)
        - Better score handling
        """
        print(f"\n{'='*60}")
        print("METHOD 2: Neural Reranking with Cross-Encoder")
        print(f"{'='*60}")
        
        # Model selection
        if use_monot5:
            print("Using monoT5 (SOTA for document ranking)")
            model_type = 'monot5'
            try:
                self._load_monot5()
            except Exception as e:
                print(f"⚠ Failed to load monoT5: {e}")
                print("Falling back to BGE cross-encoder...")
                use_monot5 = False
                model_type = 'cross-encoder'
        
        if not use_monot5:
            # Cross-encoder fallback
            model_type = 'cross-encoder'
            if model_name == 'auto':
                model_priority = ['bge-v2-m3', 'bge-large', 'minilm']
                for model_key in model_priority:
                    try:
                        full_model_name = self.RERANKER_MODELS.get(model_key, model_key)
                        print(f"Trying model: {full_model_name}...")
                        self.cross_encoder = CrossEncoder(full_model_name, max_length=512, device=self.device)
                        print(f"✓ Successfully loaded: {full_model_name}")
                        model_name = full_model_name
                        break
                    except Exception as e:
                        print(f"✗ Failed to load {full_model_name}: {e}")
                        continue
            else:
                full_model_name = self.RERANKER_MODELS.get(model_name, model_name)
                self.cross_encoder = CrossEncoder(full_model_name, max_length=512, device=self.device)
                model_name = full_model_name
        
        print(f"Model type: {model_type}")
        print(f"Initial BM25 hits: {initial_hits}, Batch size: {batch_size}")
        
        # Configure BM25 for initial retrieval
        self.searcher.set_bm25(0.9, 0.4)
        
        # Validate first
        if self.qrels:
            print("\n--- Validating on training queries FIRST ---")
            val_results = {}
            for qid in tqdm(self.val_qids[:10], desc="Validation"):  # Quick validation on 10
                query = self.queries[qid]
                bm25_hits = self.searcher.search(query, k=initial_hits)
                
                if not bm25_hits:
                    val_results[qid] = []
                    continue
                
                # Get documents
                docs = []
                doc_ids = []
                for hit in bm25_hits:
                    doc = self.searcher.doc(hit.docid)
                    if doc:
                        raw_content = doc.raw() if hasattr(doc, 'raw') else doc.contents()
                        if raw_content:
                            clean_content = self._extract_text_robust(raw_content)
                            docs.append(clean_content[:2000])
                            doc_ids.append(hit.docid)
                
                if docs:
                    if use_monot5:
                        scores = self._monot5_score(query, docs, batch_size=batch_size)
                    else:
                        pairs = [[query, doc] for doc in docs]
                        scores = self.cross_encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False)
                    
                    doc_scores = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
                    val_results[qid] = [(docid, float(score)) for docid, score in doc_scores]
                else:
                    val_results[qid] = [(hit.docid, hit.score) for hit in bm25_hits]
            
            val_map = self.compute_map(val_results)
            print(f"✓ Validation MAP (10 queries): {val_map:.4f}")
            print("--- Now running on 199 test queries ---\n")
        
        # Run on test queries
        results = {}
        
        for qid in tqdm(self.test_qids, desc="Neural Reranking"):
            query = self.queries[qid]
            
            # Stage 1: BM25 retrieval
            all_bm25_hits = self.searcher.search(query, k=1000)
            
            if not all_bm25_hits:
                results[qid] = []
                continue
            
            rerank_hits = all_bm25_hits[:initial_hits]
            remaining_hits = all_bm25_hits[initial_hits:]
            
            # Get documents for reranking
            docs = []
            doc_ids = []
            for hit in rerank_hits:
                doc = self.searcher.doc(hit.docid)
                if doc:
                    raw_content = doc.raw() if hasattr(doc, 'raw') else doc.contents()
                    if raw_content:
                        clean_content = self._extract_text_robust(raw_content)
                        docs.append(clean_content[:2000])
                        doc_ids.append(hit.docid)
            
            if not docs:
                results[qid] = [(hit.docid, hit.score) for hit in all_bm25_hits[:final_hits]]
                continue
            
            # Stage 2: Neural reranking
            try:
                if use_monot5:
                    scores = self._monot5_score(query, docs, batch_size=batch_size)
                else:
                    pairs = [[query, doc] for doc in docs]
                    scores = self.cross_encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n⚠ OOM error, reducing batch size to {batch_size//2}")
                    torch.cuda.empty_cache()
                    if use_monot5:
                        scores = self._monot5_score(query, docs, batch_size=batch_size//2)
                    else:
                        pairs = [[query, doc] for doc in docs]
                        scores = self.cross_encoder.predict(pairs, batch_size=batch_size//2, show_progress_bar=False)
                else:
                    raise
            
            # Combine reranked with remaining BM25
            reranked = list(zip(doc_ids, scores))
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            min_neural_score = min(scores) if scores else 0
            
            final_results = [(docid, float(score)) for docid, score in reranked]
            
            reranked_docids = set(doc_ids)
            for i, hit in enumerate(remaining_hits):
                if hit.docid not in reranked_docids:
                    adjusted_score = min_neural_score - 0.01 * (i + 1)
                    final_results.append((hit.docid, adjusted_score))
            
            results[qid] = final_results[:final_hits]
            
            # Periodic cleanup
            if self.device == 'cuda' and int(qid) % 50 == 0:
                torch.cuda.empty_cache()
        
        return results
    
    # ============================================================
    # METHOD 3: Weighted RRF Fusion (V2 - with query-dependent weights)
    # ============================================================
    
    def weighted_reciprocal_rank_fusion(
        self, 
        ranked_lists: List[Dict[str, List[Tuple[str, float]]]], 
        k: int = 30,
        weights: List[float] = None,
        use_query_weights: bool = False
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Weighted Reciprocal Rank Fusion with query-dependent option.
        
        V2 Improvements:
        - Query-dependent weights based on query characteristics
        - Optimized k parameter (30 instead of 60)
        """
        if weights is None:
            weights = [1.0] * len(ranked_lists)
        
        fused_results = {}
        
        # Get all query IDs
        all_qids = set()
        for results in ranked_lists:
            all_qids.update(results.keys())
        
        for qid in all_qids:
            # Get query-dependent weights if enabled
            if use_query_weights and qid in self.queries:
                query = self.queries[qid]
                q_weights = self.get_query_dependent_weights(query)
                # Extend to match number of ranked lists
                if len(ranked_lists) == 2:
                    current_weights = list(q_weights)
                else:
                    # For more lists, use default weights scaled by first two
                    current_weights = list(weights)
                    if len(current_weights) >= 2:
                        current_weights[0] = q_weights[0]
                        current_weights[1] = q_weights[1]
            else:
                current_weights = weights
            
            doc_scores = defaultdict(float)
            
            for weight, results in zip(current_weights, ranked_lists):
                if qid not in results:
                    continue
                
                for rank, (docid, _) in enumerate(results[qid], 1):
                    doc_scores[docid] += weight / (k + rank)
            
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            fused_results[qid] = sorted_docs[:1000]
        
        return fused_results
    
    def run_rrf_fusion(self, results_bm25: Dict, results_neural: Dict,
                       k: int = 30, 
                       weights: List[float] = [1.5, 0.8],
                       use_query_weights: bool = True) -> Dict[str, List[Tuple[str, float]]]:
        """
        Run RRF Fusion combining Neural + BM25+RM3.
        
        V2 Improvements:
        - Query-dependent weights option
        - Tuned k=30 (more aggressive top-ranking)
        - Weights [1.5, 0.8] favor BM25's better recall
        """
        print(f"\n{'='*60}")
        print("METHOD 3: RRF Fusion (Neural + BM25+RM3)")
        print(f"{'='*60}")
        print(f"k={k}, base_weights={weights}, query_dependent={use_query_weights}")
        
        # Validate on training set
        if self.qrels:
            print("\n--- Tuning RRF parameters on validation set ---")
            
            # Filter to validation qids
            val_bm25 = {qid: results_bm25[qid] for qid in self.val_qids if qid in results_bm25}
            val_neural = {qid: results_neural[qid] for qid in self.val_qids if qid in results_neural}
            
            # Test configurations
            k_values = [20, 25, 30, 40]
            weight_configs = [
                [1.0, 1.0],   # Equal
                [1.2, 1.0],   # Slight BM25 favor
                [1.5, 0.8],   # Current best
                [1.8, 0.6],   # Strong BM25 favor
            ]
            
            best_val_map = 0
            best_k = k
            best_weights = weights
            
            for test_k in k_values:
                for test_weights in weight_configs:
                    # Test with query-dependent weights
                    val_fused = self.weighted_reciprocal_rank_fusion(
                        [val_bm25, val_neural], k=test_k, weights=test_weights,
                        use_query_weights=use_query_weights
                    )
                    val_map = self.compute_map(val_fused)
                    
                    if val_map > best_val_map:
                        best_val_map = val_map
                        best_k = test_k
                        best_weights = test_weights
            
            print(f"✓ Best: k={best_k}, weights={best_weights} → VAL MAP: {best_val_map:.4f}")
            k = best_k
            weights = best_weights
        
        # Apply fusion
        print(f"\nApplying RRF with k={k}, weights={weights}, query_dependent={use_query_weights}")
        fused_results = self.weighted_reciprocal_rank_fusion(
            [results_bm25, results_neural], k=k, weights=weights,
            use_query_weights=use_query_weights
        )
        print(f"✓ Fused Neural + BM25+RM3")
        
        return fused_results
    
    # ============================================================
    # PARAMETER TUNING
    # ============================================================
    
    def tune_bm25_rm3_params(self):
        """Grid search for optimal BM25 + RM3 parameters"""
        if not self.qrels:
            print("No qrels available for tuning!")
            return {}
        
        print("\n" + "="*60)
        print("PARAMETER TUNING (BM25 + RM3)")
        print("="*60)
        
        best_map = 0
        best_params = {}
        
        k1_values = [0.7, 0.9, 1.0]
        b_values = [0.4, 0.5, 0.65]
        fb_terms_values = [10, 50, 70]
        fb_docs_values = [5, 10]
        orig_weight_values = [0.25, 0.5]
        
        total = len(k1_values) * len(b_values) * len(fb_terms_values) * len(fb_docs_values) * len(orig_weight_values)
        
        with tqdm(total=total, desc="Tuning") as pbar:
            for k1 in k1_values:
                for b in b_values:
                    for fb_terms in fb_terms_values:
                        for fb_docs in fb_docs_values:
                            for orig_weight in orig_weight_values:
                                self.searcher.set_bm25(k1, b)
                                self.searcher.set_rm3(fb_terms, fb_docs, orig_weight)
                                
                                results = {}
                                for qid in self.train_qids:
                                    hits = self.searcher.search(self.queries[qid], k=1000)
                                    results[qid] = [(hit.docid, hit.score) for hit in hits]
                                
                                self.searcher.unset_rm3()
                                
                                map_score = self.compute_map(results)
                                
                                if map_score > best_map:
                                    best_map = map_score
                                    best_params = {
                                        'k1': k1, 'b': b,
                                        'fb_terms': fb_terms, 'fb_docs': fb_docs,
                                        'original_weight': orig_weight
                                    }
                                
                                pbar.update(1)
        
        print(f"\nBest MAP: {best_map:.4f}")
        print(f"Best params: {best_params}")
        
        return best_params
    
    # ============================================================
    # MAIN EXECUTION
    # ============================================================
    
    def run_all_methods(self, tune_params: bool = False, use_monot5: bool = True):
        """Run all three methods with V2 improvements"""
        print("\n" + "="*80)
        print("ROBUST04 TEXT RETRIEVAL - V2 ENHANCED")
        print("="*80)
        print(f"Output directory: {self.output_dir}")
        print(f"Test queries: {len(self.test_qids)}")
        print(f"Using monoT5: {use_monot5}")
        
        # Parameter caching
        params_cache_file = os.path.join(self.output_dir, "best_params.json")
        
        if tune_params and self.qrels:
            if os.path.exists(params_cache_file):
                print(f"\n📁 Found cached parameters at {params_cache_file}")
                with open(params_cache_file, 'r') as f:
                    best_params = json.load(f)
                print(f"   Using cached params: {best_params}")
            else:
                best_params = self.tune_bm25_rm3_params()
                with open(params_cache_file, 'w') as f:
                    json.dump(best_params, f, indent=2)
                print(f"\n💾 Saved best parameters to {params_cache_file}")
        else:
            if os.path.exists(params_cache_file):
                with open(params_cache_file, 'r') as f:
                    best_params = json.load(f)
            else:
                best_params = {
                    'k1': 0.7, 'b': 0.4,
                    'fb_terms': 50, 'fb_docs': 5,
                    'original_weight': 0.5
                }
        
        # ============================================================
        # RUN 1: BM25 + RM3
        # ============================================================
        results_1 = self.run_bm25_rm3(
            k1=best_params.get('k1', 0.7),
            b=best_params.get('b', 0.4),
            fb_terms=best_params.get('fb_terms', 50),
            fb_docs=best_params.get('fb_docs', 5),
            original_weight=best_params.get('original_weight', 0.5)
        )
        output_1 = os.path.join(self.output_dir, "run_1.res")
        self._write_trec_run(results_1, "run_1", output_1)
        
        # ============================================================
        # RUN 2: Neural Reranking (with monoT5 option)
        # ============================================================
        results_2 = self.run_neural_reranking(
            model_name='auto',
            initial_hits=250,  # V2: Increased from 200
            batch_size=16 if use_monot5 else 32,  # Smaller batch for monoT5
            use_monot5=use_monot5
        )
        output_2 = os.path.join(self.output_dir, "run_2.res")
        self._write_trec_run(results_2, "run_2", output_2)
        
        # ============================================================
        # RUN 3: RRF Fusion with query-dependent weights
        # ============================================================
        results_3 = self.run_rrf_fusion(
            results_1, results_2,
            k=30,
            weights=[1.5, 0.8],
            use_query_weights=True  # V2: Query-dependent weights
        )
        output_3 = os.path.join(self.output_dir, "run_3.res")
        self._write_trec_run(results_3, "run_3", output_3)
        
        print("\n" + "="*80)
        print("COMPLETION SUMMARY")
        print("="*80)
        print(f"\nGenerated files:")
        print(f"  1. {output_1} - BM25 + RM3 (Query Expansion)")
        print(f"  2. {output_2} - Neural Reranking ({'monoT5' if use_monot5 else 'BGE'})")
        print(f"  3. {output_3} - RRF Fusion (Neural + BM25+RM3) ⭐ BEST")
        print("\nV2 Improvements applied:")
        print("  ✓ Deeper reranking (250 docs)")
        print("  ✓ Query-dependent fusion weights")
        print("  ✓ Optimized RRF k=30")
        if use_monot5:
            print("  ✓ monoT5 reranker (SOTA)")
        print("\nReady for submission!")
        
        return {
            'run_1': results_1,
            'run_2': results_2,
            'run_3': results_3
        }


def main():
    parser = argparse.ArgumentParser(description='ROBUST04 Ranking Competition Solution V2')
    parser.add_argument('--queries', type=str, required=True,
                        help='Path to queriesROBUST.txt')
    parser.add_argument('--qrels', type=str, default=None,
                        help='Path to qrels_50_Queries (optional)')
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--tune', action='store_true',
                        help='Run parameter tuning')
    parser.add_argument('--method', type=str, default='all',
                        choices=['all', 'bm25_rm3', 'neural', 'rrf'],
                        help='Which method to run')
    parser.add_argument('--use-monot5', action='store_true', default=True,
                        help='Use monoT5 for neural reranking (default: True)')
    parser.add_argument('--no-monot5', action='store_true',
                        help='Disable monoT5, use BGE instead')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for neural reranking')
    
    args = parser.parse_args()
    
    use_monot5 = not args.no_monot5
    
    retriever = ROBUST04RetrieverV2(
        queries_path=args.queries,
        qrels_path=args.qrels,
        output_dir=args.output
    )
    
    if args.method == 'all':
        retriever.run_all_methods(tune_params=args.tune, use_monot5=use_monot5)
    elif args.method == 'bm25_rm3':
        results = retriever.run_bm25_rm3()
        retriever._write_trec_run(results, "run_1", 
                                  os.path.join(args.output, "run_1.res"))
    elif args.method == 'neural':
        results = retriever.run_neural_reranking(use_monot5=use_monot5)
        retriever._write_trec_run(results, "run_2",
                                  os.path.join(args.output, "run_2.res"))
    elif args.method == 'rrf':
        print("RRF requires both BM25 and Neural results. Running all methods...")
        retriever.run_all_methods(tune_params=args.tune, use_monot5=use_monot5)


if __name__ == "__main__":
    main()