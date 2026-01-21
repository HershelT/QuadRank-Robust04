#!/usr/bin/env python3
"""
ROBUST04 Text Retrieval Ranking Competition Solution
=====================================================

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

# Install required packages if not present
def install_packages():
    """Install required packages"""
    packages = [
        'pyserini',
        'torch',
        'transformers>=4.51.0',  # Required for Qwen3 models
        'sentence-transformers>=2.7.0',  # Required for newer models
        'tqdm',
        'numpy',
        'FlagEmbedding',  # For BGE reranker models
    ]
    import subprocess
    for pkg in packages:
        pkg_name = pkg.split('>=')[0].split('==')[0]
        try:
            __import__(pkg_name.replace('-', '_').lower())
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

# Try to import, install if needed
try:
    from pyserini.search.lucene import LuceneSearcher
    from pyserini.index.lucene import IndexReader
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sentence_transformers import CrossEncoder
    from tqdm import tqdm
    import numpy as np
except ImportError:
    print("Installing required packages...")
    install_packages()
    from pyserini.search.lucene import LuceneSearcher
    from pyserini.index.lucene import IndexReader
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sentence_transformers import CrossEncoder
    from tqdm import tqdm
    import numpy as np


class ROBUST04Retriever:
    """
    ROBUST04 Retrieval System with multiple methods
    """
    
    def __init__(self, queries_path: str, qrels_path: str = None, output_dir: str = "./output"):
        """
        Initialize the retriever
        
        Args:
            queries_path: Path to queriesROBUST.txt
            qrels_path: Path to qrels_50_Queries (optional, for validation)
            output_dir: Directory to save output files
        """
        self.queries_path = queries_path
        self.qrels_path = qrels_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load queries
        self.queries = self._load_queries(queries_path)
        print(f"Loaded {len(self.queries)} queries")
        
        # Split into train (first 50) and test (remaining 199)
        self.train_qids = sorted(self.queries.keys())[:50]
        self.test_qids = sorted(self.queries.keys())[50:]
        print(f"Train queries: {len(self.train_qids)}, Test queries: {len(self.test_qids)}")
        
        # Load qrels for validation if provided
        self.qrels = None
        if qrels_path and os.path.exists(qrels_path):
            self.qrels = self._load_qrels(qrels_path)
            print(f"Loaded qrels for {len(self.qrels)} queries")
        
        # Initialize searcher
        print("Initializing Pyserini searcher with prebuilt robust04 index...")
        self.searcher = LuceneSearcher.from_prebuilt_index('robust04')
        print("Searcher initialized!")
        
        # Device setup for neural models
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
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
                        # Try space-separated format
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
        """
        Extract clean text from ROBUST04 SGML-formatted documents.
        
        ROBUST04 documents (TREC Disks 4 & 5) are stored in SGML format with:
        - XML-like tags: <DOC>, <DOCNO>, <TEXT>, <HEADLINE>, etc.
        - Potential null bytes from encoding issues
        - Metadata headers that should not be fed to neural models
        
        This method extracts the actual document content suitable for
        neural reranking models.
        
        Args:
            raw_content: Raw SGML document string from Pyserini
            
        Returns:
            Clean text with title and body concatenated
        """
        if not raw_content:
            return ""
        
        text = raw_content
        
        # 1. Remove null bytes (encoding artifacts that cause "s p a c e d" text)
        text = text.replace('\x00', '')
        
        # 2. Remove SGML comments
        text = re.sub(r'<!--.*?-->', ' ', text, flags=re.DOTALL)
        
        # 3. Extract title from various possible tags
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
        
        # 4. Extract body from various possible tags
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
        
        # 5. Fallback: if no structured content found, strip all tags
        if not body:
            # Remove all tags and use the middle portion (skip metadata at start)
            clean_text = re.sub(r'<[^>]+>', ' ', text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            if len(clean_text) > 100:
                body = clean_text[100:]  # Skip likely metadata
                if not title:
                    title = clean_text[:100]
            else:
                body = clean_text
        
        def clean_text_segment(s: str) -> str:
            """Remove remaining tags and normalize whitespace"""
            s = re.sub(r'<[^>]+>', ' ', s)  # Remove any remaining tags
            s = re.sub(r'\s+', ' ', s)       # Normalize whitespace
            return s.strip()
        
        title = clean_text_segment(title)
        body = clean_text_segment(body)
        
        # 6. Fix "spaced out" text from null byte artifacts
        # Detection: if >40% of characters are spaces in a long string, it's corrupted
        combined = f"{title}. {body}" if title else body
        if len(combined) > 50:
            space_ratio = combined.count(' ') / len(combined)
            if space_ratio > 0.4:
                # Remove single spaces between single characters: "w o r d" -> "word"
                combined = re.sub(r'(?<=\w) (?=\w)', '', combined)
        
        return combined

    # ============================================================
    # METHOD 1: BM25 + RM3 (Query Expansion)
    # ============================================================
    
    def run_bm25_rm3(self, k1: float = 0.9, b: float = 0.4,
                     fb_terms: int = 10, fb_docs: int = 10, 
                     original_weight: float = 0.5,
                     hits: int = 1000) -> Dict[str, List[Tuple[str, float]]]:
        """
        Run BM25 with RM3 query expansion
        
        RM3 (Relevance Model 3) performs pseudo-relevance feedback by:
        1. Retrieving top-k documents with initial query
        2. Extracting expansion terms from those documents
        3. Re-querying with expanded query
        
        Args:
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (document length normalization)
            fb_terms: Number of expansion terms for RM3
            fb_docs: Number of feedback documents for RM3
            original_weight: Weight of original query vs expansion terms
            hits: Number of documents to retrieve
        """
        print(f"\n{'='*60}")
        print("METHOD 1: BM25 + RM3 Query Expansion")
        print(f"{'='*60}")
        print(f"Parameters: k1={k1}, b={b}, fb_terms={fb_terms}, "
              f"fb_docs={fb_docs}, original_weight={original_weight}")
        
        # Configure BM25 and RM3
        self.searcher.set_bm25(k1, b)
        self.searcher.set_rm3(fb_terms, fb_docs, original_weight)
        
        results = {}
        for qid in tqdm(self.test_qids, desc="BM25+RM3"):
            query = self.queries[qid]
            hits_list = self.searcher.search(query, k=hits)
            results[qid] = [(hit.docid, hit.score) for hit in hits_list]
        
        # Disable RM3 for other methods
        self.searcher.unset_rm3()
        
        # Validate on training set
        train_results = {}
        self.searcher.set_bm25(k1, b)
        self.searcher.set_rm3(fb_terms, fb_docs, original_weight)
        for qid in self.train_qids:
            query = self.queries[qid]
            hits_list = self.searcher.search(query, k=hits)
            train_results[qid] = [(hit.docid, hit.score) for hit in hits_list]
        self.searcher.unset_rm3()
        
        if self.qrels:
            train_map = self.compute_map(train_results)
            print(f"Train MAP (50 queries): {train_map:.4f}")
        
        return results
    
    # ============================================================
    # METHOD 2: Neural Reranking with Cross-Encoder
    # ============================================================
    
    # Available models ranked by performance (2025-2026 SOTA)
    RERANKER_MODELS = {
        # Best performance (2025 SOTA) - Requires transformers>=4.51.0
        'qwen3-0.6b': 'Qwen/Qwen3-Reranker-0.6B',
        'qwen3-0.6b-cls': 'tomaarsen/Qwen3-Reranker-0.6B-seq-cls',  # Easier to use
        
        # Excellent performance (2024)
        'bge-v2-m3': 'BAAI/bge-reranker-v2-m3',
        'bge-large': 'BAAI/bge-reranker-large',
        'bge-base': 'BAAI/bge-reranker-base',
        
        # Legacy (2021) - fallback
        'minilm': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
    }
    
    def run_neural_reranking(self, model_name: str = 'auto',
                             initial_hits: int = 100, final_hits: int = 1000,
                             batch_size: int = 32) -> Dict[str, List[Tuple[str, float]]]:
        """
        Two-stage retrieval with neural reranking
        
        Stage 1: BM25 retrieval for initial candidates
        Stage 2: Cross-encoder reranking for precision
        
        Cross-encoders process query-document pairs together, allowing
        for deep semantic interaction - significantly better than bi-encoders
        but slower (hence two-stage approach).
        
        Models (best to fallback):
        - 'qwen3-0.6b-cls': Qwen3 Reranker 0.6B (June 2025 SOTA) ⭐ RECOMMENDED
        - 'bge-v2-m3': BGE Reranker v2 M3 (2024, excellent)
        - 'bge-large': BGE Reranker Large (2023, good)
        - 'minilm': MS-MARCO MiniLM (2021, legacy fallback)
        - 'auto': Try best available model
        
        Args:
            model_name: Model key or full HuggingFace model path
            initial_hits: Number of BM25 candidates to rerank
            final_hits: Final number of results to return
            batch_size: Batch size for neural model
        """
        print(f"\n{'='*60}")
        print("METHOD 2: Neural Reranking with Cross-Encoder")
        print(f"{'='*60}")
        
        # Model selection with auto-fallback
        if model_name == 'auto':
            # Try models in order of preference
            model_priority = ['qwen3-0.6b-cls', 'bge-v2-m3', 'bge-large', 'minilm']
            cross_encoder = None
            
            for model_key in model_priority:
                try:
                    full_model_name = self.RERANKER_MODELS.get(model_key, model_key)
                    print(f"Trying model: {full_model_name}...")
                    cross_encoder = CrossEncoder(full_model_name, max_length=512, device=self.device)
                    print(f"✓ Successfully loaded: {full_model_name}")
                    model_name = full_model_name
                    break
                except Exception as e:
                    print(f"✗ Failed to load {full_model_name}: {e}")
                    continue
            
            if cross_encoder is None:
                raise RuntimeError("No reranker model could be loaded!")
        else:
            # Use specified model
            full_model_name = self.RERANKER_MODELS.get(model_name, model_name)
            print(f"Loading model: {full_model_name}")
            cross_encoder = CrossEncoder(full_model_name, max_length=512, device=self.device)
            model_name = full_model_name
        
        print(f"Model: {model_name}")
        print(f"Initial BM25 hits: {initial_hits}, Batch size: {batch_size}")
        
        # Configure BM25 for initial retrieval
        self.searcher.set_bm25(0.9, 0.4)
        
        results = {}
        
        for qid in tqdm(self.test_qids, desc="Neural Reranking"):
            query = self.queries[qid]
            
            # Stage 1: BM25 retrieval
            bm25_hits = self.searcher.search(query, k=initial_hits)
            
            if not bm25_hits:
                results[qid] = []
                continue
            
            # Prepare query-document pairs
            pairs = []
            doc_ids = []
            for hit in bm25_hits:
                doc = self.searcher.doc(hit.docid)
                if doc:
                    # CRITICAL: Use robust extraction for SGML documents
                    # Raw content contains XML tags, null bytes, and metadata
                    raw_content = doc.raw() if hasattr(doc, 'raw') else doc.contents()
                    if raw_content:
                        clean_content = self._extract_text_robust(raw_content)
                        # Truncate to ~512 tokens (approx 2000 chars)
                        pairs.append([query, clean_content[:2000]])
                        doc_ids.append(hit.docid)
            
            if not pairs:
                results[qid] = [(hit.docid, hit.score) for hit in bm25_hits]
                continue
            
            # Stage 2: Cross-encoder reranking
            try:
                scores = cross_encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False)
            except RuntimeError as e:
                # Handle OOM by reducing batch size
                if "out of memory" in str(e).lower():
                    print(f"\n⚠ OOM error, reducing batch size to {batch_size//2}")
                    torch.cuda.empty_cache()
                    scores = cross_encoder.predict(pairs, batch_size=batch_size//2, show_progress_bar=False)
                else:
                    raise
            
            # Combine and sort
            doc_scores = list(zip(doc_ids, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            results[qid] = [(docid, float(score)) for docid, score in doc_scores[:final_hits]]
            
            # Periodic GPU memory cleanup
            if self.device == 'cuda' and int(qid) % 50 == 0:
                torch.cuda.empty_cache()
        
        # Clear GPU memory
        del cross_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results
    
    # ============================================================
    # METHOD 3: RRF Fusion (Reciprocal Rank Fusion)
    # ============================================================
    
    def reciprocal_rank_fusion(self, ranked_lists: List[Dict[str, List[Tuple[str, float]]]], 
                               k: int = 60) -> Dict[str, List[Tuple[str, float]]]:
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion
        
        RRF Formula: score(d) = Σ [1 / (k + rank(d))]
        
        Key advantages:
        - No score normalization needed
        - Robust to different score scales
        - Documents consistently ranked high across lists get boosted
        
        Args:
            ranked_lists: List of result dictionaries from different rankers
            k: Ranking constant (typically 60)
        """
        fused_results = {}
        
        # Get all query IDs
        all_qids = set()
        for results in ranked_lists:
            all_qids.update(results.keys())
        
        for qid in all_qids:
            doc_scores = defaultdict(float)
            
            for results in ranked_lists:
                if qid not in results:
                    continue
                
                for rank, (docid, _) in enumerate(results[qid], 1):
                    doc_scores[docid] += 1.0 / (k + rank)
            
            # Sort by fused score
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            fused_results[qid] = sorted_docs[:1000]
        
        return fused_results
    
    def run_rrf_fusion(self, k: int = 60) -> Dict[str, List[Tuple[str, float]]]:
        """
        Run RRF Fusion combining multiple BM25 variants
        
        Combines:
        1. BM25 with default parameters
        2. BM25 with tuned parameters
        3. BM25 + RM3 with different configurations
        
        This creates diversity in retrieval and RRF combines them optimally.
        """
        print(f"\n{'='*60}")
        print("METHOD 3: RRF Fusion (Hybrid Approach)")
        print(f"{'='*60}")
        print(f"RRF k parameter: {k}")
        
        ranked_lists = []
        
        # Variant 1: BM25 default (k1=0.9, b=0.4)
        print("\n  Variant 1: BM25 (k1=0.9, b=0.4)")
        self.searcher.set_bm25(0.9, 0.4)
        results_v1 = {}
        for qid in tqdm(self.test_qids, desc="  BM25 v1"):
            hits = self.searcher.search(self.queries[qid], k=1000)
            results_v1[qid] = [(hit.docid, hit.score) for hit in hits]
        ranked_lists.append(results_v1)
        
        # Variant 2: BM25 tuned (k1=0.7, b=0.65) - from Anserini research
        print("\n  Variant 2: BM25 (k1=0.7, b=0.65)")
        self.searcher.set_bm25(0.7, 0.65)
        results_v2 = {}
        for qid in tqdm(self.test_qids, desc="  BM25 v2"):
            hits = self.searcher.search(self.queries[qid], k=1000)
            results_v2[qid] = [(hit.docid, hit.score) for hit in hits]
        ranked_lists.append(results_v2)
        
        # Variant 3: BM25 + RM3 (aggressive expansion)
        print("\n  Variant 3: BM25 + RM3 (aggressive)")
        self.searcher.set_bm25(0.7, 0.65)
        self.searcher.set_rm3(70, 10, 0.25)  # Many terms, low original weight
        results_v3 = {}
        for qid in tqdm(self.test_qids, desc="  BM25+RM3"):
            hits = self.searcher.search(self.queries[qid], k=1000)
            results_v3[qid] = [(hit.docid, hit.score) for hit in hits]
        self.searcher.unset_rm3()
        ranked_lists.append(results_v3)
        
        # Variant 4: BM25 + RM3 (conservative expansion)
        print("\n  Variant 4: BM25 + RM3 (conservative)")
        self.searcher.set_bm25(0.9, 0.4)
        self.searcher.set_rm3(10, 10, 0.5)  # Fewer terms, balanced weight
        results_v4 = {}
        for qid in tqdm(self.test_qids, desc="  BM25+RM3"):
            hits = self.searcher.search(self.queries[qid], k=1000)
            results_v4[qid] = [(hit.docid, hit.score) for hit in hits]
        self.searcher.unset_rm3()
        ranked_lists.append(results_v4)
        
        # Apply RRF fusion
        print("\n  Applying Reciprocal Rank Fusion...")
        fused_results = self.reciprocal_rank_fusion(ranked_lists, k=k)
        print(f"  Fused {len(ranked_lists)} result lists")
        
        return fused_results
    
    # ============================================================
    # PARAMETER TUNING
    # ============================================================
    
    def tune_bm25_rm3_params(self):
        """
        Grid search for optimal BM25 + RM3 parameters on training set
        """
        if not self.qrels:
            print("No qrels available for tuning!")
            return
        
        print("\n" + "="*60)
        print("PARAMETER TUNING (BM25 + RM3)")
        print("="*60)
        
        best_map = 0
        best_params = {}
        
        # Parameter grid based on Anserini research
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
                                # Configure
                                self.searcher.set_bm25(k1, b)
                                self.searcher.set_rm3(fb_terms, fb_docs, orig_weight)
                                
                                # Run on training queries
                                results = {}
                                for qid in self.train_qids:
                                    hits = self.searcher.search(self.queries[qid], k=1000)
                                    results[qid] = [(hit.docid, hit.score) for hit in hits]
                                
                                self.searcher.unset_rm3()
                                
                                # Compute MAP
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
    
    def run_all_methods(self, tune_params: bool = False):
        """
        Run all three methods and save results
        """
        print("\n" + "="*80)
        print("ROBUST04 TEXT RETRIEVAL RANKING COMPETITION")
        print("="*80)
        print(f"Output directory: {self.output_dir}")
        print(f"Test queries: {len(self.test_qids)}")
        
        # Optional: Parameter tuning
        if tune_params and self.qrels:
            best_params = self.tune_bm25_rm3_params()
        else:
            # Default optimized parameters from research
            best_params = {
                'k1': 0.7, 'b': 0.65,
                'fb_terms': 70, 'fb_docs': 10,
                'original_weight': 0.25
            }
        
        # ============================================================
        # RUN 1: BM25 + RM3
        # ============================================================
        results_1 = self.run_bm25_rm3(
            k1=best_params.get('k1', 0.7),
            b=best_params.get('b', 0.65),
            fb_terms=best_params.get('fb_terms', 70),
            fb_docs=best_params.get('fb_docs', 10),
            original_weight=best_params.get('original_weight', 0.25)
        )
        output_1 = os.path.join(self.output_dir, "run_1.res")
        self._write_trec_run(results_1, "run_1", output_1)
        
        # ============================================================
        # RUN 2: Neural Reranking (2025 SOTA models with fallback)
        # ============================================================
        results_2 = self.run_neural_reranking(
            model_name='auto',  # Will try: Qwen3 → BGE-v2 → BGE-large → MiniLM
            initial_hits=100,
            batch_size=32
        )
        output_2 = os.path.join(self.output_dir, "run_2.res")
        self._write_trec_run(results_2, "run_2", output_2)
        
        # ============================================================
        # RUN 3: RRF Fusion
        # ============================================================
        results_3 = self.run_rrf_fusion(k=60)
        output_3 = os.path.join(self.output_dir, "run_3.res")
        self._write_trec_run(results_3, "run_3", output_3)
        
        print("\n" + "="*80)
        print("COMPLETION SUMMARY")
        print("="*80)
        print(f"\nGenerated files:")
        print(f"  1. {output_1} - BM25 + RM3 (Query Expansion)")
        print(f"  2. {output_2} - Neural Reranking (Cross-Encoder)")
        print(f"  3. {output_3} - RRF Fusion (Hybrid)")
        print("\nReady for submission!")
        
        return {
            'run_1': results_1,
            'run_2': results_2,
            'run_3': results_3
        }


def main():
    parser = argparse.ArgumentParser(description='ROBUST04 Ranking Competition Solution')
    parser.add_argument('--queries', type=str, required=True,
                        help='Path to queriesROBUST.txt')
    parser.add_argument('--qrels', type=str, default=None,
                        help='Path to qrels_50_Queries (optional)')
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--tune', action='store_true',
                        help='Run parameter tuning on training set')
    parser.add_argument('--method', type=str, default='all',
                        choices=['all', 'bm25_rm3', 'neural', 'rrf'],
                        help='Which method to run')
    parser.add_argument('--reranker', type=str, default='auto',
                        choices=['auto', 'qwen3-0.6b-cls', 'bge-v2-m3', 'bge-large', 'minilm'],
                        help='Reranker model (auto=try best available)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for neural reranking')
    
    args = parser.parse_args()
    
    # Initialize retriever
    retriever = ROBUST04Retriever(
        queries_path=args.queries,
        qrels_path=args.qrels,
        output_dir=args.output
    )
    
    # Run methods
    if args.method == 'all':
        retriever.run_all_methods(tune_params=args.tune)
    elif args.method == 'bm25_rm3':
        results = retriever.run_bm25_rm3()
        retriever._write_trec_run(results, "run_1", 
                                  os.path.join(args.output, "run_1.res"))
    elif args.method == 'neural':
        results = retriever.run_neural_reranking(
            model_name=args.reranker,
            batch_size=args.batch_size
        )
        retriever._write_trec_run(results, "run_2",
                                  os.path.join(args.output, "run_2.res"))
    elif args.method == 'rrf':
        results = retriever.run_rrf_fusion()
        retriever._write_trec_run(results, "run_3",
                                  os.path.join(args.output, "run_3.res"))


if __name__ == "__main__":
    main()
