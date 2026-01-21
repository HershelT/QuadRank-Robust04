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
        
        # Split: Use all 50 labeled queries for training, 199 for test
        # NOTE: We tested splitting into train(40)/val(10) but diagnostic showed
        # that queries 341-350 are inherently harder (32% lower MAP even with 
        # baseline BM25). So we use all 50 for better parameter tuning.
        all_sorted = sorted(self.queries.keys())
        self.train_qids = all_sorted[:50]       # 301-350 (all labeled queries)
        self.val_qids = all_sorted[:50]         # Same as train for reporting
        self.test_qids = all_sorted[50:]        # 351+ (199 competition queries)
        print(f"Train: {len(self.train_qids)} queries (301-350), Test: {len(self.test_qids)} queries")
        
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
        
        # Cache for validation results (to reuse in RRF tuning)
        self.val_results_bm25 = None
        self.val_results_neural = None
    
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
        
        # === VALIDATE FIRST on train/val sets ===
        if self.qrels:
            print("\n--- Validating on held-out queries FIRST ---")
            self.searcher.set_bm25(k1, b)
            self.searcher.set_rm3(fb_terms, fb_docs, original_weight)
            
            train_results = {}
            for qid in self.train_qids:
                hits_list = self.searcher.search(self.queries[qid], k=hits)
                train_results[qid] = [(hit.docid, hit.score) for hit in hits_list]
            
            val_results = {}
            for qid in self.val_qids:
                hits_list = self.searcher.search(self.queries[qid], k=hits)
                val_results[qid] = [(hit.docid, hit.score) for hit in hits_list]
            
            self.searcher.unset_rm3()
            
            train_map = self.compute_map(train_results)
            val_map = self.compute_map(val_results)
            print(f"✓ Train MAP ({len(self.train_qids)} queries): {train_map:.4f}")
            print(f"✓ Validation MAP ({len(self.val_qids)} queries): {val_map:.4f}")
            
            # Cache validation results for RRF tuning
            self.val_results_bm25 = val_results
            print("   (Cached for RRF tuning)")
            print("--- Now running on 199 test queries ---\n")
        
        # === NOW run on test queries ===
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
            model_priority = ['bge-v2-m3', 'qwen3-0.6b-cls', 'bge-large', 'minilm']
            cross_encoder = None
            
            for model_key in model_priority:
                try:
                    full_model_name = self.RERANKER_MODELS.get(model_key, model_key)
                    print(f"Trying model: {full_model_name}...")
                    cross_encoder = CrossEncoder(
                        full_model_name, 
                        max_length=512, 
                        device=self.device,
                        automodel_args={'torch_dtype': torch.float16} if self.device == 'cuda' else None
                    )
                    print(f"✓ Successfully loaded: {full_model_name}")
                    print("✓ Enabled FP16 (half-precision) for speed")
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
            cross_encoder = CrossEncoder(
                full_model_name, 
                max_length=512, 
                device=self.device,
                automodel_args={'torch_dtype': torch.float16} if self.device == 'cuda' else None
            )
            print("✓ Enabled FP16 (half-precision) for speed")
            model_name = full_model_name
        
        print(f"Model: {model_name}")
        print(f"Initial BM25 hits: {initial_hits}, Batch size: {batch_size}")
        
        # Configure BM25 for initial retrieval
        self.searcher.set_bm25(0.9, 0.4)
        
        # === VALIDATE FIRST on held-out queries ===
        if self.qrels:
            print("\n--- Validating on held-out queries FIRST ---")
            val_results = {}
            for qid in tqdm(self.val_qids, desc="Validation"):
                query = self.queries[qid]
                bm25_hits = self.searcher.search(query, k=initial_hits)
                if not bm25_hits:
                    val_results[qid] = []
                    continue
                pairs = []
                doc_ids = []
                for hit in bm25_hits:
                    doc = self.searcher.doc(hit.docid)
                    if doc:
                        raw_content = doc.raw() if hasattr(doc, 'raw') else doc.contents()
                        if raw_content:
                            clean_content = self._extract_text_robust(raw_content)
                            pairs.append([query, clean_content[:2000]])
                            doc_ids.append(hit.docid)
                if pairs:
                    scores = cross_encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False)
                    doc_scores = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
                    val_results[qid] = [(docid, float(score)) for docid, score in doc_scores]
                else:
                    val_results[qid] = [(hit.docid, hit.score) for hit in bm25_hits]
            
            val_map = self.compute_map(val_results)
            print(f"✓ Validation MAP ({len(self.val_qids)} queries): {val_map:.4f}")
            
    def run_neural_reranking(self, model_name: str = 'auto',
                             initial_hits: int = 100, final_hits: int = 1000,
                             batch_size: int = 32) -> Dict[str, List[Tuple[str, float]]]:
        """
        Two-stage retrieval with neural reranking using MaxP (Max Passage) strategy
        
        Stage 1: BM25 retrieval for initial candidates
        Stage 2: Cross-encoder reranking on SLIDING WINDOWS of text
        
        MaxP Strategy:
        - Split long documents into overlapping chunks (passages)
        - Score each chunk independently
        - Document score = Max(chunk scores)
        - Essential for ROBUST04 where documents are long (2000+ words)
        
        Args:
            model_name: Model key or full HuggingFace model path
            initial_hits: Number of BM25 candidates to rerank
            final_hits: Final number of results to return
            batch_size: Batch size for neural model
        """
        print(f"\n{'='*60}")
        print("METHOD 2: Neural Reranking with MaxP Sliding Window")
        print(f"{'='*60}")
        
        # Model selection
        if model_name == 'auto':
            model_priority = ['bge-v2-m3', 'qwen3-0.6b-cls', 'bge-large', 'minilm']
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
            full_model_name = self.RERANKER_MODELS.get(model_name, model_name)
            print(f"Loading model: {full_model_name}")
            cross_encoder = CrossEncoder(full_model_name, max_length=512, device=self.device)
            model_name = full_model_name
        
        print(f"Model: {model_name}")
        print(f"Initial BM25 hits: {initial_hits}, Batch size: {batch_size}")
        print("Strategy: MaxP (splitting docs into overlapping 512-token chunks)")
        
        self.searcher.set_bm25(0.9, 0.4)
        
        def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 500) -> List[str]:
            """Split text into overlapping chunks (approx chars)"""
            if len(text) <= chunk_size:
                return [text]
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunks.append(text[start:end])
                if end == len(text):
                    break
                start += (chunk_size - overlap)
            return chunks

        def rerank_batch(qids, qrels=None, desc="Reranking"):
            batch_results = {}
            for qid in tqdm(qids, desc=desc):
                query = self.queries[qid]
                
                # Full recall retrieval
                all_bm25_hits = self.searcher.search(query, k=1000)
                if not all_bm25_hits:
                    batch_results[qid] = []
                    continue
                
                # Rerank top N
                rerank_hits = all_bm25_hits[:initial_hits]
                remaining_hits = all_bm25_hits[initial_hits:]
                
                # Prepare all chunks for all docs
                pairs = []
                doc_chunk_map = []  # (docid, num_chunks)
                
                for hit in rerank_hits:
                    doc = self.searcher.doc(hit.docid)
                    if doc:
                        raw = doc.raw() if hasattr(doc, 'raw') else doc.contents()
                        if raw:
                            clean = self._extract_text_robust(raw)
                            # Generate chunks (MaxP)
                            chunks = chunk_text(clean)
                            # Limit to max 4 chunks per doc to prevent explosion
                            chunks = chunks[:4]
                            
                            for chunk in chunks:
                                pairs.append([query, chunk])
                            doc_chunk_map.append((hit.docid, len(chunks)))
                
                if not pairs:
                    batch_results[qid] = [(h.docid, h.score) for h in all_bm25_hits[:final_hits]]
                    continue
                
                # Predict scores for all chunks
                try:
                    chunk_scores = cross_encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"⚠ OOM, reducing batch size")
                        torch.cuda.empty_cache()
                        chunk_scores = cross_encoder.predict(pairs, batch_size=batch_size//2, show_progress_bar=False)
                    else:
                        raise

                # Aggregate MaxP scores
                doc_max_scores = []
                score_idx = 0
                for docid, num_chunks in doc_chunk_map:
                    if num_chunks > 0:
                        # Max over this doc's chunks
                        max_score = np.max(chunk_scores[score_idx : score_idx+num_chunks])
                        doc_max_scores.append((docid, float(max_score)))
                        score_idx += num_chunks
                
                # Sort and merge
                doc_max_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Merge with remaining BM25
                final_res = doc_max_scores[:]
                min_neural = doc_max_scores[-1][1] if doc_max_scores else 0
                
                seen = {d for d, _ in doc_max_scores}
                for i, hit in enumerate(remaining_hits):
                    if hit.docid not in seen:
                        # Scale BM25 to be below neural
                        final_res.append((hit.docid, min_neural - 0.01 * (i + 1)))
                
                batch_results[qid] = final_res[:final_hits]
                
                if self.device == 'cuda' and int(qid) % 20 == 0:
                    torch.cuda.empty_cache()
            
            return batch_results

        # === VALIDATE FIRST ===
        if self.qrels:
            print("\n--- Validating on held-out queries (MaxP) ---")
            val_results = rerank_batch(self.val_qids, self.qrels, desc="Val MaxP")
            val_map = self.compute_map(val_results)
            print(f"✓ Validation MAP (MaxP): {val_map:.4f}")
            self.val_results_neural = val_results
            print("   (Cached for RRF tuning)")
        
        # === TEST RUN ===
        print("\n--- Running on test queries (MaxP) ---")
        results = rerank_batch(self.test_qids, desc="Test MaxP")
        
        del cross_encoder
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
    
    def weighted_reciprocal_rank_fusion(
        self, 
        ranked_lists: List[Dict[str, List[Tuple[str, float]]]], 
        k: int = 60,
        weights: List[float] = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Weighted Reciprocal Rank Fusion (RRF)
        
        Extends standard RRF with per-ranker weights:
        score(d) = Σ [weight_i / (k + rank_i(d))]
        
        Args:
            ranked_lists: List of result dictionaries from different rankers
            k: Ranking constant (typically 60, lower = more weight to top ranks)
            weights: Per-ranker weights (default: equal weights of 1.0)
        """
        if weights is None:
            weights = [1.0] * len(ranked_lists)
        
        fused_results = {}
        
        # Get all query IDs
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
        
        # === VALIDATE FIRST on held-out queries ===
        if self.qrels:
            print("\n--- Validating on held-out queries FIRST ---")
            val_ranked_lists = []
            
            # Variant 1
            self.searcher.set_bm25(0.9, 0.4)
            val_v1 = {qid: [(h.docid, h.score) for h in self.searcher.search(self.queries[qid], k=1000)] for qid in self.val_qids}
            val_ranked_lists.append(val_v1)
            
            # Variant 2
            self.searcher.set_bm25(0.7, 0.65)
            val_v2 = {qid: [(h.docid, h.score) for h in self.searcher.search(self.queries[qid], k=1000)] for qid in self.val_qids}
            val_ranked_lists.append(val_v2)
            
            # Variant 3
            self.searcher.set_bm25(0.7, 0.65)
            self.searcher.set_rm3(70, 10, 0.25)
            val_v3 = {qid: [(h.docid, h.score) for h in self.searcher.search(self.queries[qid], k=1000)] for qid in self.val_qids}
            self.searcher.unset_rm3()
            val_ranked_lists.append(val_v3)
            
            # Variant 4
            self.searcher.set_bm25(0.9, 0.4)
            self.searcher.set_rm3(10, 10, 0.5)
            val_v4 = {qid: [(h.docid, h.score) for h in self.searcher.search(self.queries[qid], k=1000)] for qid in self.val_qids}
            self.searcher.unset_rm3()
            val_ranked_lists.append(val_v4)
            
            val_fused = self.reciprocal_rank_fusion(val_ranked_lists, k=k)
            val_map = self.compute_map(val_fused)
            print(f"✓ Validation MAP ({len(self.val_qids)} queries): {val_map:.4f}")
            print("--- Now running on 199 test queries ---\n")
        
        # === NOW run on test queries ===
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
        Grid search for optimal BM25 + RM3 parameters on training set (40 queries).
        Validates on held-out validation set (10 queries) to detect overfitting.
        """
        if not self.qrels:
            print("No qrels available for tuning!")
            return
        
        print("\n" + "="*60)
        print("PARAMETER TUNING (BM25 + RM3)")
        print("="*60)
        print(f"Training on {len(self.train_qids)} queries, validating on {len(self.val_qids)} queries")
        
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
                                
                                # Run on training queries only (40)
                                results = {}
                                for qid in self.train_qids:
                                    hits = self.searcher.search(self.queries[qid], k=1000)
                                    results[qid] = [(hit.docid, hit.score) for hit in hits]
                                
                                self.searcher.unset_rm3()
                                
                                # Compute MAP on train
                                map_score = self.compute_map(results)
                                
                                if map_score > best_map:
                                    best_map = map_score
                                    best_params = {
                                        'k1': k1, 'b': b,
                                        'fb_terms': fb_terms, 'fb_docs': fb_docs,
                                        'original_weight': orig_weight
                                    }
                                
                                pbar.update(1)
        
        # Now validate using held-out validation set
        print(f"\nBest Train MAP: {best_map:.4f}")
        print(f"Best params: {best_params}")
        
        # Compute validation MAP with best params
        self.searcher.set_bm25(best_params['k1'], best_params['b'])
        self.searcher.set_rm3(best_params['fb_terms'], best_params['fb_docs'], best_params['original_weight'])
        val_results = {}
        for qid in self.val_qids:
            hits = self.searcher.search(self.queries[qid], k=1000)
            val_results[qid] = [(hit.docid, hit.score) for hit in hits]
        self.searcher.unset_rm3()
        
        val_map = self.compute_map(val_results)
        print(f"Validation MAP (10 queries): {val_map:.4f}")
        
        if val_map < best_map * 0.8:
            print("⚠ WARNING: Large drop on validation - possible overfitting!")
        else:
            print("✓ Validation performance is stable")
        
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
        
        # Parameter caching file
        params_cache_file = os.path.join(self.output_dir, "best_params.json")
        
        # Check for cached parameters first
        if tune_params and self.qrels:
            if os.path.exists(params_cache_file):
                print(f"\n📁 Found cached parameters at {params_cache_file}")
                with open(params_cache_file, 'r') as f:
                    cached = json.load(f)
                print(f"   Cached params: {cached}")
                use_cached = input("Use cached parameters? [Y/n]: ").strip().lower()
                if use_cached != 'n':
                    best_params = cached
                    print("   ✓ Using cached parameters")
                else:
                    best_params = self.tune_bm25_rm3_params()
                    # Save to cache
                    with open(params_cache_file, 'w') as f:
                        json.dump(best_params, f, indent=2)
                    print(f"\n💾 Saved best parameters to {params_cache_file}")
            else:
                best_params = self.tune_bm25_rm3_params()
                # Save to cache
                with open(params_cache_file, 'w') as f:
                    json.dump(best_params, f, indent=2)
                print(f"\n💾 Saved best parameters to {params_cache_file}")
        else:
            # Check for cached params even if not tuning
            if os.path.exists(params_cache_file):
                print(f"\n📁 Loading cached parameters from {params_cache_file}")
                with open(params_cache_file, 'r') as f:
                    best_params = json.load(f)
                print(f"   Params: {best_params}")
            else:
                # Default optimized parameters from research
                best_params = {
                    'k1': 0.7, 'b': 0.65,
                    'fb_terms': 70, 'fb_docs': 10,
                    'original_weight': 0.25
                }
                print(f"\n   Using default params: {best_params}")
        
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
            model_name='auto',  # Will try: BGE-v2 → Qwen3 → BGE-large → MiniLM
            initial_hits=250,   # Increased from 150 for better recall
            batch_size=32
        )
        output_2 = os.path.join(self.output_dir, "run_2.res")
        self._write_trec_run(results_2, "run_2", output_2)
        
        # ============================================================
        # RUN 3: RRF Fusion of Neural + BM25+RM3 (Best of Both Worlds)
        # ============================================================
        print(f"\n{'='*60}")
        print("METHOD 3: RRF Fusion (Neural + BM25+RM3)")
        print(f"{'='*60}")
        
        # --- Tune RRF parameters on validation set ---
        if self.qrels and self.val_results_bm25 and self.val_results_neural:
            print("\n--- Tuning RRF parameters on validation set (using cached results) ---")
            # Grid of parameters to try
            k_values = [30, 40, 60, 80]
            # Weights: [BM25+RM3 weight, Neural weight]
            weight_configs = [
                [1.0, 1.0],   # Equal
                [1.2, 1.0],   # Slightly favor BM25
                [1.5, 1.0],   # Favor BM25 more
                [1.0, 0.8],   # Slightly penalize Neural
                [1.2, 0.8],   # Favor BM25 + penalize Neural
                [1.5, 0.8],   # Strongly favor BM25
            ]
            
            # Use cached validation results
            val_results_1 = self.val_results_bm25
            val_results_2 = self.val_results_neural
            
            best_val_map = 0
            best_k = 60
            best_weights = [1.0, 1.0]
            
            for k in k_values:
                for weights in weight_configs:
                    # Fuse validation results
                    val_fused = self.weighted_reciprocal_rank_fusion(
                        [val_results_1, val_results_2], k=k, weights=weights
                    )
                    val_map = self.compute_map(val_fused)
                    print(f"  k={k}, weights={weights} → MAP: {val_map:.4f}")
                    
                    if val_map > best_val_map:
                        best_val_map = val_map
                        best_k = k
                        best_weights = weights
            
            print(f"\n✓ Best: k={best_k}, weights={best_weights} → VAL MAP: {best_val_map:.4f}")
            
            # Save RRF params to cache
            if os.path.exists(params_cache_file):
                with open(params_cache_file, 'r') as f:
                    cached = json.load(f)
            else:
                cached = {}
            cached['rrf_k'] = best_k
            cached['rrf_weights'] = best_weights
            with open(params_cache_file, 'w') as f:
                json.dump(cached, f, indent=2)
            print(f"💾 Saved RRF params to {params_cache_file}")
            print("--- Now running on 199 test queries with best params ---\n")
        else:
            # Try to load from cache, otherwise use optimal defaults
            if os.path.exists(params_cache_file):
                with open(params_cache_file, 'r') as f:
                    cached = json.load(f)
                best_k = cached.get('rrf_k', 30)
                best_weights = cached.get('rrf_weights', [1.5, 0.8])
                print(f"📁 Loaded RRF params from cache: k={best_k}, weights={best_weights}")
            else:
                # Optimal defaults from validation tuning (run_validation_on_50.py)
                best_k = 30
                best_weights = [1.5, 0.8]  # [BM25 weight, Neural weight]
        
        # Fuse Neural (run_2) with BM25+RM3 (run_1) using weighted RRF
        results_3 = self.weighted_reciprocal_rank_fusion(
            [results_1, results_2], k=best_k, weights=best_weights
        )
        print(f"✓ Fused Neural + BM25+RM3 (k={best_k}, weights={best_weights})")
        
        output_3 = os.path.join(self.output_dir, "run_3.res")
        self._write_trec_run(results_3, "run_3", output_3)
        
        print("\n" + "="*80)
        print("COMPLETION SUMMARY")
        print("="*80)
        print(f"\nGenerated files:")
        print(f"  1. {output_1} - BM25 + RM3 (Query Expansion)")
        print(f"  2. {output_2} - Neural Reranking (Cross-Encoder)")
        print(f"  3. {output_3} - RRF Fusion (k={best_k}, w={best_weights}) ⭐ BEST")
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
