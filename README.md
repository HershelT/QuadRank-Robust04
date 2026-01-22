# ROBUST04 Text Retrieval Ranking System

A multi-method information retrieval system for the ROBUST04 document collection, implementing three distinct ranking approaches to maximize Mean Average Precision (MAP). Developed for the Text Retrieval and Search Engines course ranking competition.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command-Line Interface](#command-line-interface)
- [Methods](#methods)
  - [Method 1: BM25 + RM3 Query Expansion](#method-1-bm25--rm3-query-expansion)
  - [Method 2: Neural Reranking](#method-2-neural-reranking)
  - [Method 3: Reciprocal Rank Fusion](#method-3-reciprocal-rank-fusion)
- [Output Format](#output-format)
- [Expected Performance](#expected-performance)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

This system implements a complete retrieval pipeline for the TREC ROBUST04 test collection, which consists of 528,155 newswire documents and 249 queries. The implementation provides three complementary retrieval methods:

1. **BM25 + RM3**: Classical probabilistic retrieval with pseudo-relevance feedback
2. **Neural Reranking**: Two-stage retrieval using transformer-based cross-encoders
3. **RRF Fusion (Neural + BM25+RM3)**: Hybrid approach combining neural precision with BM25 recall ⭐ Best performer

The system is designed to run on consumer hardware with 8GB VRAM and produces output in standard TREC format for evaluation.

### Results Summary

| Run | Method | MAP | P@10 | NDCG@20 |
|-----|--------|-----|------|--------|
| **run_3** | **RRF Fusion (k=30, weights=[1.5, 1.0])** | **0.3144** ⭐ | 0.5095 | 0.4778 |
| run_1 | BM25 + RM3 | 0.3006 | 0.4683 | 0.4385 |
| run_2 | Neural Reranking (BGE-v2, MaxP) | 0.2714 | 0.4864 | 0.4542 |

### Project Structure

```
robust04_ranking/
├── robust04_ranking_solution.py    # Main implementation
├── README.md                       # This file
├── files/                          # Input files (user-provided)
│   ├── queriesROBUST.txt          # Query file (249 queries)
│   └── qrels_50_Queries           # Relevance judgments (50 queries)
└── output/                         # Generated results
    ├── run_1.res                  # BM25 + RM3 results
    ├── run_2.res                  # Neural reranking results
    └── run_3.res                  # RRF fusion results
```

---

## Requirements

### Hardware

- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended for neural reranking)
- **RAM**: 16GB minimum, 32GB+ recommended
- **Storage**: 5GB free space (for Pyserini index and models)

### Software

- Python 3.8+
- Java 21 (required by Pyserini/Anserini)
- CUDA 11.0+ (for GPU acceleration)

### Python Dependencies

```
pyserini>=0.35.0
torch>=2.0.0
transformers>=4.51.0
sentence-transformers>=2.7.0
FlagEmbedding
tqdm
numpy
```

---

## Installation

### 1. Install Java 21

Pyserini requires Java 21. Install via your package manager:

```bash
# Ubuntu/Debian
sudo apt install openjdk-21-jdk

# macOS (Homebrew)
brew install openjdk@21

# Windows: Download from https://adoptium.net/
```

Verify installation:
```bash
java -version
```

### 2. Install Python Dependencies

```bash
pip install pyserini torch transformers>=4.51.0 sentence-transformers>=2.7.0 tqdm numpy

# Optional: For BGE reranker models
pip install FlagEmbedding
```

### 3. Verify GPU Setup (Optional)

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### 4. First Run Index Download

On first execution, Pyserini will automatically download the ROBUST04 index (~1.7GB). This only occurs once.

---

## Quick Start

### Basic Usage

```bash
python robust04_ranking_solution.py --queries files/queriesROBUST.txt --output output/
```

### Full Run with Validation

```bash
python robust04_ranking_solution.py \
    --queries files/queriesROBUST.txt \
    --qrels files/qrels_50_Queries \
    --output output/ \
    --tune
```

### Results

After execution, three result files will be generated in the output directory:
- `run_1.res` - BM25 + RM3 results
- `run_2.res` - Neural reranking results  
- `run_3.res` - RRF fusion results

---

## Command-Line Interface

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--queries PATH` | Path to the query file (e.g., `queriesROBUST.txt`). File should be tab-separated with format: `query_id<TAB>query_text` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--qrels PATH` | None | Path to relevance judgments file for validation. When provided, computes MAP on training queries (first 50). |
| `--output DIR` | `./output` | Directory to save result files. Created automatically if it does not exist. |
| `--method METHOD` | `all` | Which retrieval method(s) to execute. See [Method Selection](#method-selection) below. |
| `--tune` | False | Enable parameter tuning on training queries before test execution. Adds ~20-30 minutes runtime. |
| `--reranker MODEL` | `auto` | Specify the neural reranker model. See [Reranker Models](#reranker-models) below. |
| `--batch-size N` | 32 | Batch size for neural reranking. Reduce if encountering GPU memory errors. |

### Method Selection

The `--method` argument accepts the following values:

| Value | Description | Output File |
|-------|-------------|-------------|
| `all` | Execute all three methods (default) | `run_1.res`, `run_2.res`, `run_3.res` |
| `bm25_rm3` | BM25 with RM3 query expansion only | `run_1.res` |
| `neural` | Neural reranking only | `run_2.res` |
| `rrf` | Reciprocal Rank Fusion only | `run_3.res` |

### Reranker Models

The `--reranker` argument accepts the following values:

| Value | Model | Year | Notes |
|-------|-------|------|-------|
| `auto` | Automatic selection | - | Tries models in order of quality, falls back on failure |
| `qwen3-0.6b-cls` | Qwen3-Reranker-0.6B | 2025 | State-of-the-art, recommended |
| `bge-v2-m3` | BGE-Reranker-v2-M3 | 2024 | Excellent multilingual support |
| `bge-large` | BGE-Reranker-Large | 2023 | Good balance of speed and quality |
| `minilm` | MS-MARCO-MiniLM-L-12 | 2021 | Legacy fallback, fastest |

### Usage Examples

**Run only neural reranking with a specific model:**
```bash
python robust04_ranking_solution.py \
    --queries files/queriesROBUST.txt \
    --method neural \
    --reranker bge-v2-m3 \
    --batch-size 16
```

**Run BM25+RM3 with parameter tuning:**
```bash
python robust04_ranking_solution.py \
    --queries files/queriesROBUST.txt \
    --qrels files/qrels_50_Queries \
    --method bm25_rm3 \
    --tune
```

**Run RRF fusion only (no GPU required):**
```bash
python robust04_ranking_solution.py \
    --queries files/queriesROBUST.txt \
    --method rrf
```

**Full execution with all options:**
```bash
python robust04_ranking_solution.py \
    --queries files/queriesROBUST.txt \
    --qrels files/qrels_50_Queries \
    --output results/ \
    --method all \
    --tune \
    --reranker auto \
    --batch-size 32
```

---

## Methods

### Method 1: BM25 + RM3 Query Expansion

**Classification**: Standard retrieval method (class material)

#### Description

BM25 (Best Matching 25) is a probabilistic ranking function that scores documents based on term frequency, inverse document frequency, and document length normalization. RM3 (Relevance Model 3) extends BM25 with pseudo-relevance feedback.

#### Algorithm

1. Execute initial BM25 retrieval for query Q
2. Assume top-k documents are relevant (pseudo-relevance)
3. Extract discriminative terms from pseudo-relevant documents
4. Expand original query with extracted terms
5. Re-execute BM25 with expanded query

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| k1 | 0.7 | Term frequency saturation parameter |
| b | 0.65 | Document length normalization |
| fb_terms | 70 | Number of expansion terms |
| fb_docs | 10 | Number of feedback documents |
| original_weight | 0.25 | Weight of original query vs. expansion |

#### Tuned Parameters

When `--tune` is enabled, the system performs grid search over:
- k1: [0.7, 0.9, 1.0]
- b: [0.4, 0.5, 0.65]
- fb_terms: [10, 50, 70]
- fb_docs: [5, 10]
- original_weight: [0.25, 0.5]

**Best parameters found (MAP 0.2714 on training set):**
```
k1=0.7, b=0.4, fb_terms=50, fb_docs=5, original_weight=0.5
```

---

### Method 2: Neural Reranking

**Classification**: Advanced method (beyond class material)

#### Description

Two-stage retrieval combining efficient first-stage retrieval with precise neural reranking using the **MaxP (Max Passage) strategy** to handle long documents. A cross-encoder processes query-document pairs jointly, enabling deep semantic understanding that keyword-based methods cannot achieve.

#### MaxP (Max Passage) Strategy

ROBUST04 documents are news articles averaging 500-2000 words, but cross-encoder models typically have a 512-token context limit. MaxP handles this by:

1. Splitting each document into overlapping chunks (1500 chars with 500 char overlap)
2. Scoring each chunk independently with the cross-encoder
3. Taking the **maximum chunk score** as the document's relevance score

This ensures relevant passages within long documents are not truncated and lost.

#### Algorithm

1. **Stage 1 (Retrieval)**: BM25 retrieves top-250 candidate documents
2. **Stage 2 (Chunking)**: Each document is split into overlapping 512-token chunks (max 4 chunks per doc)
3. **Stage 3 (Scoring)**: Cross-encoder scores all query-chunk pairs
4. **Stage 4 (Aggregation)**: MaxP aggregation: document score = max(chunk scores)
5. **Stage 5 (Merge)**: Reranked documents merged with remaining BM25 results for full recall

#### Cross-Encoder Architecture

Unlike bi-encoders that encode queries and documents separately, cross-encoders:
- Process the concatenated [query, document] pair through a transformer
- Enable full attention between query and document tokens
- Produce a single relevance score per pair

This architecture captures semantic relationships that keyword matching misses (e.g., synonyms, paraphrases, implicit relevance).

#### Document Text Extraction

ROBUST04 documents are stored in SGML format (TREC Disks 4 & 5), which requires preprocessing before neural reranking:

```
Raw SGML:
<DOC>
<DOCNO>LA041590-0140</DOCNO>
<DATE><P>April 15, 1990</P></DATE>
<HEADLINE><P>Coyotes in Suburbs</P></HEADLINE>
<TEXT><P>The actual article content...</P></TEXT>
</DOC>
```

The system automatically:
1. Removes null bytes (encoding artifacts causing "s p a c e d" text)
2. Extracts content from `<TEXT>`, `<HEADLINE>`, `<BODY>` tags
3. Strips remaining SGML markup
4. Repairs character spacing corruption
5. Concatenates title and body for the model

This extraction is critical for neural model performance. Without it, models receive XML markup instead of clean text, severely degrading results.

#### Available Models

| Model | Parameters | Context | Performance |
|-------|------------|---------|-------------|
| Qwen3-Reranker-0.6B | 600M | 8192 | State-of-the-art (2025) |
| BGE-Reranker-v2-M3 | 568M | 8192 | Excellent (2024) |
| BGE-Reranker-Large | 335M | 512 | Very good (2023) |
| MS-MARCO-MiniLM-L-12 | 33M | 512 | Adequate (2021) |

#### Memory Considerations

- Models are loaded in FP16 by default
- Batch size of 32 typically works with 8GB VRAM
- OOM errors automatically trigger batch size reduction
- GPU memory is cleared every 50 queries

---

### Method 3: Reciprocal Rank Fusion (Neural + BM25+RM3)

**Classification**: Advanced method (beyond class material)

#### Description

RRF combines the Neural Reranking results with BM25+RM3 results to leverage the strengths of both approaches:
- **BM25+RM3** provides excellent recall (finds more relevant documents)
- **Neural Reranking** provides excellent precision (ranks relevant docs higher at top)

By fusing these complementary methods, we achieve the best overall MAP.

#### Algorithm

For each document d, compute:

```
RRF_score(d) = Σ [weight_r / (k + rank_r(d))]
```

Where:
- k = 30 (tuned via validation)
- rank_r(d) = position of document d in ranker r
- weights = [1.5, 1.0] for [BM25+RM3, Neural] (favors BM25)

#### Why This Works

| Method | Strength | Weakness |
|--------|----------|----------|
| BM25+RM3 | High recall (77% @ 1000) | Lower precision at top |
| Neural | High precision (P@10: 0.50) | Lower recall (71% @ 1000) |
| **RRF Fusion** | **Best of both** | **Minimal tradeoffs** |

#### Advantages

- Combines semantic understanding with keyword matching
- Weighted fusion favors the stronger ranker (BM25)
- No score normalization required
- Achieves highest MAP (**0.3144**)

---

## Output Format

Results are written in standard TREC format:

```
query_id Q0 document_id rank score run_name
```

Example:
```
301 Q0 FBIS3-10082 1 25.432100 run_1
301 Q0 LA041590-0140 2 24.891200 run_1
301 Q0 FT943-11066 3 24.523400 run_1
...
```

Each result file contains up to 1000 documents per query, ordered by decreasing score.

---

## Evaluation Results

Actual performance on 199 test queries (evaluated with full ROBUST04 qrels):

| Run | Method | MAP | MRR | P@10 | Recall@1000 | Runtime |
|-----|--------|-----|-----|------|-------------|---------|
| **3** | **4-Way RRF Fusion** ⭐ | **0.3309** | **0.7714** | **0.5181** | **0.8116** | ~30 min |
| 1 | BM25 + RM3 | 0.3006 | 0.6875 | 0.4683 | 0.7735 | ~12 sec |
| 2 | Neural Reranking | 0.2723 | 0.6740 | 0.4995 | 0.7139 | ~27 min |

### Key Observations
1. **RRF Fusion is the clear winner** (+10% over BM25, +21% over Neural), proving that combining diverse signals (Lexical + Semantic + LLM) outperforms any single method.
2. **Neural Reranking (Run 2)** excels at precision (P@10 ≈ 0.50) but suffers from lower recall (0.71), likely due to the limited candidate pool (top-250 reranked).
3. **BM25+RM3 (Run 1)** is a robust baseline with high recall (0.77) but lower precision at top ranks.
4. **4-Way Fusion** successfully merges:
   - High Recall of BM25
   - High Precision of Neural
   - Knowledge Expansion of Query2Doc
   - Diversity of BM25-plain

---

## The Journey: From Baseline to SOTA (MAP 0.3309)
*A summary of the development process.*

### Phase 1: The "Kitchen Sink" & The MaxP Trap (Fail)
**Goal**: Implement a "perfect" Neural Reranker.
**Action**: We implemented a state-of-the-art Cross-Encoder (BGE-Reranker) with **MaxP Chunking**.
- MaxP splits long documents into overlapping 512-token chunks (e.g., 4 chunks per doc).
- We scored *every* chunk to find the best passage.
**Result**: 
- **Performance**: Good P@10 (0.49), but MAP was average (0.27).
- **Failure**: Runtime was approximately 1 hour 45 minutes for just 199 queries.
- **Lesson**: Computational complexity must be balanced with iterative capability.

### Phase 2: The Efficiency Pivot ("Fast Mode")
**Goal**: Drastically reduce runtime to allow for experimentation.
**Action**: 
- Removed MaxP chunking.
- Implemented **First-512 Truncation** (Title + Lead Paragraph).
- This aligns with the "Inverted Pyramid" structure common in newswire text.
**Result**:
- **Runtime**: Reduced from 105 minutes to 27 minutes (4x improvement).
- **Trade-off**: MAP remained stable (0.27), validating the truncation strategy.

### Phase 3: The Data Augmentation Breakthrough (Query2Doc)
**Goal**: Address the "Vocabulary Mismatch" problem (e.g., Query: "bad weather", Doc: "cyclone").
**Action**: 
- Implemented **LLM Augmentation** using Gemini-Flash.
- For every query, the system generates a pseudo-document containing predicted relevant terms.
- This pseudo-document is concatenated with the query for retrieval.
**Result**:
- Successfully bridges the semantic gap between query terms and document vocabulary.
- **Implementation Note**: Fixed a critical cache key issue by switching from MD5 hashes to Query IDs, enabling reliable expansion retrieval.

### Phase 4: The Fusion "Super-Ensemble" (Success)
**Goal**: Leverage the complementary strengths of individual methods.
**Action**: Analyzed individual method performance:
1. **BM25+RM3**: High Recall (0.77).
2. **Neural**: High Precision (0.50).
3. **Query2Doc**: Contextual understanding.

Implemented a **4-Way Weighted Reciprocal Rank Fusion (RRF)**:
- **Inputs**: (1) BM25+RM3, (2) BM25+Query2Doc, (3) BM25-Plain, (4) Neural.
- **Innovation**: **Query-Dependent Weighting**.
   - Short queries (<3 words): Higher weight to BM25 (keyword match).
   - Long queries (>5 words): Higher weight to Neural (semantic meaning).

### Final Performance
- **Baseline (BM25)**: MAP 0.3006
- **Neural Only**: MAP 0.2723
- **Fusion (Method 3)**: **MAP 0.3309** (+10% relative improvement)
- **MRR**: **0.7714**

---

## Troubleshooting

### Java Not Found

```
Error: Java not found
```

Solution: Install Java 21 and ensure `JAVA_HOME` is set correctly.

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

Solutions:
1. Reduce batch size: `--batch-size 16` or `--batch-size 8`
2. Use a smaller model: `--reranker minilm`
3. Run on CPU (significantly slower): Set `CUDA_VISIBLE_DEVICES=""`

### Model Download Failures

```
OSError: Can't load tokenizer for 'Qwen/Qwen3-Reranker-0.6B'
```

Solutions:
1. Update transformers: `pip install --upgrade transformers>=4.51.0`
2. Use fallback model: `--reranker bge-v2-m3`
3. Check internet connectivity

### Index Download Timeout

```
ConnectionError: Failed to download robust04 index
```

Solution: The index can be downloaded manually from:
```
https://git.uwaterloo.ca/jimmylin/anserini-indexes/raw/master/index-robust04-20191213.tar.gz
```

Extract to `~/.cache/pyserini/indexes/`

### Import Errors

```
ModuleNotFoundError: No module named 'pyserini'
```

Solution: Install all dependencies:
```bash
pip install pyserini torch transformers>=4.51.0 sentence-transformers>=2.7.0 tqdm numpy
```

---

## References

### Papers

1. Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*.

2. Lavrenko, V., & Croft, W. B. (2001). Relevance-Based Language Models. *SIGIR '01*.

3. Nogueira, R., & Cho, K. (2019). Passage Re-ranking with BERT. *arXiv:1901.04085*.

4. Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. *SIGIR '09*.

5. Qwen Team. (2025). Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models.

### Software

- [Pyserini](https://github.com/castorini/pyserini) - Python toolkit for reproducible IR research
- [Sentence Transformers](https://www.sbert.net/) - Cross-encoder implementations
- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) - BGE model implementations

### Dataset

- TREC 2004 Robust Track: 528,155 documents from TREC Disks 4 & 5 (newswire)
- 249 queries with graded relevance judgments

---

## License

This project is developed for academic purposes as part of the Text Retrieval and Search Engines course.

## Author

Text Retrieval Course - Ranking Competition Submission  
January 2026
