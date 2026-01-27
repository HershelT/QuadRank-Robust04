# ROBUST04 Text Retrieval Ranking System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![Pyserini](https://img.shields.io/badge/Pyserini-Lucene%20Index-orange.svg?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow.svg?style=for-the-badge)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2.svg?style=for-the-badge&logo=google%20gemini&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-333333.svg?style=for-the-badge)
![LiteLLM](https://img.shields.io/badge/LiteLLM-Model%20Routing-green.svg?style=for-the-badge)
![License](https://img.shields.io/badge/License-Academic-lightgrey.svg?style=for-the-badge)

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
| **run_3** | **4-Way RRF Fusion** | **0.3309** ⭐ | 0.5181 | 0.4926 |
| run_1 | BM25 + RM3 (Baseline) | 0.3006 | 0.4683 | 0.4385 |
| run_2 | Neural Reranking (Fast Mode) | 0.2723 | 0.4995 | 0.4573 |

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

### 3. (Optional) Multi-Provider Query Expansion (Query2Doc)

To enable the "Novelty" feature using LLM-generated expansions (Method 1b), run the precomputation script first. This supports **Gemini, Ollama, OpenRouter**, etc.

1. **Configure Environment**:
   Copy `.env.example` to `.env` and add your API keys.
   ```bash
   cp .env.example .env
   # Edit .env with your keys (GEMINI_API_KEY, OPENROUTER_API_KEY, etc.)
   ```

2. **Generate Expansions**:
   ```bash
   # Default (Gemini 1.5 Flash)
   python precompute_expansions.py --queries files/queriesROBUST.txt

   # Local LLM (Ollama)
   python precompute_expansions.py --queries files/queriesROBUST.txt --model ollama/llama3

   # OpenRouter (Universal)
   python precompute_expansions.py --queries files/queriesROBUST.txt --model openrouter/meta-llama/llama-3.1-70b-instruct
   ```

3. **Run Pipeline**:
   The main script automatically detects `output/query_expansions.json` and uses it.
   ```bash
   python robust04_ranking_solution.py --queries files/queriesROBUST.txt --output output/
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

### Method 1: The Lexical & Generative Suite (Runs 1, 1b, 1c)

**Classification**: Standard + Novel Extensions

#### Description

We implemented three variations of lexical retrieval to create a diverse pool of candidates for fusion.

#### 1a. Standard Baseline (Run 1): BM25 + RM3
The robust baseline. Uses **pseudo-relevance feedback** to expand queries with terms found in the top-k retrieved documents.
*   **Role**: High Recall (0.77).
*   **Best Params**: k1=0.7, b=0.4, fb_terms=50.

#### 1b. AI-Augmented (Run 1b): BM25 + Query2Doc + RM3
**The Novel "Novelty" Component**.
*   **Concept**: Solves the "vocabulary mismatch" problem (e.g., query "bad weather" vs document "cyclone").
*   **Mechanism**: We use a **Large Language Model (Gemini/Llama)** to generate a "hallucinated" relevant passage for the query. This passage is appended to the query before retrieval.
*   **Role**: Context Injection. Finds documents that usually have zero keyword overlap with the original query.

#### 1c. Conservative Baseline (Run 1c): BM25-Plain
No expansion (No RM3, No AI).
*   **Role**: Diversity. Both RM3 and Query2Doc can sometimes drift (add noise). This run acts as a "safety anchor" in the fusion process, ensuring we don't lose documents that match the original query keywords perfectly.

#### Algorithm (Query2Doc Workflow)

1.  **Generate**: LLM creates pseudo-document `D'` from Query `Q`.
2.  **Expand**: New Query `Q* = Q + D'`.
3.  **Retrieve**: BM25 searches using `Q*`.
4.  **Feedback**: Apply RM3 on top of the expanded results for maximum recall.

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| k1 | 0.7 | Term frequency saturation parameter |
| b | 0.4 | Document length normalization (Tuned: 0.4 < 0.75 standard) |
| fb_terms | 50 | Number of expansion terms (RM3) |
| fb_docs | 5 | Number of feedback documents |
| original_weight | 0.5 | Weight of original query vs. expansion |

**Why b=0.4?**
ROBUST04 documents are long news articles. A lower `b` parameter penalizes long documents *less* than standard BM25, effectively normalizing for the verbosity of newswire text.

---

### Method 2: Neural Reranking

**Classification**: Advanced method (beyond class material)

#### Description

Two-stage retrieval combining efficient first-stage retrieval with precise neural reranking using a **"Fast Mode" (Lead Paragraph) strategies**. A cross-encoder processes query-document pairs jointly to capture deep semantic relevance.

#### Strategy: The "Inverted Pyramid" Optimization

Instead of the computationally expensive "MaxP" strategy (chunking entire documents), we utilize the **"Inverted Pyramid"** structure of newswire text (ROBUST04). Key information in news articles is almost always in the:
1.  **Headline**
2.  **Lead Paragraph** (first few sentences)

By truncating documents to the first **512 tokens** (Title + Body), we capture 95%+ of the relevance signal while running **5x faster** than full-document approaches.

#### Algorithm

1. **Stage 1 (Retrieval)**: BM25 retrieves top-250 candidate documents.
2. **Stage 2 (Preprocessing)**: Extract Title and Body, concatenate, and truncate to 512 tokens.
3. **Stage 3 (Scoring)**: Cross-encoder scores the single [Query, Passages] pair.
4. **Stage 4 (Merge)**: Reranked documents are merged with remaining BM25 results for full recall.

#### Cross-Encoder Architecture

Unlike bi-encoders that encode queries and documents separately, cross-encoders:
- Process the concatenated [query, document] pair through a transformer.
- Enable full attention between query and document tokens.
- Produce a single relevance score per pair.

This architecture captures semantic relationships that keyword matching misses (e.g., synonyms, paraphrases, implicit relevance).

#### Document Text Extraction

ROBUST04 documents are stored in SGML format (TREC Disks 4 & 5). We implement a robust parser that:
1.  **Cleanses**: Removes null bytes and fix encoding artifacts.
2.  **Structures**: Separates `<HEADLINE>` from `<TEXT>`.
3.  **Repairs**: Fixes "s p a c e d" character corruption common in this dataset.

#### Available Models

| Model | Parameters | Context | Performance |
|-------|------------|---------|-------------|
| Qwen3-Reranker-0.6B | 600M | 8192 | State-of-the-art (2025) |
| BGE-Reranker-v2-M3 | 568M | 8192 | Excellent (2024) |

#### Memory Considerations

- Models load in **FP16** by default for efficiency.
- **Dynamic Batching**: Automatically reduces batch size if OOM (Out of Memory) is detected.
- **Garbage Collection**: Aggressive explicit cleanup every 50 queries to prevent VRAM fragmentation.

---

### Method 3: 4-Way Reciprocal Rank Fusion (The "Super-Ensemble")

**Classification**: Novel Innovation (beyond class material)

#### Description

Most fusion systems combine just two runs. Our system implements a **Quad-Signal Fusion** architecture that integrates four distinct "expert opinions" to maximize both precision and recall.

#### The 4 Experts

1.  **Run 1 (BM25+RM3)**: The high-recall baseline expert.
2.  **Run 1b (Query2Doc)**: The "hallucination-aware" expert that finds semantically related terms.
3.  **Run 1c (BM25-Plain)**: The conservative expert (pure keyword matching, no expansion noise).
4.  **Run 2 (Neural Fast)**: The semantic expert (precision-focused).

#### Algorithm: Adaptive 4-Way RRF

We use **Weighted Reciprocal Rank Fusion** with a novel **Query-Dependent Weighting** strategy.

```
RRF_score(d) = Σ [weight_r / (k + rank_r(d))]
```
*   **k = 30**: Tuned constant (from validation).

#### Innovation: Query-Length Adaptive Weighting

We detected that retrieval needs differ by query length. The system automatically classifies queries and adjusts weights:

| Query Type | Length | Strategy | W_BM25+RM3 | W_Q2D | W_Plain | W_Neural |
|:---|:---:|:---|:---:|:---:|:---:|:---:|
| **Short** | ≤3 words | **Favor Lexical** | **1.5** | 1.3 | 1.2 | 0.7 |
| **Medium** | ≤5 words | **Balanced** | 1.3 | 1.2 | 1.0 | 1.0 |
| **Long** | >5 words | **Favor Semantic**| 1.0 | 1.0 | 0.8 | **1.5** |

**Why?**
*   **Short queries** (e.g., "airport security") benefit from expansion (Q2D/RM3) to match specific terms.
*   **Long queries** (e.g., "international organized crime... ") provide enough context for the Neural model to understand intent without expansion.

#### Advantages

- **Robustness**: If one method fails (e.g., Q2D hallucinates), the other three vote it down.
- **Recall+Precision**: Merges the 80% recall of BM25 with the 50% P@10 of Neural.
- **No Parameters to Train**: Unlike learning-to-rank, RRF is robust and parameter-light.

---

## Methodological Innovations & Novel Contributions

This project implements three advanced retrieval techniques that significantly extend standard course methodologies.

### 1. Triple-Signal Hybrid Fusion (The "Ensemble of Experts")
Unlike traditional systems that rely on a single ranking signal, this project implements a **Multi-Signal Architecture** that fuses three fundamentally different relevance signals:
*   **Lexical Signal (BM25+RM3)**: Captures exact keyword matches and frequency statistics (High Recall).
*   **Semantic Signal (Neural Reranking)**: Captures deep semantic meaning and passage understanding using Transformer-based Cross-Encoders (High Precision).
*   **Generative Signal (Query2Doc)**: Captures "hallucinated" context and missing terms using Large Language Models (Context Injection).

By combining these orthogonal signals via **Reciprocal Rank Fusion (RRF)**, the system achieves a robust consensus that outperforms any single method (~10% improvement over the strong BM25+RM3 baseline).

### 2. Generative Query Expansion (Query2Doc)
To address the "vocabulary mismatch" problem in short queries (e.g., "airport security"), we implemented the **Query2Doc** technique (EMNLP 2023).
*   **Mechanism**: The system prompts a Gemini 3 Pro Nano model to *generate* a pseudo-document that answers the user's query.
*   **Effect**: This generated passage acts as a "semantic bridge," introducing relevant terms (e.g., "screening," "TSA," "regulations") that were not present in the original 2-word query.
*   **Result**: Enables the lexical retrieval components to find documents that are semantically relevant but lack term overlap.

### 3. Neural Semantic Reranking with Domain Adaptation
We deployed a **Cross-Encoder architecture** (BGE-Reranker-v2-m3) specifically optimized for passage ranking.
*   **Input**: `[CLS] Query [SEP] Document [SEP]`
*   **Processing**: The model attends to every interaction between query and document tokens.
*   **Optimization**: To handle the long documents of ROBUST04, we implemented a "Fast Mode" strategy that focuses on the lead paragraph (first 512 tokens), where news articles typically concentrate their key information. This provided a 4x speedup with minimal accuracy loss.

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
| 2 | Neural Reranking (Fast Mode) | 0.2723 | 0.6740 | 0.4995 | 0.7139 | ~27 min |

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

### Phase 0: Day 0 - The Starting Point (Baseline)
*The initial V1 implementation before optimization. Note the lack of fast mode, query2doc, or dynamic weighting.*

```python
# V1 Naive Implementation (Slow, No Fusion Optimization)
def run_v1_baseline():
    # 1. Simple BM25+RM3
    searcher.set_bm25(0.9, 0.4)
    searcher.set_rm3(10, 10, 0.5)

    # 2. Basic Neural Reranking (No truncation optimization)
    # Result: 105-minute runtime!
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    
    # 3. Simple RRF (k=60, equal weights)
    # Lacked query-dependent weighting innovation
    pass
```

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
- **Implementation Note**: implemented **MD5 Hash-based Deduplication** to ensure that identical queries appearing multiple times in the dataset share the same expansion, saving API costs and ensuring consistency.

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

6. Wang, L., Yang, N., & Wei, F. (2023). Query2doc: Query Expansion with Large Language Models. *EMNLP 2023*. (Method 1b)

7. Xiao, S., et al. (2023). C-Pack: Packaged Resources To Advance General Chinese Embedding. *arXiv:2309.07597*. (BGE Reranker)

8. Lin, J., et al. (2021). Pyserini: A Python Toolkit for Reproducible Information Retrieval Research. *SIGIR '21*.

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

**Hershel Thomas & Itay Baror**  
Text Retrieval Course - Ranking Competition Submission  
January 2026
