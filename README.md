# Survey Review Analyzer — Topic Modeling & Sentiment Analysis

This project performs large-scale topic classification and sentiment analysis on general survey or review text using a small/quantized LLM pipeline and standard data tooling. The primary implementation is a Jupyter notebook: `survey_topic_modeling_sentiment.ipynb` which contains end-to-end code for loading data, batching inference, and saving results.



# E‑commerce Reviews — Topic Modeling & Sentiment Analysis

This project performs large-scale topic classification and sentiment analysis on  reviews using a small/quantized LLM pipeline and standard data tooling. The primary implementation is a Jupyter notebook: `survey_topic_modeling_sentiment.ipynb` which contains end-to-end code for loading data, batching inference, and saving results.

## Key Features

- Batch inference using Hugging Face `pipeline` with optional GPU acceleration and 8-bit quantization (via `bitsandbytes`).
- Topic classification and sentiment extraction using a single-shot prompt-based text-generation pipeline.
- Post-processing and merging of predictions into Excel output for downstream analysis.
- Optional two-stage Retrieval-Augmented Generation (RAG) architecture with FAISS + cross-encoder reranking for qualitative insights.

## Quick Start

1. Clone or copy the notebook into your workspace: the main notebook is `survey_topic_modeling_sentiment.ipynb`.
2. Install common dependencies (use a venv or conda env):

```bash
pip install -r requirements.txt
# or minimal set:
pip install pandas openpyxl transformers torch bitsandbytes faiss-cpu tqdm
```

3. Open the notebook and update any file paths (Excel input and output). The notebook expects two primary input files:

- `review_data.xlsx` — the reviews or survey responses to classify (one record per row).
- `top_subtopics.xlsx` — a lookup file containing the canonical sub-topic labels used for classification and mapping.

4. Run cells sequentially. If using Colab, the notebook mounts Google Drive to save results.

## Technical Project Overview: Hybrid RAG with Reranking

1. Architecture Overview

This project implements a two-stage Retrieval-Augmented Generation (RAG) pipeline designed to handle e-commerce reviews. The system routes queries between a Pandas-based analytical agent (for quantitative metrics) and a Semantic Search pipeline (for qualitative insights).

2. The Retrieval Strategy (Two-Stage)

We chose a two-stage approach to balance speed and accuracy:

- Stage 1: Bi-Encoder Retrieval (FAISS + BGE-Small)
  - Use `BAAI/bge-small-en-v1.5` to create embeddings stored in a FAISS index for high-speed similarity search across large review sets. This narrows results from thousands to the top ~100 candidates.

- Stage 2: Cross-Encoder Reranking (`mxbai-rerank-large-v1`)
  - Why: Bi-encoders are fast but evaluate query and document independently and can miss fine-grained relevance.
  - Purpose: The reranker processes query+document jointly to produce a better-ordered shortlist for the LLM context window.

3. Routing Mechanism

To avoid hallucinated statistics, a Router sends analytical queries (e.g., "% of negative reviews") to a `create_pandas_dataframe_agent` for exact computations, and insight-style queries (e.g., "What are common pain points?") to the reranked RAG chain for thematic analysis.

4. Technical Stack Highlights

- Vector DB: FAISS for local, low-latency vector search.
- Reranker: `mixedbread-ai/mxbai-rerank-large-v1` for cross-encoder reranking.
- LLM: GPT-3.5-Turbo (used as the final synthesizer in the pipeline). Language-agnostic orchestration via LangChain.

5. Why LangChain

- Modularity & Abstraction: swap components (embeddings, vector stores, LLMs) easily.
- Chain & Agent Synergy: use RetrievalQA and `create_pandas_dataframe_agent` for deterministic analytics and retrieval workflows.
- Ecosystem Integration: built-in support for FAISS and rerankers simplifies the two-stage retrieval flow.

6. Why FAISS

- In-memory speed and optimized CPU/GPU computations for dense vectors — suitable for datasets ~25k rows.
- Designed for scale (works for much larger datasets) and supports local persistence.

## Notebook Notes & Tips

- The notebook demonstrates model quantization and loading with `BitsAndBytesConfig(load_in_8bit=True)` to reduce GPU memory usage.
- Use batching when running `pipeline` over thousands of reviews to avoid OOM and to increase throughput.
- If you plan to run the RAG pipeline, persist FAISS indices to disk (or Drive) to avoid recomputing embeddings.