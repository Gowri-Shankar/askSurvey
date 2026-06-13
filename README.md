# askSurvey

Python CLI and package for e-commerce review analysis — topic classification, sentiment analysis, and retrieval-based qualitative insights.

## Features

| Capability | Model / Tool |
|---|---|
| Topic + sentiment classification | `microsoft/Phi-3-mini-4k-instruct` (4-bit, local GPU) |
| Quantitative metric queries | LangChain pandas agent + GPT |
| Qualitative RAG insights | FAISS + cross-encoder reranker + GPT |

---

## Project Layout

```
src/                 Core package modules
  classification.py  Topic/sentiment parsing and batch classification
  config.py          Model and pipeline configuration
  data_io.py         Excel/CSV load and save utilities
  model_loader.py    HuggingFace pipeline loader (4-bit / 8-bit / fp16)
  preprocessing.py   Text cleaning utilities
  prompts.py         Prompt templates and topic/sentiment label lists
  metrics_agent.py   Pandas-based metric agent (OpenAI)
  rag.py             FAISS retrieval + reranking + GPT synthesis
  router.py          Question router (metric vs. insight)
  cli.py             CLI entry point
scripts/             Convenience run scripts
tests/               Unit tests (19 tests, no GPU or API key required)
notebooks/           Reference notebook
legacy/              Legacy script (pre-migration)
```

---

## Setup

### Requirements

- Python 3.8+
- NVIDIA GPU with 4 GB+ VRAM (GTX 1650 or better) for local classification
- CUDA driver 11.8+ (tested on driver 581 / CUDA 13.0)
- OpenAI API key (only for query/RAG mode)

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
# or
venv\Scripts\activate          # Windows CMD / PowerShell
```

### 2. Install PyTorch with CUDA support

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> For CUDA 11.8 use `cu118` instead of `cu121`.

### 3. Install bitsandbytes (4-bit quantization)

```bash
pip install bitsandbytes --prefer-binary \
  --extra-index-url https://huggingface.github.io/bitsandbytes-windows-webui
```

### 4. Install the package and remaining dependencies

```bash
pip install -e .
pip install transformers accelerate openpyxl tqdm
```

### 5. Set your OpenAI API key (query mode only)

```bash
export OPENAI_API_KEY=your_key_here        # Git Bash / Linux / Mac
$env:OPENAI_API_KEY="your_key_here"        # PowerShell
```

---

## Usage

### Classify reviews (topic + sentiment)

```bash
# Full dataset
ask-survey classify \
  --input reviews.xlsx \
  --output results.xlsx \
  --review-column text

# With topic lookup table
ask-survey classify \
  --input reviews.xlsx \
  --output results.xlsx \
  --review-column text \
  --topics top_subtopics.xlsx

# Optional flags
  --model-name microsoft/Phi-3-mini-4k-instruct   # default
  --use-4bit true                                  # default, ~1.5 GB VRAM
  --use-8bit false                                 # ~2.5 GB VRAM
  --max-new-tokens 64                              # default
  --temperature 0.2                                # default
```

Output columns added to your file:

| Column | Description |
|---|---|
| `model_pred_sub_topic` | Predicted topic (one of 34 labels) |
| `model_pred_sentiment` | `Positive`, `Neutral`, or `Negative` |
| `model_raw_response` | Raw model output for debugging |
| `model_error` | Error message if classification failed |

### Quick sample test (5 reviews, no data file needed)

```bash
python scripts/test_sample.py
```

### Query mode (requires OpenAI API key + FAISS index)

```bash
ask-survey query \
  --question "What are common delivery complaints?" \
  --reviews-csv reviews.csv \
  --faiss-index faiss_ecom_25k
```

---

## Running Tests

No GPU or API key required — all 19 tests run on pure logic.

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Model Details

**Classification model:** `microsoft/Phi-3-mini-4k-instruct`
- 3.8B parameters, 4-bit NF4 quantization
- VRAM usage: ~1.5 GB (fits GTX 1650 with 4 GB)
- Speed: ~1–2 seconds per review on GTX 1650
- Both topic and sentiment are predicted in a single inference call

**34 topic labels** span: product quality, pricing, delivery, customer service, returns, and platform experience.

**Sentiment labels:** `Positive`, `Neutral`, `Negative`

To switch models, change `DEFAULT_CLASSIFICATION_MODEL` in `src/config.py`:

```python
# Smaller / faster (1.5 GB download, ~1 GB VRAM)
DEFAULT_CLASSIFICATION_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Better quality (2.4 GB download, ~1.5 GB VRAM) — current default
DEFAULT_CLASSIFICATION_MODEL = "microsoft/Phi-3-mini-4k-instruct"
```

---

## Notes

- Do not hard-code API keys — use environment variables or a `.env` file.
- The `flash-attention` warnings at startup are cosmetic and do not affect output.
- For the full 33K review dataset, expect 8–18 hours on a GTX 1650.
- The notebook in `notebooks/` and script in `legacy/` are reference only.

## Roadmap

- Streamlit UI on top of the CLI modules
- Batch inference for faster GPU throughput
- Support for custom topic label lists via config
