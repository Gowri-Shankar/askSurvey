# askSurvey

askSurvey is a Python package and CLI for review/survey analysis.

It supports:
- Topic and sentiment classification
- Metric questions over tabular review data
- Retrieval-based qualitative insights with reranking

## Project Layout

```text
src/              Core package modules
scripts/          Convenience entry points
notebooks/        Legacy notebook reference
legacy/           Temporary legacy script during migration
tests/            Unit tests
```

## Install

```bash
pip install -e .
```

## Environment

Create a `.env` or set:

```bash
OPENAI_API_KEY=your_key_here
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_key_here"
```

## Classification

```bash
ask-survey classify --input reviews.xlsx --output results.xlsx --topics top_subtopics.xlsx --review-column Reviews
```

## Query Mode

```bash
ask-survey query --question "What are common delivery complaints?" --reviews-csv reviews.csv --faiss-index faiss_ecom_25k
```

## Notes

- The notebook remains in `notebooks/` as a reference.
- The old script remains in `legacy/` until final cleanup.
- Do not hard-code API keys in code.

## Next Step

After the CLI/package is stable, add a Streamlit UI on top of these modules.
