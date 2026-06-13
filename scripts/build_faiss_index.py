"""Build a local FAISS index from a reviews table for offline RAG.

Embeds a text column with a local HuggingFace embedding model and saves the
FAISS index to disk. Runs on CPU by default so it does not compete with the
local LLM for GPU memory.

Usage:
    python scripts/build_faiss_index.py \
        --input results_10.xlsx --review-column text \
        --index-path faiss_results10
"""

import argparse
import sys
from pathlib import Path

# Make src/ importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import DEFAULT_EMBEDDING_MODEL  # noqa: E402
from data_io import load_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS index for offline RAG")
    parser.add_argument("--input", required=True, help="Reviews Excel/CSV file")
    parser.add_argument("--review-column", default="text", help="Text column to embed")
    parser.add_argument("--index-path", required=True, help="Output FAISS index directory")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to embed (0 = all)")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--device", default="cpu", help="Embedding device (cpu/cuda)")
    args = parser.parse_args()

    df = load_table(Path(args.input))
    if args.review_column not in df.columns:
        raise SystemExit(
            f"Column '{args.review_column}' not found. Available: {list(df.columns)}"
        )

    texts = df[args.review_column].dropna().astype(str).tolist()
    if args.limit and args.limit > 0:
        texts = texts[: args.limit]
    if not texts:
        raise SystemExit("No non-empty texts to embed.")

    print(f"Embedding {len(texts)} documents with {args.embedding_model} on {args.device}...")

    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={"device": args.device},
    )
    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore.save_local(args.index_path)
    print(f"Saved FAISS index to '{args.index_path}' ({len(texts)} vectors).")


if __name__ == "__main__":
    main()
