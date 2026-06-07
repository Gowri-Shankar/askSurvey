"""Command-line interface for askSurvey."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from classification import classify_dataframe
from config import (
    ClassificationConfig,
    DEFAULT_CLASSIFICATION_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_RERANKER_MODEL,
    QueryConfig,
    get_openai_api_key,
)
from data_io import load_table, merge_topic_lookup, save_table
from metrics_agent import MetricsAgent
from model_loader import load_text_generation_pipeline
from preprocessing import clean_text_column
from rag import RAGInsightEngine
from router import QueryRouter
from langchain_openai import ChatOpenAI


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ask-survey", description="askSurvey CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    classify_parser = subparsers.add_parser("classify", help="Classify reviews by topic and sentiment")
    classify_parser.add_argument("--input", required=True, help="Input Excel/CSV file")
    classify_parser.add_argument("--output", required=True, help="Output Excel/CSV file")
    classify_parser.add_argument("--topics", help="Topic lookup Excel/CSV file")
    classify_parser.add_argument("--review-column", default="Reviews", help="Review text column name")
    classify_parser.add_argument("--topic-lookup-column", default="Sub-Topic", help="Topic lookup column name")
    classify_parser.add_argument("--model-name", default=DEFAULT_CLASSIFICATION_MODEL, help="HF model name")
    classify_parser.add_argument("--max-new-tokens", type=int, default=246)
    classify_parser.add_argument("--temperature", type=float, default=0.2)
    classify_parser.add_argument("--use-8bit", type=_str2bool, default=True)
    classify_parser.add_argument("--allow-cpu-offload", type=_str2bool, default=False)

    query_parser = subparsers.add_parser("query", help="Route a question to metric or insight pipeline")
    query_parser.add_argument("--question", required=True, help="Question to answer")
    query_parser.add_argument("--reviews-csv", required=True, help="Reviews CSV for metric agent")
    query_parser.add_argument("--faiss-index", required=True, help="FAISS index directory")
    query_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model name")
    query_parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL, help="Reranker model name")
    query_parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL, help="OpenAI chat model")
    query_parser.add_argument("--top-k-retrieve", type=int, default=100)
    query_parser.add_argument("--top-k-context", type=int, default=10)
    query_parser.add_argument("--openai-temperature", type=float, default=0.2)
    query_parser.add_argument("--openai-max-tokens", type=int, default=256)

    return parser


def run_classify(args: argparse.Namespace) -> None:
    cfg = ClassificationConfig(
        model_name=args.model_name,
        input_path=Path(args.input),
        topics_path=Path(args.topics) if args.topics else None,
        output_path=Path(args.output),
        review_column=args.review_column,
        topic_lookup_column=args.topic_lookup_column,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_8bit=args.use_8bit,
        allow_cpu_offload=args.allow_cpu_offload,
    )

    df = load_table(cfg.input_path)
    df = clean_text_column(df, cfg.review_column)

    pipe = load_text_generation_pipeline(
        cfg.model_name,
        use_8bit=cfg.use_8bit,
        allow_cpu_offload=cfg.allow_cpu_offload,
    )

    classified_df = classify_dataframe(
        df,
        pipe,
        review_column=cfg.review_column,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
    )

    if cfg.topics_path:
        topics_df = load_table(cfg.topics_path)
        classified_df = merge_topic_lookup(
            classified_df,
            topics_df,
            predicted_topic_column="model_pred_sub_topic",
            lookup_column=cfg.topic_lookup_column,
        )

    save_table(classified_df, cfg.output_path)
    print(f"Saved results to {cfg.output_path}")


def run_query(args: argparse.Namespace) -> None:
    openai_api_key = get_openai_api_key()

    query_cfg = QueryConfig(
        reviews_csv=Path(args.reviews_csv),
        faiss_index=Path(args.faiss_index),
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        openai_model=args.openai_model,
        top_k_retrieve=args.top_k_retrieve,
        top_k_context=args.top_k_context,
        openai_temperature=args.openai_temperature,
        openai_max_tokens=args.openai_max_tokens,
    )

    reviews_df = pd.read_csv(query_cfg.reviews_csv)
    metrics_agent = MetricsAgent(
        dataframe=reviews_df,
        openai_api_key=openai_api_key,
        model_name=query_cfg.openai_model,
        temperature=0.0,
    )

    rag_engine = RAGInsightEngine(
        faiss_index_path=query_cfg.faiss_index,
        embedding_model_name=query_cfg.embedding_model,
        reranker_model=query_cfg.reranker_model,
        openai_model=query_cfg.openai_model,
        openai_api_key=openai_api_key,
        top_k_retrieve=query_cfg.top_k_retrieve,
        top_k_context=query_cfg.top_k_context,
        temperature=query_cfg.openai_temperature,
        max_tokens=query_cfg.openai_max_tokens,
    )

    router_llm = ChatOpenAI(
        model=query_cfg.openai_model,
        temperature=0.0,
        api_key=openai_api_key,
    )

    router = QueryRouter(router_llm, metrics_agent, rag_engine)
    answer = router.answer(args.question)
    print(answer)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "classify":
        run_classify(args)
    elif args.command == "query":
        run_query(args)


if __name__ == "__main__":
    main()
