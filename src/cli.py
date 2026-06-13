"""Command-line interface for askSurvey."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from classification import classify_dataframe
from config import (
    ClassificationConfig,
    DEFAULT_CLASSIFICATION_MODEL,
    DEFAULT_CONFIG_PATH,
    get_openai_api_key,
    load_app_config,
)
from data_io import load_table, merge_topic_lookup, save_table
from model_loader import load_text_generation_pipeline
from preprocessing import clean_text_column


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
    classify_parser.add_argument("--max-new-tokens", type=int, default=64)
    classify_parser.add_argument("--temperature", type=float, default=0.2)
    classify_parser.add_argument("--use-4bit", type=_str2bool, default=True)
    classify_parser.add_argument("--use-8bit", type=_str2bool, default=False)
    classify_parser.add_argument("--allow-cpu-offload", type=_str2bool, default=False)

    query_parser = subparsers.add_parser("query", help="Route a question to metric or insight pipeline")
    query_parser.add_argument("--question", required=True, help="Question to answer")
    query_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    query_parser.add_argument("--llm-provider", choices=["openai", "local"], help="Override LLM provider")
    query_parser.add_argument("--reviews", help="Reviews Excel/CSV for the metric agent")
    query_parser.add_argument("--reviews-csv", dest="reviews", help="Alias for --reviews")
    query_parser.add_argument("--review-column", help="Review text column (RAG/index)")
    query_parser.add_argument("--faiss-index", help="FAISS index directory (optional)")
    query_parser.add_argument("--embedding-model", help="Embedding model name")
    query_parser.add_argument("--reranker-model", help="Reranker model name")
    query_parser.add_argument("--top-k-retrieve", type=int, help="Docs to retrieve before rerank")
    query_parser.add_argument("--top-k-context", type=int, help="Docs to keep after rerank")

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
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        allow_cpu_offload=args.allow_cpu_offload,
    )

    df = load_table(cfg.input_path)
    df = clean_text_column(df, cfg.review_column)

    pipe = load_text_generation_pipeline(
        cfg.model_name,
        use_4bit=cfg.use_4bit,
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


class _UnavailableRAG:
    """Placeholder used when no FAISS index / RAG deps are available."""

    def __init__(self, reason: str):
        self.reason = reason

    def answer(self, question: str) -> str:
        return self.reason


def run_query(args: argparse.Namespace) -> None:
    from llm_factory import build_llm_provider
    from metrics_agent import MetricsAgent
    from router import QueryRouter

    # dataclass defaults -> config.yaml -> CLI overrides
    app_cfg = load_app_config(args.config)
    llm_cfg = app_cfg.llm
    q_cfg = app_cfg.query

    if args.llm_provider:
        llm_cfg.provider = args.llm_provider
    if args.reviews:
        q_cfg.reviews_path = Path(args.reviews)
    if args.review_column:
        q_cfg.review_column = args.review_column
    if args.faiss_index:
        q_cfg.faiss_index = Path(args.faiss_index)
    if args.embedding_model:
        q_cfg.embedding_model = args.embedding_model
    if args.reranker_model:
        q_cfg.reranker_model = args.reranker_model
    if args.top_k_retrieve is not None:
        q_cfg.top_k_retrieve = args.top_k_retrieve
    if args.top_k_context is not None:
        q_cfg.top_k_context = args.top_k_context

    openai_api_key = get_openai_api_key() if llm_cfg.provider == "openai" else None
    provider = build_llm_provider(llm_cfg, openai_api_key)

    reviews_df = load_table(q_cfg.reviews_path)
    metrics_agent = MetricsAgent(
        dataframe=reviews_df,
        llm=provider.agent_llm,
        agent_type=provider.agent_type,
        chat_llm=provider.chat_llm,
    )

    # RAG is optional: only build it when an index exists and deps import.
    if q_cfg.faiss_index and Path(q_cfg.faiss_index).exists():
        try:
            from rag import RAGInsightEngine
            rag_engine = RAGInsightEngine(
                faiss_index_path=q_cfg.faiss_index,
                embedding_model_name=q_cfg.embedding_model,
                reranker_model=q_cfg.reranker_model,
                llm=provider.chat_llm,
                top_k_retrieve=q_cfg.top_k_retrieve,
                top_k_context=q_cfg.top_k_context,
                embedding_device=q_cfg.embedding_device,
                reranker_device=q_cfg.reranker_device,
            )
        except Exception as exc:  # noqa: BLE001
            rag_engine = _UnavailableRAG(
                f"RAG unavailable ({exc}). Install RAG deps and rebuild the index."
            )
    else:
        rag_engine = _UnavailableRAG(
            "RAG unavailable: no FAISS index found at "
            f"'{q_cfg.faiss_index}'. Build one with scripts/build_faiss_index.py."
        )

    router = QueryRouter(provider.chat_llm, metrics_agent, rag_engine)
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
