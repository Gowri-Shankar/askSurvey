"""Configuration management for askSurvey."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_CLASSIFICATION_MODEL = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANKER_MODEL = "mixedbread-ai/mxbai-rerank-large-v1"
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"


def get_openai_api_key() -> str:
    """Get OpenAI API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it in your environment or .env file."
        )
    return api_key


@dataclass
class ClassificationConfig:
    """Configuration for topic/sentiment classification."""
    model_name: str = DEFAULT_CLASSIFICATION_MODEL
    input_path: Path = Path("review_data.xlsx")
    topics_path: Optional[Path] = None
    output_path: Path = Path("classification_results.xlsx")
    review_column: str = "Reviews"
    topic_lookup_column: str = "Sub-Topic"
    batch_size: int = 1
    max_new_tokens: int = 64
    temperature: float = 0.2
    use_4bit: bool = True
    use_8bit: bool = False
    allow_cpu_offload: bool = False


@dataclass
class QueryConfig:
    """Configuration for RAG/query pipeline."""
    reviews_csv: Path = Path("ecommerce_reviews_25k.csv")
    faiss_index: Path = Path("faiss_ecom_25k")
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str = DEFAULT_RERANKER_MODEL
    openai_model: str = DEFAULT_OPENAI_MODEL
    top_k_retrieve: int = 100
    top_k_context: int = 10
    openai_temperature: float = 0.2
    openai_max_tokens: int = 256