"""Configuration management for askSurvey."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DEFAULT_CLASSIFICATION_MODEL = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANKER_MODEL = "mixedbread-ai/mxbai-rerank-large-v1"
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"

# Offline / local-provider defaults
DEFAULT_LLM_PROVIDER = "openai"  # "openai" | "local"
DEFAULT_LOCAL_LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"
# Light cross-encoder that fits a 4 GB GPU alongside Phi-3 (recommended over the
# ~1.2 GB mxbai-rerank-large default when running fully offline on small hardware).
DEFAULT_LOCAL_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DEFAULT_CONFIG_PATH = "config.yaml"


def get_openai_api_key() -> str:
    """Get OpenAI API key, loading .env if present."""
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)  # won't overwrite already-set env vars
    except ImportError:
        pass
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Set it in your environment or a .env file."
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
class LLMConfig:
    """Which LLM backend the query pipeline uses (router, metrics, RAG synthesis)."""
    provider: str = DEFAULT_LLM_PROVIDER  # "openai" | "local"
    openai_model: str = DEFAULT_OPENAI_MODEL
    local_model: str = DEFAULT_LOCAL_LLM_MODEL
    temperature: float = 0.2
    max_new_tokens: int = 256
    use_4bit: bool = True
    use_8bit: bool = False
    allow_cpu_offload: bool = False


@dataclass
class QueryConfig:
    """Configuration for RAG/query pipeline."""
    reviews_path: Path = Path("results_10.xlsx")
    review_column: str = "text"
    faiss_index: Path = Path("faiss_ecom_25k")
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str = DEFAULT_RERANKER_MODEL
    embedding_device: str = "cpu"
    reranker_device: str = "cpu"
    top_k_retrieve: int = 100
    top_k_context: int = 10
    openai_temperature: float = 0.2
    openai_max_tokens: int = 256


@dataclass
class AppConfig:
    """Top-level app configuration loaded from config.yaml."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    query: QueryConfig = field(default_factory=QueryConfig)


def _apply_section(instance, values: dict) -> None:
    """Set only known dataclass fields from a dict, coercing Path-typed fields."""
    if not values:
        return
    valid = instance.__dataclass_fields__
    for key, value in values.items():
        if key not in valid:
            continue
        if value is None:
            continue
        # Coerce string paths to Path for Path-typed fields.
        annotation = valid[key].type
        if annotation in (Path, "Path") and not isinstance(value, Path):
            value = Path(value)
        setattr(instance, key, value)


def load_app_config(path=DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load AppConfig from a YAML file, falling back to dataclass defaults.

    Layering is: dataclass default -> config.yaml. CLI overrides are applied on
    top of the returned object by the caller (cli.py).
    """
    config = AppConfig()

    path = Path(path)
    if not path.exists():
        return config

    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    _apply_section(config.llm, data.get("llm", {}) or {})
    _apply_section(config.query, data.get("query", {}) or {})
    return config
