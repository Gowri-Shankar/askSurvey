"""Central factory that builds the LLM backend for the query pipeline.

Supports two providers:

* ``openai`` — ``ChatOpenAI`` for both the chat consumers (router, RAG) and the
  pandas metrics agent, using OpenAI function-calling.
* ``local`` — the already-downloaded HuggingFace model (e.g. Phi-3-mini), loaded
  once via :func:`model_loader.load_text_generation_pipeline` and shared as a
  ``HuggingFacePipeline`` (for the ReAct pandas agent) and a ``ChatHuggingFace``
  (for the router and RAG synthesis). Runs fully offline.
"""

from dataclasses import dataclass
from typing import Optional

from config import LLMConfig


class _ChatMessage:
    """Minimal response object with a .content attribute, matching ChatOpenAI output."""

    def __init__(self, content: str):
        self.content = content


class _LocalChatAdapter:
    """Thin wrapper around a HF text-generation pipeline for router + RAG synthesis.

    Converts LangChain message lists to a prompt string using the tokenizer's
    chat template (the same technique used in classification._build_prompt), then
    calls the pipeline directly.  Returns a ``_ChatMessage`` so consumers work
    identically regardless of backend.
    """

    def __init__(self, pipe, max_new_tokens: int, temperature: float):
        self._pipe = pipe
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

    def _messages_to_prompt(self, messages: list) -> str:
        tokenizer = self._pipe.tokenizer
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            chat = []
            for m in messages:
                role = getattr(m, "type", None) or "user"
                role = "system" if role == "system" else "user"
                chat.append({"role": role, "content": m.content})
            return tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        return "\n".join(getattr(m, "content", str(m)) for m in messages)

    # Phi-3 continues generating after the real answer in several patterns.
    # These markers all indicate a new example/turn is starting — everything
    # after them is hallucinated content that should be discarded.
    _STOP_MARKERS = [
        "\n### Instruction 2",
        "\n### Instruction",
        "\n**Instruction:**",
        "\n<|user|>",
        "\n<|end|>",
        "\nQuestion:",          # model starts generating a new Q&A pair
        "\n\nQuestion:",
        "\n---\n",              # section divider often precedes a new example
    ]

    def _trim_at_stop_marker(self, text: str) -> str:
        for marker in self._STOP_MARKERS:
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx]
        return text.strip()

    def invoke(self, input_) -> _ChatMessage:
        if isinstance(input_, list):
            prompt = self._messages_to_prompt(input_)
        else:
            prompt = str(input_)
        result = self._pipe(
            prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
            do_sample=self._temperature > 0,
            return_full_text=False,
        )
        text = result[0]["generated_text"].strip()
        text = self._trim_at_stop_marker(text)
        return _ChatMessage(content=text)


@dataclass
class LLMProvider:
    """Resolved LLM objects for the three query-pipeline consumers."""
    chat_llm: object   # .invoke(messages|str) -> object with .content (router + RAG)
    agent_llm: object  # LangChain LLM passed to create_pandas_dataframe_agent
    agent_type: object  # langchain AgentType.*


def _build_openai_provider(cfg: LLMConfig, openai_api_key: Optional[str]) -> LLMProvider:
    from langchain.agents.agent_types import AgentType
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=cfg.openai_model,
        temperature=cfg.temperature,
        api_key=openai_api_key,
    )
    return LLMProvider(
        chat_llm=llm,
        agent_llm=llm,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )


def _build_local_provider(cfg: LLMConfig) -> LLMProvider:
    from langchain.agents.agent_types import AgentType

    try:
        from langchain_huggingface import HuggingFacePipeline
    except ImportError:  # fall back to the (deprecated) community location
        from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

    from model_loader import load_text_generation_pipeline

    pipe = load_text_generation_pipeline(
        cfg.local_model,
        use_4bit=cfg.use_4bit,
        use_8bit=cfg.use_8bit,
        allow_cpu_offload=cfg.allow_cpu_offload,
    )

    # agent_llm: HuggingFacePipeline is the standard LangChain text-LLM interface used
    # by create_pandas_dataframe_agent (ReAct loop, no function-calling required).
    hf_llm = HuggingFacePipeline(
        pipeline=pipe,
        pipeline_kwargs={
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "do_sample": cfg.temperature > 0,
            "return_full_text": False,
        },
    )

    # chat_llm: thin adapter that applies the Phi-3 chat template and returns a
    # .content response — avoids ChatHuggingFace's Hub endpoint lookup (v0.0.3 bug).
    chat_llm = _LocalChatAdapter(pipe, cfg.max_new_tokens, cfg.temperature)

    return LLMProvider(
        chat_llm=chat_llm,
        agent_llm=hf_llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )


def build_llm_provider(cfg: LLMConfig, openai_api_key: Optional[str] = None) -> LLMProvider:
    """Build the LLM provider for the configured backend."""
    provider = (cfg.provider or "openai").strip().lower()
    if provider == "openai":
        return _build_openai_provider(cfg, openai_api_key)
    if provider == "local":
        return _build_local_provider(cfg)
    raise ValueError(
        f"Unknown llm provider '{cfg.provider}'. Use 'openai' or 'local'."
    )
