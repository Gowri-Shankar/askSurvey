"""RAG pipeline for qualitative insights."""

import warnings
from pathlib import Path
from typing import Union

from typing import TYPE_CHECKING

from prompts import RAG_PROMPT_TEMPLATE

if TYPE_CHECKING:
    from langchain_community.vectorstores import FAISS


warnings.filterwarnings("ignore", category=UserWarning, module="langchain")


def _load_reranker(model_name: str, device: str = "cpu"):
    """Load a cross-encoder reranker, falling back gracefully on Python <3.9.

    ``rerankers`` uses ``list[str]`` annotations (Python 3.9+ syntax) in its
    ColBERT module which causes an import-time TypeError on Python 3.8.  When
    that happens we fall back to ``langchain_community``'s HuggingFaceCrossEncoder
    which exposes a compatible ``.rank()`` interface via a thin wrapper.
    """
    try:
        from rerankers import Reranker
        return Reranker(model_name=model_name, model_type="cross-encoder", device=device)
    except TypeError:
        # Python 3.8: rerankers ColBERT uses 3.9+ annotation syntax at import time.
        return _LangChainCrossEncoderWrapper(model_name, device)


class _RankResult:
    """Minimal rank-result matching the rerankers.results.Result interface."""

    __slots__ = ("doc_id", "score")

    def __init__(self, doc_id: int, score: float):
        self.doc_id = doc_id
        self.score = score


class _LangChainCrossEncoderWrapper:
    """Wraps HuggingFaceCrossEncoder to present the rerankers.Reranker.rank() API."""

    def __init__(self, model_name: str, device: str = "cpu"):
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        self._model = HuggingFaceCrossEncoder(
            model_name=model_name,
            model_kwargs={"device": device},
        )

    def rank(self, query: str, docs: list, doc_ids: list):
        pairs = [[query, doc] for doc in docs]
        scores = self._model.score(pairs)
        results = sorted(
            [_RankResult(doc_id=doc_ids[i], score=float(scores[i])) for i in range(len(docs))],
            key=lambda r: r.score,
            reverse=True,
        )
        return results


def load_faiss_vectorstore(
    index_path: Union[str, Path],
    embedding_model_name: str,
    allow_dangerous_deserialization: bool = True,
    device: str = "cpu",
) -> "FAISS":
    """Load FAISS vector store from disk."""
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device},
    )
    return FAISS.load_local(
        str(index_path),
        embeddings=embedding_model,
        allow_dangerous_deserialization=allow_dangerous_deserialization,
    )


def rerank_documents(
    question: str,
    docs: list,
    reranker: "Reranker",
    top_k: int = 10,
) -> list:
    """Rerank retrieved documents using cross-encoder."""
    rerank_input = [doc.page_content for doc in docs]
    doc_ids = list(range(len(rerank_input)))

    reranked = reranker.rank(query=question, docs=rerank_input, doc_ids=doc_ids)

    # Get top-k reranked documents using their original doc_ids
    top_docs = []
    for result in reranked[:top_k]:
        doc_idx = result.doc_id
        if 0 <= doc_idx < len(docs):
            top_docs.append(docs[doc_idx])

    return top_docs


def build_context(docs: list, max_docs: int = 10) -> str:
    """Build context string from documents."""
    context_docs = docs[:max_docs]
    return "\n\n".join(doc.page_content for doc in context_docs)


class RAGInsightEngine:
    """RAG engine for generating qualitative insights."""

    def __init__(
        self,
        faiss_index_path: Union[str, Path],
        embedding_model_name: str,
        reranker_model: str,
        llm,
        top_k_retrieve: int = 100,
        top_k_context: int = 10,
        embedding_device: str = "cpu",
        reranker_device: str = "cpu",
    ):
        from langchain.prompts import PromptTemplate

        self.vectorstore = load_faiss_vectorstore(
            faiss_index_path,
            embedding_model_name,
            device=embedding_device,
        )
        self.reranker = _load_reranker(reranker_model, device=reranker_device)
        self.llm = llm
        self.top_k_retrieve = top_k_retrieve
        self.top_k_context = top_k_context

        self.prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )

    def answer(self, question: str) -> str:
        """Answer a qualitative insight question."""
        # Stage 1: Retrieve
        retrieved_docs = self.vectorstore.similarity_search(
            question,
            k=self.top_k_retrieve,
        )

        # Stage 2: Rerank
        top_docs = rerank_documents(
            question,
            retrieved_docs,
            self.reranker,
            top_k=self.top_k_context,
        )

        # Build context
        context = build_context(top_docs, max_docs=self.top_k_context)

        # Generate answer
        final_prompt = self.prompt.format(context=context, question=question)
        response = self.llm.invoke(final_prompt)

        return response.content if hasattr(response, "content") else str(response)
