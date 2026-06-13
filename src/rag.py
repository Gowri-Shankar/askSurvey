"""RAG pipeline for qualitative insights."""

import warnings
from pathlib import Path
from typing import Union

from typing import TYPE_CHECKING

from prompts import RAG_PROMPT_TEMPLATE

if TYPE_CHECKING:
    from langchain_community.vectorstores import FAISS
    from rerankers import Reranker


warnings.filterwarnings("ignore", category=UserWarning, module="langchain")


def load_faiss_vectorstore(
    index_path: Union[str, Path],
    embedding_model_name: str,
    allow_dangerous_deserialization: bool = True,
) -> "FAISS":
    """Load FAISS vector store from disk."""
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
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
        openai_model: str,
        openai_api_key: str,
        top_k_retrieve: int = 100,
        top_k_context: int = 10,
        temperature: float = 0.2,
        max_tokens: int = 256,
    ):
        from langchain.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI
        from rerankers import Reranker

        self.vectorstore = load_faiss_vectorstore(
            faiss_index_path,
            embedding_model_name,
        )
        self.reranker = Reranker(model_name=reranker_model, model_type="cross-encoder")
        self.llm = ChatOpenAI(
            model=openai_model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=openai_api_key,
        )
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

        return response.content
