"""Query router for directing questions to appropriate handler."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


def normalize_route_label(label: str) -> str:
    """Normalize router output label."""
    label = label.strip().lower()
    if "pandas" in label or "metric" in label:
        return "pandas_metric"
    if "rerank" in label or "insight" in label:
        return "rerank_insight"
    return "rerank_insight"


class QueryRouter:
    """Route questions to metric agent or RAG insight engine."""

    def __init__(
        self,
        llm: "ChatOpenAI",
        metrics_agent,
        rag_engine,
    ):
        self.llm = llm
        self.metrics_agent = metrics_agent
        self.rag_engine = rag_engine

        from langchain.schema import SystemMessage, HumanMessage as _HumanMessage  # noqa: F401
        self._HumanMessage = _HumanMessage
        self.system_prompt = SystemMessage(
            content="You are a routing assistant. You must choose exactly one label: `rerank_insight` or `pandas_metric`."
        )
        self.examples = [
            ("How many negative reviews are there?", "pandas_metric"),
            ("What percentage of positive reviews did we get?", "pandas_metric"),
            ("What are the most common complaints about delivery?", "rerank_insight"),
            ("Summarize the top pain points users mention.", "rerank_insight"),
        ]

    def classify_question(self, question: str) -> str:
        """Classify a question to determine routing."""
        HumanMessage = self._HumanMessage
        messages = [self.system_prompt]

        for q, lbl in self.examples:
            messages.append(HumanMessage(content=f"Q: {q}"))
            messages.append(HumanMessage(content=f"A: {lbl}"))

        messages.append(HumanMessage(content=f"Q: {question}"))
        messages.append(HumanMessage(content="A:"))

        resp = self.llm.invoke(messages)
        text = resp.content if hasattr(resp, "content") else str(resp)
        label = text.strip().lower()

        return normalize_route_label(label)

    def answer(self, question: str) -> str:
        """Route and answer a question."""
        route = self.classify_question(question)

        if route == "pandas_metric":
            return self.metrics_agent.answer(question)
        else:
            return self.rag_engine.answer(question)