"""Metrics agent for quantitative queries using pandas."""

import sys
from pathlib import Path
from typing import Optional, Union

import pandas as pd

# PythonAstREPLTool (used by create_pandas_dataframe_agent) requires Python >=3.9.
_PANDAS_AGENT_SUPPORTED = sys.version_info >= (3, 9)

# AgentType was removed in LangChain 0.2+; fall back to the underlying string values.
try:
    from langchain.agents import AgentType as _AgentType  # removed in LangChain 0.2+
    _AGENT_OPENAI_FUNCTIONS = _AgentType.OPENAI_FUNCTIONS
except (ImportError, AttributeError):
    _AGENT_OPENAI_FUNCTIONS = "openai-functions"


def _df_schema(df: pd.DataFrame, max_cats: int = 8) -> str:
    """Build a concise schema string for prompting the LLM."""
    lines = ["Columns (name: dtype, [unique values if few]):"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        nuniq = df[col].nunique()
        if nuniq <= max_cats:
            vals = str(df[col].dropna().unique().tolist())
            lines.append(f"  {col}: {dtype}, values={vals}")
        else:
            lines.append(f"  {col}: {dtype}")
    lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} cols")
    return "\n".join(lines)


class MetricsAgent:
    """Agent for answering quantitative metric questions over a DataFrame.

    The LLM and agent type are injected (see ``llm_factory.build_llm_provider``).

    On Python >=3.9 with an OpenAI-compatible LLM the standard LangChain
    ``create_pandas_dataframe_agent`` is used. On Python 3.8 (or when the
    pandas agent is unavailable) a fallback path prompts the LLM to emit a
    single pandas expression and evaluates it directly.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        llm,
        agent_type: Optional[object] = None,
        chat_llm=None,
    ):
        """
        Parameters
        ----------
        llm:
            LangChain LLM for the ReAct pandas agent (Python >=3.9).
        agent_type:
            AgentType.* variant; defaults to OPENAI_FUNCTIONS.
        chat_llm:
            LLM used for expression-eval prompting on Python <3.9 (e.g.
            ``_LocalChatAdapter``). Falls back to ``llm`` if not provided.
        """
        if agent_type is None:
            agent_type = _AGENT_OPENAI_FUNCTIONS

        self.dataframe = dataframe
        self.llm = llm
        # For expression-eval prompting use chat_llm when given (handles Phi-3 template).
        self._expr_llm = chat_llm if chat_llm is not None else llm
        self._agent = None
        self._use_expr_eval = False

        if _PANDAS_AGENT_SUPPORTED:
            try:
                from langchain_experimental.agents import create_pandas_dataframe_agent
                self._agent = create_pandas_dataframe_agent(
                    self.llm,
                    self.dataframe,
                    verbose=False,
                    agent_type=agent_type,
                    allow_dangerous_code=True,
                )
            except Exception:
                self._use_expr_eval = True
        else:
            # Python <3.9: PythonAstREPLTool not available; use expression-eval mode.
            self._use_expr_eval = True

    def _answer_expr(self, question: str) -> str:
        """Fallback: ask the LLM for a pandas expression, then eval it."""
        schema = _df_schema(self.dataframe)
        prompt = (
            f"You are a data analyst. A pandas DataFrame named `df` has this schema:\n"
            f"{schema}\n\n"
            f"Write a single Python expression (no imports, no assignments) using `df` "
            f"that answers the question below. Output ONLY the expression on one line, "
            f"no explanation, no markdown, just valid Python.\n\n"
            f"Question: {question}\n\nExpression:"
        )
        resp = self._expr_llm.invoke(prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)

        # Keep only the first non-empty line; strip markdown fences.
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        expr = lines[0] if lines else raw.strip()
        expr = expr.strip("`").strip()
        for fence in ("python", "py"):
            if expr.lower().startswith(fence):
                expr = expr[len(fence):].strip()

        # Drop any non-ASCII chars the model may have hallucinated.
        expr = expr.encode("ascii", errors="ignore").decode("ascii")

        try:
            result = eval(expr, {"df": self.dataframe, "pd": pd})  # noqa: S307
            return str(result)
        except Exception as exc:
            # Surface the raw LLM response so the user can understand what happened.
            safe_raw = raw.encode("ascii", errors="replace").decode("ascii")
            return f"(expression eval failed — model output: {safe_raw!r}): {exc}"

    def answer(self, question: str) -> str:
        """Answer a quantitative metric question."""
        if self._use_expr_eval:
            return self._answer_expr(question)
        return self._agent.run(question)


def load_metrics_dataframe(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load the reviews dataframe for metrics agent."""
    return pd.read_csv(csv_path)
