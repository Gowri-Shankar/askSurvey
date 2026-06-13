"""Metrics agent for quantitative queries using pandas."""

from pathlib import Path
from typing import Union

import pandas as pd


class MetricsAgent:
    """Agent for answering quantitative metric questions."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        openai_api_key: str,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
    ):
        from langchain.agents.agent_types import AgentType
        from langchain_experimental.agents import create_pandas_dataframe_agent
        from langchain_openai import ChatOpenAI

        self.dataframe = dataframe
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=openai_api_key,
        )
        self.agent = create_pandas_dataframe_agent(
            self.llm,
            self.dataframe,
            verbose=False,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            allow_dangerous_code=True,
        )

    def answer(self, question: str) -> str:
        """Answer a quantitative metric question."""
        return self.agent.run(question)


def load_metrics_dataframe(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load the reviews dataframe for metrics agent."""
    return pd.read_csv(csv_path)
