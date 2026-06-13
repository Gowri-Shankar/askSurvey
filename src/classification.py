"""Classification logic for topic/sentiment analysis."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import pandas as pd

if TYPE_CHECKING:
    from transformers.pipelines import Pipeline


@dataclass
class ClassificationResult:
    """Result of a single classification."""
    topic: str
    sentiment: str
    raw_response: str
    error: Optional[str] = None


def parse_classification_response(response: str) -> Tuple[str, str]:
    """Parse model response into topic and sentiment."""
    response = response.strip()

    # Remove markdown bold markers
    response = response.replace("**", "")

    # Take the last non-empty line
    lines = [line.strip() for line in response.split("\n") if line.strip()]
    if lines:
        response = lines[-1]

    # Split by --
    if "--" in response:
        parts = response.split("--", 1)
        topic = parts[0].strip()
        sentiment = parts[1].strip()

        # Validate sentiment
        valid_sentiments = {"Positive", "Neutral", "Negative"}
        if sentiment not in valid_sentiments:
            sentiment = "Error"

        return topic, sentiment

    return "Error", "Error"


def _build_prompt(pipe: "Pipeline", review_text: str) -> str:
    """Build prompt applying chat template if the tokenizer supports it."""
    from prompts import build_classification_prompt

    instruction = build_classification_prompt(str(review_text))
    tokenizer = pipe.tokenizer

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return instruction


def classify_review(
    pipe: "Pipeline",
    review_text: str,
    max_new_tokens: int = 64,
    temperature: float = 0.2,
) -> ClassificationResult:
    """Classify a single review."""
    try:
        prompt = _build_prompt(pipe, review_text)
        result = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        response = result[0]["generated_text"].strip()
        topic, sentiment = parse_classification_response(response)

        return ClassificationResult(
            topic=topic,
            sentiment=sentiment,
            raw_response=response,
            error=None,
        )
    except Exception as e:
        return ClassificationResult(
            topic="Error",
            sentiment="Error",
            raw_response="",
            error=str(e),
        )


def classify_dataframe(
    df: pd.DataFrame,
    pipe: "Pipeline",
    review_column: str = "Reviews",
    max_new_tokens: int = 246,
    temperature: float = 0.2,
) -> pd.DataFrame:
    """Classify all reviews in a DataFrame."""
    df = df.copy()

    from tqdm import tqdm

    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
        review_text = row.get(review_column, "")
        result = classify_review(pipe, review_text, max_new_tokens, temperature)
        results.append(result)

    df["model_pred_sub_topic"] = [r.topic for r in results]
    df["model_pred_sentiment"] = [r.sentiment for r in results]
    df["model_raw_response"] = [r.raw_response for r in results]
    df["model_error"] = [r.error for r in results]

    return df
