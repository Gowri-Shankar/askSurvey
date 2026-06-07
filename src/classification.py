"""Classification logic for topic/sentiment analysis."""

from dataclasses import dataclass

import pandas as pd
from transformers.pipelines import Pipeline


@dataclass
class ClassificationResult:
    """Result of a single classification."""
    topic: str
    sentiment: str
    raw_response: str
    error: str | None = None


def parse_classification_response(response: str) -> tuple[str, str]:
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


def classify_review(
    pipe: Pipeline,
    review_text: str,
    max_new_tokens: int = 246,
    temperature: float = 0.2,
) -> ClassificationResult:
    """Classify a single review."""
    from prompts import build_classification_prompt

    try:
        prompt = build_classification_prompt(str(review_text))
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
    pipe: Pipeline,
    review_column: str = "Reviews",
    max_new_tokens: int = 246,
    temperature: float = 0.2,
) -> pd.DataFrame:
    """Classify all reviews in a DataFrame."""
    df = df.copy()

    results = []
    for idx, row in df.iterrows():
        review_text = row.get(review_column, "")
        result = classify_review(pipe, review_text, max_new_tokens, temperature)
        results.append(result)

    df["model_pred_sub_topic"] = [r.topic for r in results]
    df["model_pred_sentiment"] = [r.sentiment for r in results]
    df["model_raw_response"] = [r.raw_response for r in results]
    df["model_error"] = [r.error for r in results]

    return df
