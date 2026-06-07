"""Text preprocessing utilities."""

import re
import pandas as pd


def clean_text(value: object) -> str:
    """Clean a single text value."""
    if value is None:
        return ""

    text = str(value).strip()

    # Remove leading/trailing double newlines
    while text.startswith("\n\n"):
        text = text[2:]
    while text.endswith("\n\n"):
        text = text[:-2]

    # Normalize multiple newlines to double newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize multiple spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def clean_text_column(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Clean a text column in a DataFrame."""
    df = df.copy()
    if column in df.columns:
        df[column] = df[column].apply(clean_text)
    return df