"""Data I/O utilities for loading and saving data files."""

import pandas as pd
from pathlib import Path


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a table from Excel or CSV file."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    elif suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: .xlsx, .xls, .csv")


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to Excel or CSV file."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        df.to_excel(path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: .xlsx, .xls, .csv")


def merge_topic_lookup(
    predictions_df: pd.DataFrame,
    topics_df: pd.DataFrame,
    predicted_topic_column: str = "model_pred_sub_topic",
    lookup_column: str = "Sub-Topic",
) -> pd.DataFrame:
    """Merge classification results with topic lookup table."""
    return predictions_df.merge(
        topics_df,
        left_on=predicted_topic_column,
        right_on=lookup_column,
        how="left",
    )