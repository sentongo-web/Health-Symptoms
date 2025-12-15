"""Data loading utilities for the healthcare symptoms dataset."""

from pathlib import Path
from typing import Union

import pandas as pd


def load_dataset(data_path: Union[str, Path]) -> pd.DataFrame:
    """Load the dataset from CSV and ensure expected columns exist.

    Args:
        data_path: Path to the CSV file.

    Returns:
        DataFrame containing the loaded data.
    """

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    expected_cols = {"Patient_ID", "Age", "Gender", "Symptoms", "Symptom_Count", "Disease"}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    return df