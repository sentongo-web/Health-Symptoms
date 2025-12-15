"""Cleaning and preprocessing helpers for the healthcare symptoms dataset."""

from __future__ import annotations

import re
from typing import List, Tuple

import pandas as pd


SYMPTOM_SPLIT_RE = re.compile(r",\s*")


def _clean_symptom_list(symptom_text: str) -> Tuple[str, List[str]]:
    """Normalize symptom text by lowercasing, trimming, and de-duplicating."""

    if pd.isna(symptom_text):
        return "", []

    tokens = [token.strip().lower() for token in SYMPTOM_SPLIT_RE.split(str(symptom_text)) if token.strip()]
    # De-duplicate while preserving order
    seen = set()
    cleaned_tokens = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            cleaned_tokens.append(token)
    cleaned_text = ", ".join(cleaned_tokens)
    return cleaned_text, cleaned_tokens


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the dataset with normalized symptom text.

    Steps:
        - Drop exact duplicate rows.
        - Remove rows missing key fields (Symptoms, Disease).
        - Normalize symptom text and ensure Symptom_Count aligns with parsed symptoms.
    """

    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()

    cleaned = cleaned.dropna(subset=["Symptoms", "Disease"])

    cleaned_symptoms = cleaned["Symptoms"].apply(_clean_symptom_list)
    cleaned["Symptoms"] = cleaned_symptoms.apply(lambda x: x[0])
    cleaned["Symptom_List"] = cleaned_symptoms.apply(lambda x: x[1])
    cleaned["Symptom_Count"] = cleaned["Symptom_List"].apply(len)

    # Standardize gender text
    cleaned["Gender"] = cleaned["Gender"].str.title().fillna("Other")

    # Ensure data types are consistent
    cleaned["Age"] = cleaned["Age"].astype(int)
    cleaned["Symptom_Count"] = cleaned["Symptom_Count"].astype(int)

    return cleaned.reset_index(drop=True)
