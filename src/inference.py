"""Simple inference helpers for the trained classifier."""

from typing import List

import pandas as pd
from sklearn.pipeline import Pipeline

from .modeling import FEATURE_COLUMNS


def predict(model: Pipeline, records: List[dict]) -> pd.DataFrame:
    """Generate predictions for a list of patient dictionaries."""

    df = pd.DataFrame(records)
    # Ensure required columns exist even if empty
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    preds = model.predict(df[FEATURE_COLUMNS])
    proba = model.predict_proba(df[FEATURE_COLUMNS])
    top_prob = proba.max(axis=1)

    output = df.copy()
    output["Predicted_Disease"] = preds
    output["Confidence"] = top_prob
    return output
