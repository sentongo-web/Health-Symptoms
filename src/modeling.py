"""Model training and evaluation helpers."""

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_COLUMNS = ["Age", "Gender", "Symptom_Count", "Symptoms"]
TARGET_COLUMN = "Disease"


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into stratified train and test sets."""

    return train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET_COLUMN],
        random_state=random_state,
    )


def build_pipeline() -> Pipeline:
    """Create a preprocessing + model pipeline."""

    numeric_features = ["Age", "Symptom_Count"]
    categorical_features = ["Gender"]
    text_features = "Symptoms"

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            )
        ]
    )
    text_transformer = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=5,
                    max_features=5000,
                ),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("text", text_transformer, text_features),
        ]
    )

    model = LogisticRegression(max_iter=1000, multi_class="auto", n_jobs=-1)

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return clf


def train_model(df_train: pd.DataFrame) -> Pipeline:
    """Train the pipeline on the provided training data."""

    clf = build_pipeline()
    clf.fit(df_train[FEATURE_COLUMNS], df_train[TARGET_COLUMN])
    return clf


def evaluate_model(model: Pipeline, df_test: pd.DataFrame) -> Dict[str, object]:
    """Evaluate the model on the test set."""

    X_test = df_test[FEATURE_COLUMNS]
    y_true = df_test[TARGET_COLUMN]
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(y_true, y_pred, output_dict=False)

    return {"accuracy": accuracy, "f1_weighted": f1, "classification_report": report}


def save_model(model: Pipeline, path: Path) -> Path:
    """Persist the trained pipeline to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path) -> Pipeline:
    """Load a persisted model pipeline."""

    return joblib.load(path)
