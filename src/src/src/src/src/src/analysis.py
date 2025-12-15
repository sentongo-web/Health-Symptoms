"""
Comprehensive pipeline for the Healthcare Symptoms–Disease Classification dataset.

Steps included:
- Data loading and cleaning
- Exploratory analysis with summary artifacts
- Feature engineering for structured + text features
- Model training, evaluation, and persistence
- Convenience prediction function for deployment-style inference

Run the script:
    python analysis.py --train

Generate sample predictions after training:
    python analysis.py --predict-samples 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("Healthcare.csv")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the CSV dataset and ensure expected columns exist."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    expected_cols = {"Patient_ID", "Age", "Gender", "Symptoms", "Symptom_Count", "Disease"}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text, standardize symptom strings, and verify counts."""
    cleaned = df.copy()
    cleaned["Symptoms"] = cleaned["Symptoms"].astype(str).str.lower()
    # Split on comma and strip whitespace
    cleaned["Symptom_List"] = cleaned["Symptoms"].str.split(",").apply(
        lambda items: [item.strip() for item in items if item.strip()]
    )
    cleaned["Normalized_Symptoms"] = cleaned["Symptom_List"].apply(" ".join)

    # Replace provided Symptom_Count with recalculated counts to prevent drift
    cleaned["Symptom_Count"] = cleaned["Symptom_List"].apply(len)

    # Ensure categorical consistency
    cleaned["Gender"] = cleaned["Gender"].fillna("Unknown").str.title()
    cleaned["Disease"] = cleaned["Disease"].str.strip()
    return cleaned


def exploratory_summary(df: pd.DataFrame) -> Dict:
    """Compute lightweight EDA summaries and save them to JSON."""
    summary = {
        "row_count": len(df),
        "age": {
            "min": float(df["Age"].min()),
            "max": float(df["Age"].max()),
            "median": float(df["Age"].median()),
            "mean": float(df["Age"].mean()),
        },
        "gender_distribution": df["Gender"].value_counts().to_dict(),
        "top_diseases": df["Disease"].value_counts().head(10).to_dict(),
        "symptom_count_distribution": df["Symptom_Count"].value_counts().sort_index().to_dict(),
        "most_common_symptoms": _most_common_symptoms(df["Symptom_List"], top_n=15),
    }

    summary_path = ARTIFACT_DIR / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def _most_common_symptoms(symptom_lists: pd.Series, top_n: int = 15) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for items in symptom_lists:
        for item in items:
            counts[item] = counts.get(item, 0) + 1
    sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return dict(sorted_items[:top_n])


def build_model_pipeline() -> Pipeline:
    """Create the preprocessing + classifier pipeline."""
    text_features = "Normalized_Symptoms"
    numeric_features = ["Age", "Symptom_Count"]
    categorical_features = ["Gender"]

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000),
                text_features,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
        ],
        sparse_threshold=0.3,
    )

    classifier = LogisticRegression(max_iter=1000, multi_class="multinomial")

    model = Pipeline(
        steps=[
            ("preprocess", preprocessing),
            ("classifier", classifier),
        ]
    )
    return model


def train_and_evaluate(df: pd.DataFrame) -> Dict:
    """Train the classifier and return evaluation metrics."""
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["Disease"], random_state=42
    )

    model = build_model_pipeline()
    model.fit(train_df, train_df["Disease"])

    predictions = model.predict(test_df)
    metrics = {
        "accuracy": float(accuracy_score(test_df["Disease"], predictions)),
        "macro_f1": float(f1_score(test_df["Disease"], predictions, average="macro")),
        "classification_report": classification_report(
            test_df["Disease"], predictions, output_dict=True
        ),
    }

    metrics_path = ARTIFACT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    model_path = ARTIFACT_DIR / "disease_classifier.joblib"
    joblib.dump(model, model_path)

    return metrics


def predict_samples(model_path: Path, df: pd.DataFrame, n_samples: int = 5) -> pd.DataFrame:
    """Generate sample predictions for quick sanity checks."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train first.")

    model: Pipeline = joblib.load(model_path)
    samples = df.sample(n=min(n_samples, len(df)), random_state=0)
    samples = samples.copy()
    samples["Predicted_Disease"] = model.predict(samples)

    output_path = ARTIFACT_DIR / "sample_predictions.csv"
    samples.to_csv(output_path, index=False)
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Healthcare symptoms classification pipeline")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model and generate EDA + metrics artifacts",
    )
    parser.add_argument(
        "--predict-samples",
        type=int,
        default=0,
        metavar="N",
        help="Generate N sample predictions using the trained model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset()
    df = clean_dataset(df)

    # Always create EDA artifact on run
    eda_summary = exploratory_summary(df)
    print("EDA summary saved to artifacts/eda_summary.json")
    print(json.dumps(eda_summary, indent=2))

    metrics = None
    if args.train:
        metrics = train_and_evaluate(df)
        print("Training completed. Metrics saved to artifacts/metrics.json")
        print(json.dumps(metrics, indent=2))

    if args.predict_samples > 0:
        model_path = ARTIFACT_DIR / "disease_classifier.joblib"
        samples = predict_samples(model_path, df, n_samples=args.predict_samples)
        print(samples[["Patient_ID", "Disease", "Predicted_Disease"]])
        print("Sample predictions saved to artifacts/sample_predictions.csv")


if __name__ == "__main__":
    main()