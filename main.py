"""End-to-end pipeline for the Healthcare Symptoms dataset.

This script performs:
- Data loading and cleaning
- Exploratory data analysis (summary + charts)
- Train/test split
- Feature engineering with mixed tabular + text data
- Model training and evaluation
- Model persistence and a brief inference demo
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from src import data_loader, data_preparation, eda, modeling
from src.inference import predict


ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "symptom_classifier.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Symptoms-to-disease classification pipeline")
    parser.add_argument("--data", type=str, default="Healthcare.csv", help="Path to dataset CSV")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Proportion reserved for the test set"
    )
    return parser.parse_args()


def run_pipeline(data_path: str, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading dataset...")
    df_raw = data_loader.load_dataset(data_path)
    print(f"Loaded {len(df_raw):,} rows with {df_raw['Disease'].nunique()} diseases.")

    print("Cleaning dataset...")
    df_clean = data_preparation.clean_dataset(df_raw)
    print(f"Cleaned dataset has {len(df_clean):,} rows after processing.")

    print("Running EDA...")
    summary = eda.basic_summary(df_clean)
    charts = eda.plot_distributions(df_clean)
    print("Summary stats:")
    for key, value in summary.items():
        print(f" - {key}: {value}")
    for name, path in charts.items():
        print(f"Chart saved: {name} -> {path}")

    print("Splitting data...")
    train_df, test_df = modeling.split_data(df_clean, test_size=test_size)
    print(f"Train rows: {len(train_df):,}; Test rows: {len(test_df):,}")

    print("Training model...")
    model = modeling.train_model(train_df)

    print("Evaluating model...")
    metrics = modeling.evaluate_model(model, test_df)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Weighted F1: {metrics['f1_weighted']:.3f}")
    print("Classification report:\n" + metrics["classification_report"])

    print(f"Saving model to {MODEL_PATH} ...")
    modeling.save_model(model, MODEL_PATH)

    print("Inference demo (first 3 patients)...")
    sample_records = test_df.head(3)[modeling.FEATURE_COLUMNS].to_dict(orient="records")
    preds = predict(model, sample_records)
    print(preds[["Symptoms", "Predicted_Disease", "Confidence"]])

    return train_df, test_df


def main() -> None:
    args = parse_args()
    run_pipeline(args.data, test_size=args.test_size)


if __name__ == "__main__":
    main()