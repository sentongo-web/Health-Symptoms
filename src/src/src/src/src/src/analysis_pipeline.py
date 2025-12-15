"""
End-to-end disease classification workflow for the Healthcare Symptoms dataset.

The script demonstrates loading data, cleaning and normalizing symptom text, feature
engineering, model training, evaluation, and model persistence for deployment.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(csv_path: str = "Healthcare.csv") -> pd.DataFrame:
    """Load the dataset, standardize column names, and enforce dtypes."""
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Symptom_Count"] = pd.to_numeric(df["Symptom_Count"], errors="coerce")
    return df


def normalize_symptom_list(symptoms: str) -> List[str]:
    """Lowercase and deduplicate comma-separated symptoms."""
    if not isinstance(symptoms, str):
        return []
    items = [item.strip().lower() for item in symptoms.split(",") if item.strip()]
    return sorted(set(items))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, handle missing values, and normalize text."""
    cleaned = df.copy()
    if "Patient_ID" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["Patient_ID"])

    cleaned["Gender"] = cleaned.get("Gender", "Unknown").fillna("Unknown").astype(str)
    cleaned["Gender"] = cleaned["Gender"].str.strip().str.title()
    cleaned["Symptoms"] = cleaned.get("Symptoms", "").fillna("")

    cleaned["Symptom_List"] = cleaned["Symptoms"].apply(normalize_symptom_list)
    cleaned["Symptom_Tokens"] = cleaned["Symptom_List"].apply(
        lambda items: " ".join(token.replace(" ", "_") for token in items)
    )

    inferred_counts = cleaned["Symptom_List"].apply(len)
    cleaned["Symptom_Count"] = cleaned.get("Symptom_Count", inferred_counts)
    cleaned["Symptom_Count"] = cleaned["Symptom_Count"].fillna(inferred_counts)

    age_default = pd.Series([pd.NA] * len(cleaned))
    age_series = pd.to_numeric(cleaned.get("Age", age_default), errors="coerce")
    median_age = age_series.median() if not age_series.dropna().empty else 0
    cleaned["Age"] = age_series.fillna(median_age)

    cleaned = cleaned.dropna(subset=["Disease"])
    return cleaned


def build_preprocessor() -> ColumnTransformer:
    """Create a preprocessing pipeline for demographic and symptom features."""
    symptom_vectorizer = CountVectorizer(
        token_pattern=r"(?u)\\b[\\w_]+\\b",
        min_df=2,
        binary=True,
    )

    gender_encoder = OneHotEncoder(handle_unknown="ignore")
    numeric_scaler = StandardScaler()

    return ColumnTransformer(
        transformers=[
            ("symptoms", symptom_vectorizer, "Symptom_Tokens"),
            ("gender", gender_encoder, ["Gender"]),
            ("numeric", numeric_scaler, ["Age", "Symptom_Count"]),
        ],
        remainder="drop",
    )


def build_classifier(preprocessor: ColumnTransformer) -> Pipeline:
    """Assemble the full modeling pipeline."""
    classifier = LogisticRegression(max_iter=250, multi_class="auto")
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[Pipeline, Dict[str, float]]:
    """Train the classifier and return evaluation metrics."""
    features = df[["Symptom_Tokens", "Gender", "Age", "Symptom_Count"]]
    labels = df["Disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    pipeline = build_classifier(build_preprocessor())
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
    }
    metrics["classification_report"] = classification_report(
        y_test, predictions, output_dict=False
    )
    metrics["confusion_matrix"] = confusion_matrix(y_test, predictions)
    return pipeline, metrics


def persist_model(model: Pipeline, output_path: str = "models/disease_classifier.joblib") -> Path:
    """Save the fitted pipeline for deployment."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    return output


def run_workflow(csv_path: str = "Healthcare.csv") -> Dict[str, object]:
    """Execute the full workflow from raw data to persisted model."""
    data = load_data(csv_path)
    cleaned = clean_data(data)
    model, metrics = train_model(cleaned)
    model_path = persist_model(model)
    return {
        "rows": len(cleaned),
        "model_path": str(model_path),
        "metrics": metrics,
    }


if __name__ == "__main__":
    results = run_workflow()
    print(f"Cleaned rows: {results['rows']}")
    print(f"Model saved to: {results['model_path']}")
    print("\nClassification report:\n")
    print(results["metrics"]["classification_report"])