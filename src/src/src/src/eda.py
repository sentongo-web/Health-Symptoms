"""Exploratory data analysis utilities."""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FIGURE_DIR = Path("reports/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def basic_summary(df: pd.DataFrame) -> Dict[str, object]:
    """Return a concise summary of the dataset."""

    return {
        "rows": len(df),
        "diseases": df["Disease"].nunique(),
        "genders": df["Gender"].value_counts().to_dict(),
        "age_range": (int(df["Age"].min()), int(df["Age"].max())),
        "top_symptoms": df["Symptoms"].value_counts().head(5).to_dict(),
    }


def plot_distributions(df: pd.DataFrame) -> Dict[str, Path]:
    """Generate key distribution plots and return their file paths."""

    outputs = {}

    plt.figure(figsize=(10, 6))
    disease_counts = df["Disease"].value_counts().sort_values(ascending=False)
    sns.barplot(x=disease_counts.values, y=disease_counts.index, palette="viridis")
    plt.title("Disease distribution")
    plt.xlabel("Patients")
    plt.tight_layout()
    disease_path = FIGURE_DIR / "disease_distribution.png"
    plt.savefig(disease_path)
    outputs["disease_distribution"] = disease_path
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(df["Age"], bins=20, kde=True, color="steelblue")
    plt.title("Age distribution")
    plt.xlabel("Age")
    plt.tight_layout()
    age_path = FIGURE_DIR / "age_distribution.png"
    plt.savefig(age_path)
    outputs["age_distribution"] = age_path
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.countplot(x="Gender", data=df, order=df["Gender"].value_counts().index, palette="magma")
    plt.title("Gender breakdown")
    plt.xlabel("Gender")
    plt.tight_layout()
    gender_path = FIGURE_DIR / "gender_distribution.png"
    plt.savefig(gender_path)
    outputs["gender_distribution"] = gender_path
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(df["Symptom_Count"], bins=10, kde=False, color="teal")
    plt.title("Symptom count distribution")
    plt.xlabel("Number of symptoms")
    plt.tight_layout()
    symptom_count_path = FIGURE_DIR / "symptom_count_distribution.png"
    plt.savefig(symptom_count_path)
    outputs["symptom_count_distribution"] = symptom_count_path
    plt.close()

    return outputs