# Healthcare Symptoms – Disease Classification

An end-to-end workflow for exploring and modeling the synthetic Healthcare Symptoms–Disease Classification dataset. The project includes data loading, cleaning, exploratory analysis, feature engineering, model training, evaluation, and a small inference demo.

## Dataset

The dataset (`Healthcare.csv`) contains 25,000 synthetic patient records with demographics, comma-separated symptom text, and a confirmed diagnosis across 30 diseases. Key columns:

- `Patient_ID`: unique identifier
- `Age`: patient age (1–90)
- `Gender`: Male, Female, or Other
- `Symptoms`: comma-separated list of 3–7 symptoms
- `Symptom_Count`: number of symptoms
- `Disease`: one of 30 diagnoses

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run the pipeline

Run the entire workflow (cleaning, EDA, model training/evaluation, inference demo):

```bash
python main.py --data Healthcare.csv --test-size 0.2
```

Outputs:
- Saved model pipeline at `artifacts/symptom_classifier.joblib`
- EDA charts under `reports/figures/`
- Console summary with dataset stats, metrics, and a sample prediction table

## Workflow details

1. **Data loading** (`src/data_loader.py`): validates expected columns and reads the CSV.
2. **Cleaning & preparation** (`src/data_preparation.py`):
   - Drops duplicate rows and missing symptoms/diagnoses
   - Normalizes symptom text (lowercase, trim, de-duplicate), recreates `Symptom_Count`, and extracts `Symptom_List`
   - Standardizes gender labels and enforces integer types for numeric columns
3. **Exploratory analysis** (`src/eda.py`):
   - Dataset summary (row counts, unique diseases, gender mix, age range)
   - Distribution plots for diseases, ages, gender, and symptom counts saved to `reports/figures/`
4. **Modeling** (`src/modeling.py`):
   - Feature columns: `Age`, `Gender`, `Symptom_Count`, `Symptoms`
   - `ColumnTransformer` combines numeric scaling, gender one-hot encoding, and TF–IDF bigrams on symptom text
   - Multiclass `LogisticRegression` trained with stratified train/test split and evaluated via accuracy, weighted F1, and a classification report
5. **Inference** (`src/inference.py`): convenience helper to score new patient records with predicted disease labels and confidences using the persisted pipeline.

## Inference example

```python
from pathlib import Path
import pandas as pd
from src.modeling import load_model
from src.inference import predict

model = load_model(Path("artifacts/symptom_classifier.joblib"))
sample_records = [
    {"Age": 45, "Gender": "Female", "Symptom_Count": 3, "Symptoms": "headache, nausea, sensitivity to light"},
    {"Age": 67, "Gender": "Male", "Symptom_Count": 4, "Symptoms": "fever, cough, sore throat, fatigue"},
]

predictions = predict(model, sample_records)
print(predictions[["Symptoms", "Predicted_Disease", "Confidence"]])
```

## Project structure

```
├── Healthcare.csv
├── README.md
├── artifacts/
├── main.py
├── requirements.txt
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── data_preparation.py
    ├── eda.py
    ├── inference.py
    └── modeling.py
```