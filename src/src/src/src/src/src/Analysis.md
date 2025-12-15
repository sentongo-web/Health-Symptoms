# Healthcare Symptoms–Disease Classification: End-to-End Plan

This guide describes how to work with `Healthcare.csv` from raw ingestion through
model training and deployment. The accompanying `analysis_pipeline.py` script
implements the core steps using scikit-learn.

## 1. Environment and Data Loading
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run the workflow end-to-end (reads `Healthcare.csv` from the repo root):
   ```bash
   python analysis_pipeline.py
   ```
   The script prints evaluation metrics and saves the fitted pipeline to
   `models/disease_classifier.joblib`.

## 2. Data Cleaning
- Standardize column names and enforce numeric types for `Age` and `Symptom_Count`.
- Drop duplicate `Patient_ID` rows to prevent leakage.
- Normalize `Gender` casing and fill missing values with `Unknown`.
- Clean symptom text by lowercasing, trimming whitespace, and deduplicating tokens.
- Recompute `Symptom_Count` from the parsed tokens when the column is missing and
  impute missing ages with the median.

## 3. Exploratory Data Analysis (EDA)
Suggested quick checks (can be executed in a notebook or short script):
- Class balance: `df['Disease'].value_counts(normalize=True)` to flag imbalance.
- Symptom popularity: explode `Symptom_List` and compute frequencies to discover
  high-impact symptoms.
- Demographics: age histogram, gender distribution, and cross-tabs with diseases
  to uncover demographic effects.
- Data quality: check missingness per column and inspect rare symptom tokens to
  decide on pruning thresholds.

## 4. Feature Engineering
- **Text features:** the script converts comma-separated symptoms into tokens and
  uses a binary `CountVectorizer` with `min_df=2` to drop extremely rare tokens.
- **Categorical:** one-hot encode `Gender` with `handle_unknown="ignore"` to keep
  the pipeline resilient to new categories.
- **Numeric:** standardize `Age` and `Symptom_Count` for algorithms sensitive to
  feature scale.

## 5. Modeling and Evaluation
- Train/test split with stratification to preserve class balance.
- Baseline classifier: multinomial logistic regression with a `max_iter` of 250
  to ensure convergence on the multi-class problem.
- Reported metrics include accuracy, a full classification report, and the
  confusion matrix (available in `results['metrics']`).
- For deeper analysis, consider macro/micro F1-scores and per-disease recall to
  highlight potential bias against rare classes.

## 6. Iteration Ideas
- Try TF-IDF features or character n-grams to capture phrase nuances.
- Evaluate linear SVM (`LinearSVC`) or gradient boosting (`HistGradientBoostingClassifier`).
- Use class weights (`class_weight="balanced"`) if you observe substantial class
  imbalance.
- Hyperparameter tuning via `GridSearchCV` or `RandomizedSearchCV` on the pipeline
  to keep preprocessing and modeling coupled.

## 7. Deployment Notes
- The persisted `joblib` artifact is a full scikit-learn pipeline that includes
  preprocessing; it only needs the raw feature columns for inference.
- To serve predictions:
  ```python
  import joblib
  import pandas as pd

  model = joblib.load("models/disease_classifier.joblib")
  new_rows = pd.DataFrame([
      {
          "Symptoms": "fever, cough, fatigue",
          "Gender": "Female",
          "Age": 34,
          "Symptom_Count": 3,
      }
  ])
  # Reuse cleaning utilities from analysis_pipeline to align preprocessing
  from analysis_pipeline import clean_data
  cleaned = clean_data(new_rows)
  preds = model.predict(cleaned[["Symptom_Tokens", "Gender", "Age", "Symptom_Count"]])
  ```
- Package the model behind a FastAPI endpoint or batch-scoring job; ship the
  `requirements.txt` file alongside the artifact for reproducible environments.