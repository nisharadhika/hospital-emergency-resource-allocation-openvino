
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "processed" / "visits_with_triage.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["acuity"]).copy()
df["acuity"] = pd.to_numeric(df["acuity"], errors="coerce")
df = df.dropna(subset=["acuity"])
df["acuity"] = df["acuity"].astype(int)

# Exclude identifiers, target, and fields unavailable at triage time.
feature_columns = [
    "gender",
    "race",
    "arrival_transport",
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    "pain",
    "chiefcomplaint",
]

X = df[feature_columns].copy()
y = df["acuity"].copy()

# Keep only classes with at least two examples for this
# initial stratified experiment.
counts = y.value_counts()
eligible_classes = counts[counts >= 2].index
keep = y.isin(eligible_classes)
X = X.loc[keep]
y = y.loc[keep]

print("Rows available:", len(y))
print("\nClass counts:")
print(y.value_counts().sort_index())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

# Detect actual data types after reading the CSV.
numeric_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = [
    col for col in X_train.columns if col not in numeric_columns
]

print("\nNumeric features:", numeric_columns)
print("Categorical features:", categorical_columns)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nBaseline evaluation")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_test, predictions), 4),
)
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

joblib.dump(model, MODEL_DIR / "baseline_model.joblib")
joblib.dump(feature_columns, MODEL_DIR / "feature_names.joblib")

print("\nModel saved to:", MODEL_DIR / "baseline_model.joblib")
print("Research prototype only; not for clinical decisions.")
