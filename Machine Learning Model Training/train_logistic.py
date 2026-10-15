
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

df = pd.read_csv("processed/visits_with_triage.csv")
df = df.dropna(subset=["acuity"]).copy()
df["acuity"] = df["acuity"].astype(int)

# Exclude acuity and identifiers/outcomes from input features
features = [
    "gender", "race", "arrival_transport",
    "temperature", "heartrate", "resprate",
    "o2sat", "sbp", "dbp", "pain", "chiefcomplaint"
]

X = df[features]
y = df["acuity"]

numeric_features = [
    "temperature", "heartrate", "resprate",
    "o2sat", "sbp", "dbp"
]
categorical_features = [
    "gender", "race", "arrival_transport",
    "pain", "chiefcomplaint"
]

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Logistic Regression baseline")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Balanced accuracy:", round(balanced_accuracy_score(y_test, predictions), 4))
print(classification_report(y_test, predictions, zero_division=0))

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/logistic_model.joblib")
print("Saved: models/logistic_model.joblib")
