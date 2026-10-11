
from pathlib import Path
import joblib
import pandas as pd

model_path = Path("models/baseline_model.joblib")
data_path = Path("processed/visits_with_triage.csv")

print("Checking saved model...")

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}")

model = joblib.load(model_path)
df = pd.read_csv(data_path)

# Use only records with a known acuity label
df = df.dropna(subset=["acuity"]).copy()

# These are the same input features used during training
features = [
    "gender", "race", "arrival_transport",
    "temperature", "heartrate", "resprate",
    "o2sat", "sbp", "dbp", "pain", "chiefcomplaint"
]

X = df[features].head(5)
predictions = model.predict(X)

print("\nModel loaded successfully!")
print("\nSample predictions:")
for i, prediction in enumerate(predictions, start=1):
    print(f"Visit {i}: predicted acuity = {prediction}")

print("\nModel input features:", len(features))
print("Verification completed.")
