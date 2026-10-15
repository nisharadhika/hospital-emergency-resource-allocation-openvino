
import time
import numpy as np
import joblib
import pandas as pd
import openvino as ov
from scipy import sparse

# Load trained pipeline and labelled data
pipeline = joblib.load("models/logistic_model.joblib")
df = pd.read_csv("processed/visits_with_triage.csv")
df = df.dropna(subset=["acuity"]).copy()

features = [
    "temperature", "heartrate", "resprate",
    "o2sat", "sbp", "dbp",
    "gender", "race", "arrival_transport",
    "pain", "chiefcomplaint"
]

X = df[features]
preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]

# Transform once so both classifiers receive identical inputs
X_transformed = preprocessor.transform(X)
if sparse.issparse(X_transformed):
    X_transformed = X_transformed.toarray()

X_transformed = np.asarray(X_transformed, dtype=np.float32)

# Compile OpenVINO model
core = ov.Core()
model = core.read_model("models/openvino/logistic_model.onnx")
compiled = core.compile_model(model, "CPU")
input_data = {"features": X_transformed}

# Warm up both implementations
for _ in range(10):
    classifier.predict(X_transformed)
    compiled(input_data)

runs = 500

# Benchmark sklearn classifier only
start = time.perf_counter()
for _ in range(runs):
    classifier.predict(X_transformed)
sklearn_ms = (time.perf_counter() - start) * 1000 / runs

# Benchmark OpenVINO classifier only
start = time.perf_counter()
for _ in range(runs):
    compiled(input_data)
openvino_ms = (time.perf_counter() - start) * 1000 / runs

print("\n--- Classifier-Only Benchmark ---")
print("Rows per batch:", len(X_transformed))
print("Iterations:", runs)
print(f"Scikit-learn average: {sklearn_ms:.6f} ms/batch")
print(f"OpenVINO average:     {openvino_ms:.6f} ms/batch")

if openvino_ms > 0:
    print(f"Speed ratio (sklearn/OpenVINO): {sklearn_ms/openvino_ms:.2f}x")

print("\nPreprocessing and model compilation are excluded.")
print("Timing is for batch inference, not single-patient latency.")
