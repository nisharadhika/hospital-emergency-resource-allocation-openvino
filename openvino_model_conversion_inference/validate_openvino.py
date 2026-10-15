
import time
import joblib
import numpy as np
import pandas as pd
import openvino as ov
from scipy import sparse

# Load the original trained pipeline
pipeline = joblib.load("models/logistic_model.joblib")

# Load labelled dataset
df = pd.read_csv("processed/visits_with_triage.csv")
df = df.dropna(subset=["acuity"]).copy()

features = [
    "temperature", "heartrate", "resprate",
    "o2sat", "sbp", "dbp",
    "gender", "race", "arrival_transport",
    "pain", "chiefcomplaint"
]

X = df[features]

# Reference predictions from the complete sklearn pipeline
sklearn_predictions = pipeline.predict(X)

# Apply the SAME fitted preprocessing used during training
preprocessor = pipeline.named_steps["preprocessor"]
X_processed = preprocessor.transform(X)

if sparse.issparse(X_processed):
    X_processed = X_processed.toarray()

X_processed = np.asarray(X_processed, dtype=np.float32)

print("Rows tested:", len(X))
print("Preprocessed shape:", X_processed.shape)

# Load and compile ONNX model using OpenVINO CPU
core = ov.Core()
onnx_model = core.read_model("models/openvino/logistic_model.onnx")
compiled_model = core.compile_model(onnx_model, "CPU")

# Run inference
result = compiled_model({ "features": X_processed })

openvino_predictions = np.asarray(
    result[compiled_model.output("label")]
).reshape(-1).astype(int)

# Compare predictions
matches = np.sum(sklearn_predictions == openvino_predictions)
total = len(sklearn_predictions)
agreement = matches / total * 100

print("\n--- Prediction Validation ---")
print("Matching predictions:", matches, "/", total)
print(f"Prediction agreement: {agreement:.2f}%")

if np.array_equal(sklearn_predictions, openvino_predictions):
    print("PASS: All predictions match.")
else:
    print("WARNING: Some predictions differ.")
    mismatches = np.where(
        sklearn_predictions != openvino_predictions
    )[0]
    for i in mismatches[:10]:
        print(
            f"Row {i}: sklearn={sklearn_predictions[i]}, "
            f"OpenVINO={openvino_predictions[i]}"
        )

# Basic CPU inference timing after compilation
for _ in range(5):
    compiled_model({"features": X_processed})

runs = 50
start = time.perf_counter()

for _ in range(runs):
    compiled_model({"features": X_processed})

elapsed = time.perf_counter() - start

print("\n--- OpenVINO Batch Inference Timing ---")
print(f"Rows per batch: {len(X_processed)}")
print(f"Average batch inference: {elapsed / runs * 1000:.3f} ms")
print("Note: preprocessing and model compilation are excluded.")
