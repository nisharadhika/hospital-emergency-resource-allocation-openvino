
# test_openvino_prediction.py

import joblib
import numpy as np
import openvino as ov
import pandas as pd

# --------------------------------------------------
# 1. Load the trained scikit-learn pipeline
# --------------------------------------------------
print("Loading Logistic Regression pipeline...")

pipeline = joblib.load("models/logistic_model.joblib")

# --------------------------------------------------
# 2. Load and compile the OpenVINO ONNX model
# --------------------------------------------------
print("Loading OpenVINO model...")

core = ov.Core()

model = core.read_model(
    "models/openvino/logistic_model.onnx"
)

compiled_model = core.compile_model(model, "CPU")

# --------------------------------------------------
# 3. Load a labelled patient record
# --------------------------------------------------
print("Loading patient data...")

df = pd.read_csv(
    "processed/visits_with_triage.csv"
)

# Select one record with a known acuity label
row = df.dropna(subset=["acuity"]).iloc[[0]]

# --------------------------------------------------
# 4. Select the same features used during training
# --------------------------------------------------
feature_columns = [
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    "gender",
    "race",
    "arrival_transport",
    "pain",
    "chiefcomplaint",
]

X = row[feature_columns]

print("Patient record loaded.")
print("Number of original features:", len(feature_columns))

# --------------------------------------------------
# 5. Apply the fitted preprocessing pipeline
# --------------------------------------------------
print("Preprocessing patient features...")

preprocessor = pipeline.named_steps["preprocessor"]

X_processed = preprocessor.transform(X)

# OneHotEncoder can produce a sparse matrix.
# Convert it to a dense array for OpenVINO.
if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

X_processed = np.asarray(
    X_processed,
    dtype=np.float32
)

print("Processed feature shape:", X_processed.shape)

# Verify that the feature count matches the ONNX model.
if X_processed.shape[1] != 168:
    raise ValueError(
        f"Expected 168 processed features, "
        f"but received {X_processed.shape[1]}."
    )

# --------------------------------------------------
# 6. Predict with scikit-learn
# --------------------------------------------------
print("\nRunning scikit-learn prediction...")

sk_prediction = pipeline.predict(X)[0]

# --------------------------------------------------
# 7. Predict with OpenVINO
# --------------------------------------------------
print("Running OpenVINO prediction...")

input_port = compiled_model.input("features")
label_port = compiled_model.output("label")
probability_port = compiled_model.output("probabilities")

results = compiled_model({
    input_port: X_processed
})

ov_prediction = results[label_port][0]

probabilities = results[probability_port][0]

# --------------------------------------------------
# 8. Display and compare predictions
# --------------------------------------------------
print("\n========== PREDICTION RESULTS ==========")

print("Patient ID:", row["subject_id"].iloc[0])
print("Actual acuity:", row["acuity"].iloc[0])

print("Scikit-learn prediction:", sk_prediction)
print("OpenVINO prediction:", ov_prediction)

print("\nOpenVINO class probabilities:")

for class_label, probability in zip(
    pipeline.classes_,
    probabilities
):
    print(f"Class {class_label}: {probability:.4f}")

predictions_match = (
    int(sk_prediction) == int(ov_prediction)
)

print("\nPredictions match:", predictions_match)

if predictions_match:
    print("SUCCESS: Both models predicted the same class.")
else:
    print("WARNING: Predictions differ; investigate model conversion.")

print("\n========================================")
