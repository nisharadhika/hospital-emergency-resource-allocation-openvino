
import joblib
import numpy as np
import openvino as ov
import pandas as pd

# Load the trained scikit-learn pipeline
pipeline = joblib.load("models/logistic_model.joblib")

# Load the dataset and select one labelled patient record
df = pd.read_csv("processed/visits_with_triage.csv")
row = df.dropna(subset=["acuity"]).iloc[[0]]

feature_columns = [
    "temperature", "heartrate", "resprate",
    "o2sat", "sbp", "dbp", "gender", "race",
    "arrival_transport", "pain", "chiefcomplaint"
]

X = row[feature_columns]

# Apply the same preprocessing used during training
preprocessor = pipeline.named_steps["preprocessor"]
X_processed = preprocessor.transform(X)

if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

X_processed = np.asarray(X_processed, dtype=np.float32)

# Original trained model prediction
sk_prediction = pipeline.predict(X)[0]

# Load the converted OpenVINO IR model
core = ov.Core()
model = core.read_model("models/openvino/logistic_model.xml")
compiled_model = core.compile_model(model, "CPU")

# Run inference
results = compiled_model({"features": X_processed})

ir_prediction = int(results[compiled_model.output("label")][0])
probabilities = results[compiled_model.output("probabilities")][0]

print("Actual acuity:", row["acuity"].iloc[0])
print("Scikit-learn prediction:", sk_prediction)
print("OpenVINO IR prediction:", ir_prediction)
print("IR class probabilities:", probabilities)
print("Predictions match:", int(sk_prediction) == ir_prediction)
