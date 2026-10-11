

import time
import statistics
import joblib
import numpy as np
import pandas as pd
import openvino as ov
from scipy import sparse

# Load the trained model and labelled dataset
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

# Transform once; both models receive the same data
X_data = preprocessor.transform(X)
if sparse.issparse(X_data):
    X_data = X_data.toarray()
X_data = np.asarray(X_data, dtype=np.float32)

core = ov.Core()
onnx_model = core.read_model("models/openvino/logistic_model.onnx")
compiled = core.compile_model(onnx_model, "CPU")
ov_input = {"features": X_data}

# Warm-up
for _ in range(30):
    classifier.predict(X_data)
    compiled(ov_input)

# Alternate benchmark order to reduce order bias
sklearn_times = []
openvino_times = []

for repeat in range(10):
    if repeat % 2 == 0:
        start = time.perf_counter()
        for _ in range(100):
            classifier.predict(X_data)
        sklearn_times.append(
            (time.perf_counter() - start) * 1000 / 100
        )

        start = time.perf_counter()
        for _ in range(100):
            compiled(ov_input)
        openvino_times.append(
            (time.perf_counter() - start) * 1000 / 100
        )
    else:
        start = time.perf_counter()
        for _ in range(100):
            compiled(ov_input)
        openvino_times.append(
            (time.perf_counter() - start) * 1000 / 100
        )

        start = time.perf_counter()
        for _ in range(100):
            classifier.predict(X_data)
        sklearn_times.append(
            (time.perf_counter() - start) * 1000 / 100
        )

def report(name, values):
    print(f"\n{name}")
    print(f"Mean batch latency: {statistics.mean(values):.6f} ms")
    print(f"Median batch latency: {statistics.median(values):.6f} ms")
    print(f"Std. deviation: {statistics.stdev(values):.6f} ms")
    print(f"Minimum: {min(values):.6f} ms")
    print(f"Maximum: {max(values):.6f} ms")

print("Rows per batch:", len(X_data))
print("Independent timing rounds:", len(sklearn_times))
print("Iterations per timing round: 100")

report("Scikit-learn classifier", sklearn_times)
report("OpenVINO CPU classifier", openvino_times)

sk_mean = statistics.mean(sklearn_times)
ov_mean = statistics.mean(openvino_times)

print("\nMean speed ratio (sklearn/OpenVINO): "
      f"{sk_mean / ov_mean:.2f}x")
print("Preprocessing and compilation are excluded.")
