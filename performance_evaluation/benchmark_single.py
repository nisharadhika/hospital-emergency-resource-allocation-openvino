
import time
import statistics
import joblib
import pandas as pd
import numpy as np
import openvino as ov
from scipy import sparse

pipeline = joblib.load("models/logistic_model.joblib")
df = pd.read_csv("processed/visits_with_triage.csv")
df = df.dropna(subset=["acuity"]).copy()

features = [
    "temperature", "heartrate", "resprate",
    "o2sat", "sbp", "dbp",
    "gender", "race", "arrival_transport",
    "pain", "chiefcomplaint"
]

X = df[features].iloc[[0]]
preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]

# Transform a single record once
X_data = preprocessor.transform(X)
if sparse.issparse(X_data):
    X_data = X_data.toarray()
X_data = np.asarray(X_data, dtype=np.float32)

core = ov.Core()
model = core.read_model("models/openvino/logistic_model.onnx")
compiled = core.compile_model(model, "CPU")
ov_input = {"features": X_data}

# Warm-up
for _ in range(50):
    classifier.predict(X_data)
    compiled(ov_input)

sk_times = []
ov_times = []
rounds = 10
iterations = 1000

for r in range(rounds):
    # Alternate which implementation runs first
    if r % 2 == 0:
        start = time.perf_counter()
        for _ in range(iterations):
            classifier.predict(X_data)
        sk_times.append(
            (time.perf_counter() - start) * 1000 / iterations
        )

        start = time.perf_counter()
        for _ in range(iterations):
            compiled(ov_input)
        ov_times.append(
            (time.perf_counter() - start) * 1000 / iterations
        )
    else:
        start = time.perf_counter()
        for _ in range(iterations):
            compiled(ov_input)
        ov_times.append(
            (time.perf_counter() - start) * 1000 / iterations
        )

        start = time.perf_counter()
        for _ in range(iterations):
            classifier.predict(X_data)
        sk_times.append(
            (time.perf_counter() - start) * 1000 / iterations
        )

print("Single-record classifier-only benchmark")
print("Timing rounds:", rounds)
print("Iterations per round:", iterations)

for name, values in [
    ("Scikit-learn", sk_times),
    ("OpenVINO CPU", ov_times)
]:
    print(f"\n{name}")
    print(f"Mean: {statistics.mean(values):.6f} ms")
    print(f"Median: {statistics.median(values):.6f} ms")
    print(f"Std deviation: {statistics.stdev(values):.6f} ms")

print(
    "\nMean speed ratio:",
    f"{statistics.mean(sk_times) / statistics.mean(ov_times):.2f}x"
)
print("Preprocessing and compilation excluded.")
