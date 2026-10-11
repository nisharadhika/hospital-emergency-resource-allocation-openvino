
import time
import statistics
import joblib
import numpy as np
import pandas as pd
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

# Use one record for the end-to-end latency test
X = df[features].iloc[[0]]

preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]

core = ov.Core()
onnx_model = core.read_model("models/openvino/logistic_model.onnx")
compiled = core.compile_model(onnx_model, "CPU")

def transform_input(data):
    transformed = preprocessor.transform(data)
    if sparse.issparse(transformed):
        transformed = transformed.toarray()
    return np.asarray(transformed, dtype=np.float32)

def sklearn_end_to_end():
    return pipeline.predict(X)

def openvino_end_to_end():
    transformed = transform_input(X)
    result = compiled({"features": transformed})
    return np.asarray(
        result[compiled.output("label")]
    ).reshape(-1)

# Warm-up
for _ in range(50):
    sklearn_end_to_end()
    openvino_end_to_end()

sk_times = []
ov_times = []
rounds = 10
iterations = 1000

for r in range(rounds):
    # Alternate order to reduce order bias
    functions = (
        [(sklearn_end_to_end, sk_times),
         (openvino_end_to_end, ov_times)]
        if r % 2 == 0 else
        [(openvino_end_to_end, ov_times),
         (sklearn_end_to_end, sk_times)]
    )

    for fn, times in functions:
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        times.append(
            (time.perf_counter() - start) * 1000 / iterations
        )

def report(name, values):
    print(f"\n{name}")
    print(f"Mean: {statistics.mean(values):.6f} ms")
    print(f"Median: {statistics.median(values):.6f} ms")
    print(f"Std deviation: {statistics.stdev(values):.6f} ms")

print("End-to-end single-record benchmark")
print("Preprocessing included; compilation excluded.")
print("Rounds:", rounds, "| Calls per round:", iterations)

report("Scikit-learn pipeline", sk_times)
report("Preprocessing + OpenVINO", ov_times)

print(
    "\nMean speed ratio:",
    f"{statistics.mean(sk_times) / statistics.mean(ov_times):.2f}x"
)
