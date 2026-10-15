
from pathlib import Path
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent

LOGISTIC_PATH = ROOT / "models" / "logistic_model.joblib"
RF_PATH = ROOT / "models" / "baseline_model.joblib"

FEATURES = [
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

# Find the processed dataset without assuming its folder name
matches = list(ROOT.rglob("visits_with_triage.csv"))

if not matches:
    raise FileNotFoundError(
        "Cannot find visits_with_triage.csv. "
        "Check that the processed dataset is inside D:\\CAO_proj."
    )

DATA_PATH = matches[0]

print("=" * 65)
print("GRAPH INPUT COLLECTION")
print("=" * 65)
print("Dataset:", DATA_PATH)

# --------------------------------------------------
# GRAPH 1: CONFUSION MATRIX
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)

missing_features = [c for c in FEATURES if c not in df.columns]
if "acuity" not in df.columns:
    raise ValueError("The dataset does not contain the 'acuity' target column.")

if missing_features:
    raise ValueError(f"Missing feature columns: {missing_features}")

df = df.dropna(subset=["acuity"]).copy()
X = df[FEATURES]
y = df["acuity"]

# Match the project's 75/25 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

logistic = joblib.load(LOGISTIC_PATH)
logistic_pred = logistic.predict(X_test)

labels = sorted(pd.unique(y))
cm = confusion_matrix(y_test, logistic_pred, labels=labels)

print("\nGRAPH 1 — CONFUSION MATRIX")
print("Acuity label order:", labels)
print("Rows = actual labels; columns = predicted labels")
print("Matrix:")
print(cm)
print("\nClassification report:")
print(classification_report(y_test, logistic_pred, zero_division=0))

# --------------------------------------------------
# GRAPH 2: MODEL PERFORMANCE COMPARISON
# --------------------------------------------------
print("\nGRAPH 2 — MODEL PERFORMANCE COMPARISON")
models = {"Logistic Regression": logistic}

if RF_PATH.exists():
    models["Random Forest"] = joblib.load(RF_PATH)
else:
    print("Random Forest model file not found:", RF_PATH)

for name, model in models.items():
    pred = model.predict(X_test)
    print(
        f"{name}: "
        f"Accuracy={accuracy_score(y_test, pred):.4f}, "
        f"Balanced Accuracy={balanced_accuracy_score(y_test, pred):.4f}"
    )

# --------------------------------------------------
# GRAPH 3 & 4: CPU VS GPU + REPEATED RUN LATENCIES
# --------------------------------------------------
print("\nGRAPH 3 & 4 — OPENVINO LATENCY BENCHMARK")

try:
    import openvino as ov

    ir_path = ROOT / "models" / "openvino" / "logistic_model.xml"

    if not ir_path.exists():
        print("IR model not found:", ir_path)
        print("Skipping OpenVINO latency benchmark.")
    else:
        core = ov.Core()
        model = core.read_model(str(ir_path))
        available = core.available_devices
        print("Available OpenVINO devices:", available)

        # This IR model expects 168 preprocessed numeric features.
        # A zero vector is a controlled benchmark input, not a patient record.
        
        input_size = 168
        sample = np.zeros((1, input_size), dtype=np.float32)
        sample = np.zeros((1, input_size), dtype=np.float32)

        for device in ["CPU", "GPU.0"]:
            if device not in available:
                print(f"{device} is unavailable; skipping.")
                continue

            try:
                compiled = core.compile_model(model, device)
                infer = compiled.create_infer_request()

                # Warm-up runs (not included in timing)
                for _ in range(50):
                    infer.infer({compiled.input(0): sample})

                run_ms = []
                repetitions = 50
                inferences_per_run = 1000

                for run in range(1, repetitions + 1):
                    start = time.perf_counter()

                    for _ in range(inferences_per_run):
                        infer.infer({compiled.input(0): sample})

                    elapsed_ms = (time.perf_counter() - start) * 1000
                    avg_ms = elapsed_ms / inferences_per_run
                    run_ms.append(avg_ms)

                    print(
                        f"{device}, run {run}: "
                        f"{avg_ms:.6f} ms per inference"
                    )

                print(
                    f"{device} summary: "
                    f"mean={np.mean(run_ms):.6f} ms, "
                    f"median={np.median(run_ms):.6f} ms, "
                    f"min={np.min(run_ms):.6f} ms, "
                    f"max={np.max(run_ms):.6f} ms, "
                    f"std={np.std(run_ms, ddof=1):.6f} ms"
                )

            except Exception as exc:
                print(f"Could not benchmark {device}: {exc}")

except ImportError:
    print("OpenVINO is not installed in this Python environment.")

# --------------------------------------------------
# GRAPH 5: HOSPITAL RESOURCE AVAILABILITY
# --------------------------------------------------
print("\nGRAPH 5 — HOSPITAL RESOURCE AVAILABILITY")
print("Use the CURRENT inventory values from your project.")

inventory_file = ROOT / "emergency_resource_allocation" / "resource_inventory.py"

if inventory_file.exists():
    print("Inventory module:", inventory_file)
    print("Open this file and check its current total, occupied, and available values.")
else:
    print("Inventory module not found at:", inventory_file)

