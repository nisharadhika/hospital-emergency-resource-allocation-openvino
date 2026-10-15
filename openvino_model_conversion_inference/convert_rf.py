
from pathlib import Path
import joblib
import pandas as pd
import onnx

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

model = joblib.load("models/baseline_model.joblib")
classifier = model.named_steps["classifier"]
preprocessor = model.named_steps["preprocessor"]

df = pd.read_csv("processed/visits_with_triage.csv")
df = df.dropna(subset=["acuity"])

features = [
    "gender", "race", "arrival_transport",
    "temperature", "heartrate", "resprate",
    "o2sat", "sbp", "dbp", "pain", "chiefcomplaint"
]

sample = preprocessor.transform(df[features].head(1))
feature_count = sample.shape[1]

print("Preprocessed feature count:", feature_count)
print("Converting Random Forest with tensor probabilities...")

onnx_model = convert_sklearn(
    classifier,
    initial_types=[
        ("features", FloatTensorType([None, feature_count]))
    ],
    target_opset=17,
    options={id(classifier): {"zipmap": False}}
)

output_dir = Path("models/openvino")
output_dir.mkdir(parents=True, exist_ok=True)
onnx_path = output_dir / "random_forest.onnx"

onnx.save_model(onnx_model, str(onnx_path))
onnx.checker.check_model(str(onnx_path))

print("ONNX conversion and validation successful!")
print("Saved to:", onnx_path)
print("Outputs:", [out.name for out in onnx_model.graph.output])
