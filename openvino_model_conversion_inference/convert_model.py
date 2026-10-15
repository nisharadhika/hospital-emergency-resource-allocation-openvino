
from pathlib import Path
import copy
import joblib
import onnx

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

model_path = Path("models/baseline_model.joblib")
output_dir = Path("models/openvino")
output_dir.mkdir(parents=True, exist_ok=True)

# Load and preserve the original trained model
original_model = joblib.load(model_path)
export_model = copy.deepcopy(original_model)

# Save the fitted imputation values for Python-side preprocessing
numeric_pipe = original_model.named_steps["preprocessor"].named_transformers_["numeric"]
categorical_pipe = original_model.named_steps["preprocessor"].named_transformers_["categorical"]

imputation_values = {
    "numeric_medians": dict(zip(
        numeric_pipe.named_steps["imputer"].feature_names_in_,
        numeric_pipe.named_steps["imputer"].statistics_
    )),
    "categorical_modes": dict(zip(
        categorical_pipe.named_steps["imputer"].feature_names_in_,
        categorical_pipe.named_steps["imputer"].statistics_
    ))
}
joblib.dump(imputation_values, output_dir / "imputation_values.joblib")

# Remove only the imputer stages from the export copy.
# Missing values will be filled in Python before inference.
export_model.set_params(
    preprocessor__numeric__imputer="passthrough",
    preprocessor__categorical__imputer="passthrough"
)

numeric_features = [
    "temperature", "heartrate", "resprate",
    "o2sat", "sbp", "dbp"
]
categorical_features = [
    "gender", "race", "arrival_transport",
    "pain", "chiefcomplaint"
]

initial_types = [
    (name, FloatTensorType([None, 1]))
    for name in numeric_features
]
initial_types += [
    (name, StringTensorType([None, 1]))
    for name in categorical_features
]

print("Converting pipeline without imputer stages...")

onnx_model = convert_sklearn(
    export_model,
    initial_types=initial_types,
    target_opset=17
)

onnx_path = output_dir / "hospital_resource_model.onnx"
onnx.save_model(onnx_model, str(onnx_path))
onnx.checker.check_model(str(onnx_path))

print("ONNX conversion and validation successful!")
print("ONNX model:", onnx_path)
print("Imputation values saved.")
