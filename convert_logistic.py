from pathlib import Path
import joblib
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

print("Loading trained Logistic Regression pipeline...")

model = joblib.load("models/logistic_model.joblib")
classifier = model.named_steps["classifier"]

# Read the fitted Logistic Regression parameters
weights = classifier.coef_.T.astype(np.float32)
bias = classifier.intercept_.astype(np.float32)
classes = classifier.classes_.astype(np.int64)

n_features, n_classes = weights.shape

print("Number of input features:", n_features)
print("Classes:", classes)

# ONNX input: already-preprocessed numeric features
X = helper.make_tensor_value_info(
    "features", TensorProto.FLOAT, [None, n_features]
)

label_output = helper.make_tensor_value_info(
    "label", TensorProto.INT64, [None]
)

prob_output = helper.make_tensor_value_info(
    "probabilities", TensorProto.FLOAT, [None, n_classes]
)

# Store trained parameters as ONNX initializers
weights_init = numpy_helper.from_array(weights, name="weights")
bias_init = numpy_helper.from_array(bias, name="bias")
classes_init = numpy_helper.from_array(classes, name="classes")

nodes = [
    helper.make_node(
        "MatMul", ["features", "weights"], ["scores"]
    ),
    helper.make_node(
        "Add", ["scores", "bias"], ["logits"]
    ),
    helper.make_node(
        "Softmax", ["logits"], ["probabilities"], axis=1
    ),
    helper.make_node(
        "ArgMax",
        ["probabilities"],
        ["class_index"],
        axis=1,
        keepdims=0
    ),
    helper.make_node(
        "Gather",
        ["classes", "class_index"],
        ["label"],
        axis=0
    ),
]

graph = helper.make_graph(
    nodes,
    "LogisticRegression",
    [X],
    [label_output, prob_output],
    initializer=[weights_init, bias_init, classes_init]
)

onnx_model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid("", 17)],
    producer_name="CAO_Project"
)

# Ensure compatibility with the installed ONNX/OpenVINO stack
onnx_model.ir_version = 10
onnx.checker.check_model(onnx_model)

output_dir = Path("models/openvino")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "logistic_model.onnx"
onnx.save_model(onnx_model, str(output_path))

print("\nONNX conversion successful!")
print("Saved to:", output_path)
print("Input:", X.name)
print("Outputs:", [label_output.name, prob_output.name])