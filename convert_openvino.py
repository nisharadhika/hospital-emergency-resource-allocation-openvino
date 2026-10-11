
from pathlib import Path
import openvino as ov

onnx_path = Path("models/openvino/random_forest.onnx")
output_dir = Path("models/openvino")
output_dir.mkdir(parents=True, exist_ok=True)

print("Reading ONNX model...")
core = ov.Core()
model = core.read_model(str(onnx_path))

print("Converting model to OpenVINO IR...")
ov.save_model(
    model,
    str(output_dir / "random_forest.xml")
)

print("\nOpenVINO conversion successful!")
print("XML file:", output_dir / "random_forest.xml")
print("BIN file:", output_dir / "random_forest.bin")
print("Available devices:", core.available_devices)
