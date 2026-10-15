
import openvino as ov

model_path = "models/openvino/logistic_model.onnx"

core = ov.Core()
model = core.read_model(model_path)

print("Model loaded successfully!")
print("\nInputs:")

for item in model.inputs:
    print("Name:", item.get_any_name())
    print("Shape:", item.get_partial_shape())
    print("Type:", item.get_element_type())

print("\nOutputs:")

for item in model.outputs:
    print("Name:", item.get_any_name())
    print("Shape:", item.get_partial_shape())
    print("Type:", item.get_element_type())

print("\nAvailable devices:", core.available_devices)
