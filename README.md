# Hospital Emergency Resource Allocation System Using OpenVINO: A Machine Learning-Based Approach to Inference Optimization and Resource Planning 
## Overview

The **Hospital Emergency Resource Allocation System Using OpenVINO** is an academic project that explores machine learning-based inference and resource planning for hospital emergency departments.

The system combines patient-related input data, machine learning inference, and rule-based resource allocation to generate proposed emergency resource assignments. OpenVINO is used to support optimized model inference, while performance evaluation scripts help analyze the inference workflow.

This project demonstrates how machine learning and inference optimization can be integrated into a resource-planning prototype.

## Objectives

- Develop a machine learning-based approach for hospital emergency resource planning.
- Process and prepare the input datasets for model training and inference.
- Convert the trained model into an OpenVINO-compatible Intermediate Representation (IR), where applicable.
- Perform inference using OpenVINO.
- Generate proposed resource allocations based on the system's implemented logic.
- Evaluate inference performance and visualize experimental results.

## System Workflow

1. **Data Collection and Preprocessing:** Prepare datasets and process the input features.
2. **Machine Learning Model Training:** Train the model using the project's training pipeline.
3. **Model Conversion:** Convert the trained model into a format compatible with OpenVINO.
4. **OpenVINO Inference:** Load the converted model and perform inference on input data.
5. **Resource Allocation:** Apply the implemented allocation logic to generate proposed resource assignments.
6. **Performance Evaluation:** Analyze inference performance and generate graphs for reporting.
7. **Dashboard and Results:** Present relevant inputs, outputs, or inference results through the project's dashboard components.

## Technologies Used

- **Python** — implementation and data processing
- **OpenVINO** — machine learning model inference and optimization
- **Machine Learning** — prediction using the project's trained model
- **Pandas and NumPy** — data handling and numerical operations, where used
- **Matplotlib** — result visualization, where used
- **Streamlit or other dashboard tools** — use the framework specified in the dashboard implementation

## Repository Structure

```text
CAO_proj/
├── Data Collection and Preprocessing/
├── Machine Learning Model Training/
├── dashboard_user_inference/
├── emergency_resource_allocation/
├── models/
├── openvino_model_conversion_inference/
├── performance_evaluation/
├── graphs_for_report/
├── images/
├── docs/
├── .gitignore
└── README.md
```

*Note: The structure above highlights the main project components. Refer to the repository for the complete list of files.*

## Model and OpenVINO

The `models/` directory contains model artifacts used in the project, including serialized model files and model formats for inference.

OpenVINO Intermediate Representation (IR) models typically use:
- **`.xml`** — model architecture and layer information.
- **`.bin`** — model weights.

The conversion and inference scripts are located in `openvino_model_conversion_inference/`. Refer to those scripts to identify the exact conversion procedure and model used.

## Performance Evaluation

The `performance_evaluation/` directory contains the project's performance evaluation components. The generated graphs are available in `graphs_for_report/` and `images/`.

These results can be used to examine the performance of the implemented inference workflow. Actual performance conclusions should be drawn from the measured results and experimental setup documented in the project.

## Documentation

The `docs/` directory contains the project's report and paper-related documents. Please refer to these files for further details about the methodology, implementation, and experimental results.

## How to Run

1. Clone or download this repository.
2. Install the dependencies required by the project.
3. Open the project folder in VS Code.
4. Follow the instructions and file paths in the relevant preprocessing, training, conversion, inference, and dashboard scripts.
5. Run the appropriate entry-point script for the component you want to test.

**Dependency installation and execution commands:** Add the verified commands from your project environment or a `requirements.txt` file. These depend on the libraries and entry-point scripts used in your implementation.

## Limitations and Responsible Use

This is an **academic prototype**, not a clinically validated hospital decision-support system. Its proposed resource allocations must not be used as a substitute for clinical judgment, validated hospital protocols, or decisions made by qualified healthcare professionals.

The quality of its predictions and allocations depends on the datasets, trained model, implemented rules, and evaluation methodology. Further validation would be required before any real-world clinical deployment.

## Future Enhancements

- Evaluate the system using additional datasets and validation procedures.
- Improve model evaluation and inference benchmarking.
- Extend resource availability tracking and allocation logic.
- Improve dashboard usability and result visualization.
- Investigate deployment and integration requirements for real-world environments.




**Disclaimer:** This project is intended for educational and research purposes only.
