

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import openvino as ov

from resource_inventory import resource_inventory
from allocation_algorithm import allocate_resources

st.set_page_config(
    page_title="Emergency Resource Allocation",
    page_icon="🏥",
    layout="wide"
)

FEATURE_COLUMNS = [
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
    "chiefcomplaint"
]


@st.cache_resource
def load_models():
    """Load the trained pipeline and compile the ONNX model once."""
    pipeline = joblib.load("models/logistic_model.joblib")

    core = ov.Core()
    model = core.read_model("models/openvino/logistic_model.onnx")
    compiled_model = core.compile_model(model, "CPU")

    return pipeline, compiled_model


def predict_acuity(patient_features):
    """Preprocess one patient and predict acuity with OpenVINO."""
    pipeline, compiled_model = load_models()

    # Preserve the feature names and order used during training.
    patient_df = pd.DataFrame(
        [patient_features],
        columns=FEATURE_COLUMNS
    )

    preprocessor = pipeline.named_steps["preprocessor"]
    processed = preprocessor.transform(patient_df)

    # OpenVINO expects a dense float32 matrix for this ONNX model.
    if hasattr(processed, "toarray"):
        processed = processed.toarray()

    processed = np.asarray(processed, dtype=np.float32)

    if processed.shape != (1, 168):
        raise ValueError(
            f"Expected input shape (1, 168), got {processed.shape}"
        )

    results = compiled_model({"features": processed})

    predicted_class = int(results[compiled_model.output("label")][0])
    probabilities = results[
        compiled_model.output("probabilities")
    ][0]

    classes = list(pipeline.classes_)

    probability_table = {
        int(class_label): float(probability)
        for class_label, probability in zip(classes, probabilities)
    }

    return predicted_class, probability_table


st.title("🏥 Hospital Emergency Resource Allocation System")
st.caption(
    "Academic prototype | OpenVINO inference | Simulated inventory | "
    "Human review required"
)

if "assignments" not in st.session_state:
    st.session_state.assignments = []

if "next_patient" not in st.session_state:
    st.session_state.next_patient = 1

if "latest_proposal" not in st.session_state:
    st.session_state.latest_proposal = None

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None


def current_status():
    """
    Use the inventory module's current values directly.
    allocate_resources(confirm=True) updates occupied counts itself.
    Do not add assignment counts again, as that would double-count.
    """
    result = {}

    for name, resource in resource_inventory.items():
        total = resource["total"]
        occupied = resource["occupied"]

        result[name] = {
            "total": total,
            "occupied": occupied,
            "available": max(0, total - occupied)
        }

    return result


labels = {
    "emergency_beds": "🛏️ Emergency Beds",
    "medical_staff": "👩‍⚕️ Medical Staff Units",
    "patient_monitors": "❤️ Patient Monitors",
    "oxygen_units": "🫁 Oxygen Units"
}


# 1. Inventory
st.header("1. Current Resource Inventory")

status = current_status()
columns = st.columns(4)

for column, (name, item) in zip(columns, status.items()):
    with column:
        st.metric(
            labels.get(name, name),
            f"{item['available']} available",
            f"{item['occupied']} occupied / {item['total']} total"
        )

st.divider()


# 2. Patient input and OpenVINO inference
st.header("2. Patient Assessment")

st.info(
    "Enter a patient record to run the trained model. "
    "Use the same units and category values as the training dataset."
)

with st.form("patient_form"):
    patient_id = st.text_input(
        "Patient ID",
        value=f"P{st.session_state.next_patient:03d}"
    )

    st.subheader("Vital Signs")

    c1, c2, c3 = st.columns(3)

    with c1:
        temperature = st.number_input(
            "Temperature (°C)",
            min_value=25.0,
            max_value=45.0,
            value=37.0,
            step=0.1
        )
        heartrate = st.number_input(
            "Heart rate (beats/min)",
            min_value=0.0,
            max_value=300.0,
            value=80.0
        )

    with c2:
        resprate = st.number_input(
            "Respiratory rate (breaths/min)",
            min_value=0.0,
            max_value=100.0,
            value=18.0
        )
        o2sat = st.number_input(
            "Oxygen saturation (%)",
            min_value=0.0,
            max_value=100.0,
            value=98.0
        )

    with c3:
        sbp = st.number_input(
            "Systolic blood pressure (mmHg)",
            min_value=0.0,
            max_value=350.0,
            value=120.0
        )
        dbp = st.number_input(
            "Diastolic blood pressure (mmHg)",
            min_value=0.0,
            max_value=250.0,
            value=80.0
        )

    st.subheader("Patient Information")

    gender = st.text_input("Gender category", value="Female")
    race = st.text_input("Race category", value="Unknown")
    arrival_transport = st.text_input(
        "Arrival transport category",
        value="WALK IN"
    )
    pain = st.text_input("Pain category", value="Unknown")
    chiefcomplaint = st.text_input(
        "Chief complaint",
        value="General complaint"
    )

    submitted = st.form_submit_button("Predict Acuity with OpenVINO")


if submitted:
    if not patient_id.strip():
        st.error("Enter a patient ID.")
    elif any(
        a["patient_id"] == patient_id.strip()
        for a in st.session_state.assignments
    ):
        st.error("This patient ID already exists in the allocation records.")
    else:
        patient_features = {
            "temperature": temperature,
            "heartrate": heartrate,
            "resprate": resprate,
            "o2sat": o2sat,
            "sbp": sbp,
            "dbp": dbp,
            "gender": gender.strip(),
            "race": race.strip(),
            "arrival_transport": arrival_transport.strip(),
            "pain": pain.strip(),
            "chiefcomplaint": chiefcomplaint.strip()
        }

        try:
            prediction, probabilities = predict_acuity(patient_features)

            patient = {
                "patient_id": patient_id.strip(),
                "acuity": prediction
            }

            proposal = allocate_resources(patient, confirm=False)

            st.session_state.latest_prediction = {
                "patient_id": patient_id.strip(),
                "acuity": prediction,
                "probabilities": probabilities
            }
            st.session_state.latest_proposal = proposal
            st.session_state.next_patient += 1

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# 3. Display model prediction
prediction_info = st.session_state.latest_prediction

if prediction_info:
    st.divider()
    st.header("3. OpenVINO Prediction")

    st.write("Patient ID:", prediction_info["patient_id"])
    st.metric("Predicted acuity class", prediction_info["acuity"])

    st.write("Model class probabilities:")

    probability_data = pd.DataFrame({
        "Acuity class": [
            f"Class {class_id}"
            for class_id in prediction_info["probabilities"]
        ],
        "Probability (%)": [
            round(probability * 100, 2)
            for probability in prediction_info["probabilities"].values()
        ]
    })

    st.dataframe(probability_data, hide_index=True, use_container_width=True)

    st.caption(
        "Model probabilities are not guarantees of correctness. "
        "This research prototype is not validated for clinical triage."
    )


# 4. Resource proposal and human confirmation
proposal = st.session_state.latest_proposal

if proposal:
    st.divider()
    st.header("4. Resource Allocation Proposal")

    st.write("Patient ID:", proposal["patient_id"])
    st.write("Acuity class:", proposal.get("acuity", "Unknown"))
    st.write("Proposal status:", proposal["status"])

    if proposal.get("shortages"):
        st.error(
            "Unavailable resources: "
            + ", ".join(proposal["shortages"])
        )
        st.warning(
            "Refer the proposal for staff review. "
            "Do not allocate unavailable resources."
        )

    elif proposal.get("proposed_resources"):
        st.write("Proposed resource quantities:")
        st.json(proposal["proposed_resources"])
        st.info(proposal["message"])

        if st.button("Confirm simulated allocation"):
            # Recheck current inventory before confirming.
            result = allocate_resources(
                {
                    "patient_id": proposal["patient_id"],
                    "acuity": proposal["acuity"]
                },
                confirm=True
            )

            if result["status"] == "SIMULATED ALLOCATION CONFIRMED":
                st.session_state.assignments.append(result)
                st.session_state.latest_proposal = None
                st.session_state.latest_prediction = None
                st.success("Simulated allocation confirmed.")
                st.rerun()
            else:
                st.error(result["message"])


# 5. Allocation records
st.divider()
st.header("5. Patient Allocation Records")

if st.session_state.assignments:
    records = [
        {
            "Patient ID": assignment["patient_id"],
            "Acuity": assignment["acuity"],
            "Status": assignment["status"],
            "Resources": str(
                assignment.get("allocated_resources", {})
            )
        }
        for assignment in st.session_state.assignments
    ]

    st.dataframe(records, use_container_width=True)
else:
    st.info("No simulated allocations have been confirmed yet.")


# 6. Remaining inventory
st.header("6. Remaining Resources")

for name, item in current_status().items():
    st.write(
        f"**{labels.get(name, name)}:** "
        f"{item['available']} available out of {item['total']}"
    )

    if item["total"] > 0:
        st.progress(item["available"] / item["total"])

st.caption(
    "Academic demonstration only. Resource inventory and allocation "
    "rules are simulated. Predictions and allocation proposals must not "
    "be used to make real clinical decisions."
)
