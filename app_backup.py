
# app.py
# Hospital Emergency Resource Allocation Dashboard
# Academic prototype using simulated hospital inventory.

import streamlit as st

from resource_inventory import resource_inventory, get_resource_status
from allocation_algorithm import allocate_resources

st.set_page_config(
    page_title="Emergency Resource Allocation",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Hospital Emergency Resource Allocation System")
st.caption(
    "Academic prototype | Simulated inventory | "
    "All proposed allocations require clinical review"
)

# Keep simulated assignments in the dashboard session.
if "assignments" not in st.session_state:
    st.session_state.assignments = []

if "next_patient" not in st.session_state:
    st.session_state.next_patient = 1


def get_confirmed_counts():
    counts = {
        name: 0 for name in resource_inventory
    }

    for assignment in st.session_state.assignments:
        if assignment["status"] == "SIMULATED ALLOCATION CONFIRMED":
            for resource, quantity in assignment.get(
                "allocated_resources", {}
            ).items():
                counts[resource] += quantity

    return counts


def current_status():
    """Calculate availability without double-counting assignments."""
    counts = get_confirmed_counts()
    result = {}

    for name, resource in resource_inventory.items():
        total = resource["total"]
        occupied = resource["occupied"] + counts[name]

        result[name] = {
            "total": total,
            "occupied": occupied,
            "available": total - occupied
        }

    return result


# Resource inventory
st.header("1. Current Resource Inventory")

status = current_status()
columns = st.columns(4)

labels = {
    "emergency_beds": "🛏️ Emergency Beds",
    "medical_staff": "👩‍⚕️ Medical Staff Units",
    "patient_monitors": "❤️ Patient Monitors",
    "oxygen_units": "🫁 Oxygen Units"
}

for column, (name, item) in zip(columns, status.items()):
    with column:
        st.metric(
            labels[name],
            f"{item['available']} available",
            f"{item['occupied']} occupied / {item['total']} total"
        )

st.divider()

# Patient input
st.header("2. Generate a Resource Allocation Proposal")

with st.form("patient_form"):
    patient_id = st.text_input(
        "Patient ID",
        value=f"P{st.session_state.next_patient:03d}"
    )

    acuity = st.selectbox(
        "Predicted triage acuity class",
        options=[1, 2, 3, 4],
        help=(
            "Demo input for now. Connect this field to your "
            "OpenVINO model prediction in the next integration step."
        )
    )

    submitted = st.form_submit_button("Generate Proposal")

if submitted:
    if not patient_id.strip():
        st.error("Enter a patient ID.")
    elif any(
        a["patient_id"] == patient_id.strip()
        for a in st.session_state.assignments
    ):
        st.error("This patient ID already exists in this session.")
    else:
        patient = {
            "patient_id": patient_id.strip(),
            "acuity": acuity
        }

        # Proposals do not reserve any resources.
        result = allocate_resources(patient, confirm=False)
        st.session_state["latest_proposal"] = result
        st.session_state["next_patient"] += 1

proposal = st.session_state.get("latest_proposal")

if proposal:
    st.subheader("Allocation Proposal")
    st.write("Patient:", proposal["patient_id"])
    st.write("Predicted acuity:", proposal.get("acuity", "Unknown"))
    st.write("Status:", proposal["status"])

    if proposal.get("shortages"):
        st.error("Unavailable resources: " + ", ".join(proposal["shortages"]))
        st.warning("Escalate for staff review; do not over-allocate.")
    elif proposal.get("proposed_resources"):
        st.write("Proposed resource quantities:")
        st.json(proposal["proposed_resources"])
        st.info(proposal["message"])

        if st.button("Confirm simulated allocation"):
            # Recheck availability at confirmation time.
            result = allocate_resources(
                {
                    "patient_id": proposal["patient_id"],
                    "acuity": proposal["acuity"]
                },
                confirm=True
            )

            if result["status"] == "SIMULATED ALLOCATION CONFIRMED":
                st.session_state.assignments.append(result)
                st.session_state.pop("latest_proposal", None)
                st.success("Simulated allocation confirmed.")
                st.rerun()
            else:
                st.error(result["message"])

st.divider()

# Patient assignment history
st.header("3. Patient Allocation Records")

if st.session_state.assignments:
    st.dataframe(
        [
            {
                "Patient ID": a["patient_id"],
                "Acuity": a["acuity"],
                "Status": a["status"],
                "Resources": str(a.get("allocated_resources", {}))
            }
            for a in st.session_state.assignments
        ],
        use_container_width=True
    )
else:
    st.info("No simulated allocations have been confirmed yet.")

st.header("4. Remaining Resources")

for name, item in current_status().items():
    st.write(
        f"**{labels[name]}:** "
        f"{item['available']} available out of {item['total']}"
    )
    st.progress(
        item["available"] / item["total"] if item["total"] else 0
    )

st.caption(
    "Demonstration only. Inventory values and allocation rules are simulated. "
    "This dashboard is not a clinical decision-making tool."
)
