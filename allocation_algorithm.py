
# allocation_algorithm.py

from resource_inventory import resource_inventory, get_resource_status


def allocate_resources(patient, confirm=False):
    """
    Generate a proposed resource allocation for a patient.

    This is a simulated academic prototype.
    Acuity alone does not determine real clinical resource needs.
    Assignments require appropriate clinical review.
    """

    acuity = patient["acuity"]

    # Initial proposal rules for demonstration only.
    # These are NOT validated clinical protocols.
    if acuity == 1:
        required = {
            "emergency_beds": 1,
            "medical_staff": 1,
            "patient_monitors": 1,
            "oxygen_units": 1
        }
    elif acuity == 2:
        required = {
            "emergency_beds": 1,
            "medical_staff": 1,
            "patient_monitors": 1,
            "oxygen_units": 0
        }
    elif acuity == 3:
        required = {
            "emergency_beds": 1,
            "medical_staff": 1,
            "patient_monitors": 0,
            "oxygen_units": 0
        }
    elif acuity == 4:
        required = {
            "emergency_beds": 0,
            "medical_staff": 1,
            "patient_monitors": 0,
            "oxygen_units": 0
        }
    else:
        return {
            "patient_id": patient["patient_id"],
            "status": "REJECTED",
            "message": "Invalid acuity class."
        }

    status = get_resource_status()
    shortages = []

    for resource, quantity in required.items():
        if status[resource]["available"] < quantity:
            shortages.append(resource)

    if shortages:
        return {
            "patient_id": patient["patient_id"],
            "acuity": acuity,
            "status": "RESOURCE SHORTAGE",
            "proposed_resources": required,
            "shortages": shortages,
            "message": "Do not assign unavailable resources. Escalate for staff review."
        }

    if not confirm:
        return {
            "patient_id": patient["patient_id"],
            "acuity": acuity,
            "status": "PROPOSED — REVIEW REQUIRED",
            "proposed_resources": required,
            "shortages": [],
            "message": "Proposal only. No resources have been reserved."
        }

    # Reserve the resources in the simulated inventory.
    for resource, quantity in required.items():
        resource_inventory[resource]["occupied"] += quantity

    return {
        "patient_id": patient["patient_id"],
        "acuity": acuity,
        "status": "SIMULATED ALLOCATION CONFIRMED",
        "allocated_resources": required,
        "shortages": [],
        "message": "Simulation updated. Clinical authorization is not implied."
    }


if __name__ == "__main__":
    # Example patient for testing; not a real patient.
    demo_patient = {
        "patient_id": "P001",
        "acuity": 2
    }

    print("\n=== RESOURCE ALLOCATION TEST ===")
    result = allocate_resources(demo_patient)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== INVENTORY AFTER PROPOSAL ===")
    for name, item in get_resource_status().items():
        print(
            f"{name}: {item['available']} available "
            f"out of {item['total']}"
        )
