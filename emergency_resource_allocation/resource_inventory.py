
# resource_inventory.py
# Simulated hospital resource inventory for the CAO project.
# These are demonstration values, not real hospital availability.

resource_inventory = {
    "emergency_beds": {
        "total": 20,
        "occupied": 12
    },
    "medical_staff": {
        "total": 10,
        "occupied": 4
    },
    "patient_monitors": {
        "total": 8,
        "occupied": 4
    },
    "oxygen_units": {
        "total": 7,
        "occupied": 2
    }
}


def get_resource_status():
    """Calculate available resources from total and occupied quantities."""
    status = {}

    for name, resource in resource_inventory.items():
        total = resource["total"]
        occupied = resource["occupied"]

        status[name] = {
            "total": total,
            "occupied": occupied,
            "available": total - occupied
        }

    return status


def display_inventory():
    print("\n=== HOSPITAL RESOURCE INVENTORY ===")

    status = get_resource_status()

    for name, resource in status.items():
        print(f"\nResource: {name.replace('_', ' ').title()}")
        print(f"Total:     {resource['total']}")
        print(f"Occupied:  {resource['occupied']}")
        print(f"Available: {resource['available']}")


if __name__ == "__main__":
    display_inventory()
