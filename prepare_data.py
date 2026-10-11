
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load visit-level tables
visits = pd.read_csv(DATA_DIR / "edstays.csv")
triage = pd.read_csv(DATA_DIR / "triage.csv")

# Each visit should have one row in both tables
assert visits["stay_id"].is_unique, "Duplicate stay IDs in edstays"
assert triage["stay_id"].is_unique, "Duplicate stay IDs in triage"

# Keep the target separate from the input features
df = visits.merge(
    triage,
    on=["subject_id", "stay_id"],
    how="left",
    validate="one_to_one",
    suffixes=("_visit", "_triage"),
)

print("Merged visit rows:", len(df))
print("Unique visit IDs:", df["stay_id"].nunique())
print("\nTriage acuity distribution:")
print(df["acuity"].value_counts(dropna=False).sort_index())

print("\nDisposition distribution:")
print(df["disposition"].value_counts(dropna=False))

# Save the joined dataset for inspection, not model training yet
output_path = OUTPUT_DIR / "visits_with_triage.csv"
df.to_csv(output_path, index=False)

print(f"\nSaved inspection dataset to: {output_path}")
print("Input CSV files were not modified.")
