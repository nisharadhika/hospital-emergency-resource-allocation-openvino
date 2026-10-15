
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"

files = {
    name: pd.read_csv(DATA / f"{name}.csv")
    for name in [
        "edstays",
        "triage",
        "vitalsign",
        "diagnosis",
        "medrecon",
        "pyxis",
    ]
}

print("=" * 60)
print("1. TABLE SIZES AND DUPLICATE VISIT IDS")
print("=" * 60)

for name, df in files.items():
    print(f"\n{name}: {len(df)} rows")
    if "stay_id" in df.columns:
        print("Unique stay IDs:", df["stay_id"].nunique())
        print("Missing stay IDs:", df["stay_id"].isna().sum())
        print(
            "Repeated stay IDs:",
            df["stay_id"].duplicated().sum()
        )

print("\n" + "=" * 60)
print("2. MISSING VALUES")
print("=" * 60)

for name, df in files.items():
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    print(f"\n{name}:")
    if missing.empty:
        print("No missing values")
    else:
        print(missing.to_string())

print("\n" + "=" * 60)
print("3. TRIAGE ACUITY DISTRIBUTION")
print("=" * 60)

triage = files["triage"]

if "acuity" in triage.columns:
    print(triage["acuity"].value_counts(dropna=False).sort_index())
else:
    print("Acuity column not found")

print("\n" + "=" * 60)
print("4. VISIT DISPOSITION DISTRIBUTION")
print("=" * 60)

edstays = files["edstays"]

if "disposition" in edstays.columns:
    print(edstays["disposition"].value_counts(dropna=False))
else:
    print("Disposition column not found")

print("\n" + "=" * 60)
print("5. VISIT ID OVERLAP")
print("=" * 60)

base_ids = set(edstays["stay_id"].dropna())

for name in ["triage", "vitalsign", "diagnosis", "medrecon", "pyxis"]:
    ids = set(files[name]["stay_id"].dropna())
    print(
        f"edstays vs {name}:",
        f"{len(base_ids & ids)} shared stay IDs",
        f"of {len(base_ids)} visits"
    )

print("\nData inspection complete.")
