from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "graphs_for_report"
OUTPUT_DIR.mkdir(exist_ok=True)

# Graph 1: Confusion matrix
labels = ["Acuity 1", "Acuity 2", "Acuity 3", "Acuity 4"]
cm = np.array([[2, 3, 0, 0], [6, 9, 8, 1], [1, 5, 16, 1], [0, 0, 0, 0]])
fig, ax = plt.subplots(figsize=(7.5, 6))
im = ax.imshow(cm, interpolation="nearest")
fig.colorbar(im, ax=ax, label="Number of patients")
ax.set(xticks=np.arange(4), yticks=np.arange(4), xticklabels=labels, yticklabels=labels,
       xlabel="Predicted acuity", ylabel="Actual acuity",
       title="Confusion Matrix — Logistic Regression")
plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
for i in range(4):
    for j in range(4):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black")
fig.text(0.5, 0.01, "No acuity-4 cases occurred in the 52-record test set.", ha="center", fontsize=9)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(OUTPUT_DIR / "graph1_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Graph 2: Model performance comparison
models = ["Logistic Regression", "Random Forest"]
accuracy = [51.92, 46.15]
balanced = [49.02, 39.34]
x = np.arange(2)
width = 0.34
fig, ax = plt.subplots(figsize=(8, 5.5))
b1 = ax.bar(x - width/2, accuracy, width, label="Accuracy (%)")
b2 = ax.bar(x + width/2, balanced, width, label="Balanced accuracy (%)")
ax.set_ylabel("Score (%)")
ax.set_title("Model Performance Comparison")
ax.set_xticks(x, models)
ax.set_ylim(0, 65)
ax.legend()
ax.bar_label(b1, fmt="%.2f", padding=3)
ax.bar_label(b2, fmt="%.2f", padding=3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "graph2_model_performance.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Graph 3 and 4: 50 repeated OpenVINO runs, milliseconds per inference
cpu = [
0.135590,0.102424,0.123138,0.118286,0.130035,0.127581,0.152656,0.126142,0.111149,0.093571,
0.102808,0.114871,0.106589,0.127604,0.094933,0.099645,0.100088,0.097993,0.102479,0.094357,
0.113811,0.098856,0.093883,0.180533,0.112128,0.131116,0.134837,0.133729,0.096336,0.183180,
0.103091,0.107955,0.122989,0.135067,0.116939,0.097715,0.096833,0.104081,0.099510,0.093423,
0.102241,0.102814,0.100502,0.115252,0.095227,0.095295,0.098721,0.109768,0.093953,0.098287
]
gpu = [
0.323782,0.496067,0.340682,0.309153,0.326254,0.338924,0.312585,0.289527,0.399897,0.302591,
0.299736,0.300115,0.301326,0.409092,0.352282,0.351309,0.326525,0.309827,0.328024,0.302128,
0.339618,0.365412,0.400428,0.409863,0.347458,0.372622,0.407136,0.497645,0.345421,0.339861,
0.357821,0.505128,0.385277,0.359304,0.464541,0.448755,0.343461,0.370288,0.409777,0.382642,
0.371701,0.364069,0.350856,0.345134,0.326294,0.346330,0.416344,0.491409,0.458475,0.381795
]
devices = ["CPU", "GPU.0"]
means = [np.mean(cpu), np.mean(gpu)]
stds = [np.std(cpu, ddof=1), np.std(gpu, ddof=1)]
fig, ax = plt.subplots(figsize=(7, 5.5))
bars = ax.bar(devices, means, yerr=stds, capsize=7)
ax.set_ylabel("Latency (ms per inference)")
ax.set_title("OpenVINO CPU vs GPU.0 Inference Latency")
ax.bar_label(bars, labels=[f"{v:.4f} ms" for v in means], padding=5)
ax.text(0.5, -0.18, "50 runs × 1,000 inferences per run; error bars show standard deviation.",
        transform=ax.transAxes, ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "graph3_cpu_vs_gpu_latency.png", dpi=300, bbox_inches="tight")
plt.close(fig)

runs = np.arange(1, 51)
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(runs, cpu, marker="o", markersize=3, linewidth=1.3, label="CPU")
ax.plot(runs, gpu, marker="o", markersize=3, linewidth=1.3, label="GPU.0")
ax.set_xlabel("Benchmark run")
ax.set_ylabel("Average latency (ms per inference)")
ax.set_title("OpenVINO Latency Across 50 Repeated Benchmark Runs")
ax.set_xticks(np.arange(0, 51, 5))
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "graph4_repeated_run_latency.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Graph 5: use current values from resource_inventory.py
print("\nGRAPH 5 — Enter CURRENT inventory values from resource_inventory.py.")
print("For each resource, enter total, occupied, and available as shown in your file.")
names = ["Emergency beds", "Medical staff", "Patient monitors", "Oxygen units"]
totals, occupied, available = [], [], []
for name in names:
    print(f"\n{name}")
    total = int(input("  Total: "))
    occ = int(input("  Occupied: "))
    avail = int(input("  Available: "))
    if min(total, occ, avail) < 0:
        raise ValueError("Inventory values cannot be negative.")
    if occ + avail != total:
        print("WARNING: occupied + available does not equal total; check these values.")
    totals.append(total)
    occupied.append(occ)
    available.append(avail)

x = np.arange(len(names))
width = 0.25
fig, ax = plt.subplots(figsize=(10, 5.5))
b1 = ax.bar(x-width, totals, width, label="Total")
b2 = ax.bar(x, occupied, width, label="Occupied")
b3 = ax.bar(x+width, available, width, label="Available")
ax.set_ylabel("Number of units / staff")
ax.set_title("Hospital Emergency Resource Availability")
ax.set_xticks(x, names)
ax.legend()
ax.bar_label(b1, padding=2)
ax.bar_label(b2, padding=2)
ax.bar_label(b3, padding=2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "graph5_resource_availability.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nAll five graphs saved in:", OUTPUT_DIR)
for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(" -", file.name)
