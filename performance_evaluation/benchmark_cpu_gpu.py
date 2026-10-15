
import time
import numpy as np
import openvino as ov
import statistics

core = ov.Core()
model = core.read_model("models/openvino/logistic_model.onnx")
x = np.zeros((1, 168), dtype=np.float32)

devices = ["CPU", "GPU.0"]
num_repeats = 10
warmup_runs = 50
timed_iterations = 1000

all_results = {}

print("\nRepeated CPU vs GPU Benchmark")
print("=" * 55)
print(f"Benchmark repetitions: {num_repeats}")
print(f"Timed inferences per repetition: {timed_iterations}")

for device in devices:
    print(f"\nTesting {device}...")

    compiled_model = core.compile_model(model, device)
    infer_request = compiled_model.create_infer_request()
    input_layer = compiled_model.input(0)

    # Warm-up before collecting measurements
    for _ in range(warmup_runs):
        infer_request.infer({input_layer: x})

    run_latencies = []

    for run in range(1, num_repeats + 1):
        start = time.perf_counter()

        for _ in range(timed_iterations):
            infer_request.infer({input_layer: x})

        elapsed = time.perf_counter() - start
        avg_ms = elapsed * 1000 / timed_iterations
        run_latencies.append(avg_ms)

        print(f"Run {run:02d}: {avg_ms:.6f} ms per inference")

    all_results[device] = run_latencies

print("\n" + "=" * 55)
print("SUMMARY")
print("-" * 55)

for device, values in all_results.items():
    print(f"\n{device}")
    print(f"Mean latency:   {statistics.mean(values):.6f} ms")
    print(f"Median latency: {statistics.median(values):.6f} ms")
    print(f"Minimum:        {min(values):.6f} ms")
    print(f"Maximum:        {max(values):.6f} ms")
    print(f"Std deviation:  {statistics.stdev(values):.6f} ms")

cpu_mean = statistics.mean(all_results["CPU"])
gpu_mean = statistics.mean(all_results["GPU.0"])

print("\n" + "-" * 55)
if cpu_mean < gpu_mean:
    print(f"CPU was {gpu_mean / cpu_mean:.2f}x faster on average.")
else:
    print(f"GPU was {cpu_mean / gpu_mean:.2f}x faster on average.")

print("\nBenchmark completed successfully.")
