import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results/figures", exist_ok=True)

data = pd.DataFrame({
    "Configuration": [
        "CPU FP32",
        "GPU FP32",
        "GPU FP16",
        "TensorRT FP16"
    ],
    "Latency (ms)": [
        44.12,
        20.06,
        22.88,
        7.47
    ],
    "Throughput (FPS)": [
        22.67,
        49.86,
        43.71,
        133.88
    ],
    "Inference (ms)": [
        38.50,
        13.02,
        15.26,
        2.51
    ],
    "P95 Latency (ms)": [
        51.16,
        24.54,
        29.24,
        9.06
    ]
})


def save_bar(column, ylabel, title, filename):
    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        data["Configuration"],
        data[column]
    )

    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, data[column]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()

    plt.savefig(
        f"results/figures/{filename}",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


save_bar(
    "Throughput (FPS)",
    "Frames per Second",
    "Multi-Object Tracking Throughput",
    "throughput_comparison.png"
)

save_bar(
    "Latency (ms)",
    "Latency (ms)",
    "Average Tracking Pipeline Latency",
    "latency_comparison.png"
)

save_bar(
    "Inference (ms)",
    "Inference Time (ms)",
    "Object Detection Inference Time",
    "inference_comparison.png"
)

save_bar(
    "P95 Latency (ms)",
    "P95 Latency (ms)",
    "P95 Tracking Pipeline Latency",
    "p95_latency_comparison.png"
)

data.to_csv(
    "results/benchmark_final.csv",
    index=False
)

print("Charts generated successfully.")
print("\nSaved:")
print("results/figures/throughput_comparison.png")
print("results/figures/latency_comparison.png")
print("results/figures/inference_comparison.png")
print("results/figures/p95_latency_comparison.png")
print("results/benchmark_final.csv")