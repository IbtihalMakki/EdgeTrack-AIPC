import os
import time
import csv
import statistics

import cv2
import psutil
import torch
from ultralytics import YOLO


# ============================================================
# EdgeTrack-AIPC
# Reproducible CPU FP32 vs GPU FP32 vs GPU FP16 Benchmark
# ============================================================

MODEL_NAME = "yolo11n.pt"
VIDEO_PATH = "data/benchmark.mp4"
RESULTS_DIR = "results"

CONFIDENCE = 0.4
TRACKER = "bytetrack.yaml"

# Ignore initial frames when calculating steady-state metrics
WARMUP_FRAMES = 10

# None = process the whole video
MAX_FRAMES = None


EXPERIMENTS = [
    {
        "name": "cpu_fp32",
        "device": "cpu",
        "half": False,
    },
    {
        "name": "gpu_fp32",
        "device": 0,
        "half": False,
    },
    {
        "name": "gpu_fp16",
        "device": 0,
        "half": True,
    },
]


def get_vram_mb(device):
    """Return currently allocated VRAM."""
    if device == "cpu":
        return 0.0

    return torch.cuda.memory_allocated(device) / (1024 ** 2)


def run_experiment(config):

    name = config["name"]
    device = config["device"]
    half = config["half"]

    print("\n")
    print("=" * 65)
    print(f"Experiment : {name}")
    print(f"Device     : {device}")
    print(f"Precision  : {'FP16' if half else 'FP32'}")
    print("=" * 65)

    # --------------------------------------------------------
    # Load a fresh model for every experiment
    # --------------------------------------------------------
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    total_video_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(
        f"Video      : {width}x{height}, "
        f"{video_fps:.2f} FPS, "
        f"{total_video_frames} frames"
    )

    process = psutil.Process(os.getpid())

    if device != "cpu":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    rows = []

    frame_id = 0

    experiment_start = time.perf_counter()

    # --------------------------------------------------------
    # Process video
    # --------------------------------------------------------
    while True:

        success, frame = cap.read()

        if not success:
            break

        if MAX_FRAMES is not None and frame_id >= MAX_FRAMES:
            break

        start = time.perf_counter()

        results = model.track(
            frame,
            tracker=TRACKER,
            persist=True,
            conf=CONFIDENCE,
            verbose=False,
            device=device,
            half=half,
        )

        # CUDA operations are asynchronous.
        # Synchronize before stopping the timer.
        if device != "cpu":
            torch.cuda.synchronize(device)

        end = time.perf_counter()

        latency_ms = (end - start) * 1000.0

        instantaneous_fps = (
            1000.0 / latency_ms
            if latency_ms > 0
            else 0.0
        )

        inference_ms = (
            results[0].speed.get("inference", 0.0)
        )

        ram_mb = (
            process.memory_info().rss /
            (1024 ** 2)
        )

        vram_mb = get_vram_mb(device)

        boxes = results[0].boxes

        object_count = (
            len(boxes)
            if boxes is not None
            else 0
        )

        # Count actual track IDs
        track_count = 0

        if boxes is not None and boxes.id is not None:
            track_count = len(boxes.id)

        rows.append({
            "frame": frame_id,
            "latency_ms": latency_ms,
            "inference_ms": inference_ms,
            "fps": instantaneous_fps,
            "ram_mb": ram_mb,
            "vram_mb": vram_mb,
            "objects": object_count,
            "tracks": track_count,
        })

        # Progress indicator
        if frame_id % 50 == 0:
            print(
                f"Frame {frame_id:4d} | "
                f"Latency {latency_ms:7.2f} ms | "
                f"Objects {object_count:2d} | "
                f"Tracks {track_count:2d}"
            )

        frame_id += 1

    cap.release()

    experiment_end = time.perf_counter()

    # --------------------------------------------------------
    # Save raw CSV
    # --------------------------------------------------------
    csv_path = os.path.join(
        RESULTS_DIR,
        f"{name}.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        fieldnames = [
            "frame",
            "latency_ms",
            "inference_ms",
            "fps",
            "ram_mb",
            "vram_mb",
            "objects",
            "tracks",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # Remove warm-up frames from summary
    # --------------------------------------------------------
    measured_rows = rows[WARMUP_FRAMES:]

    if not measured_rows:
        raise RuntimeError(
            "Not enough frames after warm-up."
        )

    latencies = [
        r["latency_ms"]
        for r in measured_rows
    ]

    inference_times = [
        r["inference_ms"]
        for r in measured_rows
    ]

    ram_values = [
        r["ram_mb"]
        for r in measured_rows
    ]

    vram_values = [
        r["vram_mb"]
        for r in measured_rows
    ]

    object_counts = [
        r["objects"]
        for r in measured_rows
    ]

    track_counts = [
        r["tracks"]
        for r in measured_rows
    ]

    # Effective throughput is more meaningful than
    # averaging reciprocal per-frame FPS values.
    total_measured_seconds = (
        sum(latencies) / 1000.0
    )

    throughput_fps = (
        len(measured_rows) /
        total_measured_seconds
    )

    # p95 latency
    sorted_latencies = sorted(latencies)

    p95_index = int(
        0.95 * (len(sorted_latencies) - 1)
    )

    p95_latency = sorted_latencies[p95_index]

    summary = {
        "experiment": name,
        "device": str(device),
        "precision": "FP16" if half else "FP32",
        "frames": len(rows),
        "measured_frames": len(measured_rows),

        "avg_latency_ms":
            statistics.mean(latencies),

        "median_latency_ms":
            statistics.median(latencies),

        "p95_latency_ms":
            p95_latency,

        "avg_inference_ms":
            statistics.mean(inference_times),

        "throughput_fps":
            throughput_fps,

        "avg_ram_mb":
            statistics.mean(ram_values),

        "avg_vram_mb":
            statistics.mean(vram_values),

        "avg_objects":
            statistics.mean(object_counts),

        "avg_tracks":
            statistics.mean(track_counts),

        "runtime_sec":
            experiment_end - experiment_start,
    }

    if device != "cpu":

        summary["peak_vram_mb"] = (
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 2)
        )

    else:

        summary["peak_vram_mb"] = 0.0

    print("\nResults:")
    print(
        f"Average latency : "
        f"{summary['avg_latency_ms']:.2f} ms"
    )
    print(
        f"Median latency  : "
        f"{summary['median_latency_ms']:.2f} ms"
    )
    print(
        f"P95 latency     : "
        f"{summary['p95_latency_ms']:.2f} ms"
    )
    print(
        f"Inference       : "
        f"{summary['avg_inference_ms']:.2f} ms"
    )
    print(
        f"Throughput      : "
        f"{summary['throughput_fps']:.2f} FPS"
    )
    print(
        f"Objects/frame   : "
        f"{summary['avg_objects']:.2f}"
    )
    print(
        f"Tracks/frame    : "
        f"{summary['avg_tracks']:.2f}"
    )
    print(
        f"RAM             : "
        f"{summary['avg_ram_mb']:.2f} MB"
    )

    if device != "cpu":
        print(
            f"VRAM            : "
            f"{summary['avg_vram_mb']:.2f} MB"
        )
        print(
            f"Peak VRAM       : "
            f"{summary['peak_vram_mb']:.2f} MB"
        )

    print(f"Saved           : {csv_path}")

    # Clean up before next experiment
    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def save_summary(summaries):

    summary_path = os.path.join(
        RESULTS_DIR,
        "benchmark_summary.csv"
    )

    fieldnames = [
        "experiment",
        "device",
        "precision",
        "frames",
        "measured_frames",
        "avg_latency_ms",
        "median_latency_ms",
        "p95_latency_ms",
        "avg_inference_ms",
        "throughput_fps",
        "avg_ram_mb",
        "avg_vram_mb",
        "peak_vram_mb",
        "avg_objects",
        "avg_tracks",
        "runtime_sec",
    ]

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(summaries)

    print("\n")
    print("=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)

    for s in summaries:

        print(
            f"{s['experiment']:12s} | "
            f"{s['avg_latency_ms']:7.2f} ms | "
            f"{s['throughput_fps']:7.2f} FPS | "
            f"{s['avg_inference_ms']:7.2f} ms inference"
        )

    print("=" * 65)

    print(
        f"\nSummary saved to: {summary_path}"
    )


def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(
            f"Benchmark video not found: {VIDEO_PATH}"
        )

    print("\nEdgeTrack-AIPC Benchmark")
    print("-------------------------")
    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.version.cuda}")
    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU     : "
            f"{torch.cuda.get_device_name(0)}"
        )

    summaries = []

    for experiment in EXPERIMENTS:

        if (
            experiment["device"] != "cpu"
            and not torch.cuda.is_available()
        ):
            print(
                f"Skipping {experiment['name']}: "
                "CUDA unavailable."
            )
            continue

        summary = run_experiment(
            experiment
        )

        summaries.append(summary)

    save_summary(summaries)


if __name__ == "__main__":
    main()