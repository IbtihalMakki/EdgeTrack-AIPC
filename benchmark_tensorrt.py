import os
import time
import csv
import statistics

import cv2
import psutil
import torch
from ultralytics import YOLO


MODEL_PATH = "yolo11n.engine"
VIDEO_PATH = "data/benchmark.mp4"
OUTPUT_CSV = "results/tensorrt_fp16.csv"

CONFIDENCE = 0.4
TRACKER = "bytetrack.yaml"
WARMUP_FRAMES = 10


def main():

    print("\nEdgeTrack-AIPC — TensorRT FP16 Benchmark")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(VIDEO_PATH)

    os.makedirs("results", exist_ok=True)

    print(f"Model : {MODEL_PATH}")
    print(f"GPU   : {torch.cuda.get_device_name(0)}")

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError("Could not open benchmark video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video : {total_frames} frames @ {video_fps:.2f} FPS")
    print("=" * 60)

    process = psutil.Process(os.getpid())

    rows = []
    frame_id = 0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    while True:

        success, frame = cap.read()

        if not success:
            break

        start = time.perf_counter()

        results = model.track(
            frame,
            tracker=TRACKER,
            persist=True,
            conf=CONFIDENCE,
            verbose=False,
            device=0
        )

        torch.cuda.synchronize(0)

        end = time.perf_counter()

        latency_ms = (end - start) * 1000

        inference_ms = results[0].speed.get(
            "inference", 0.0
        )

        ram_mb = (
            process.memory_info().rss /
            (1024 ** 2)
        )

        vram_mb = (
            torch.cuda.memory_allocated(0) /
            (1024 ** 2)
        )

        boxes = results[0].boxes

        objects = (
            len(boxes)
            if boxes is not None
            else 0
        )

        tracks = 0

        if boxes is not None and boxes.id is not None:
            tracks = len(boxes.id)

        rows.append({
            "frame": frame_id,
            "latency_ms": latency_ms,
            "inference_ms": inference_ms,
            "ram_mb": ram_mb,
            "vram_mb": vram_mb,
            "objects": objects,
            "tracks": tracks
        })

        if frame_id % 50 == 0:
            print(
                f"Frame {frame_id:4d} | "
                f"Latency {latency_ms:7.2f} ms | "
                f"Inference {inference_ms:7.2f} ms | "
                f"Objects {objects:2d}"
            )

        frame_id += 1

    cap.release()

    # Save all frame measurements
    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    # Exclude warm-up frames
    measured = rows[WARMUP_FRAMES:]

    latencies = [
        r["latency_ms"]
        for r in measured
    ]

    inference = [
        r["inference_ms"]
        for r in measured
    ]

    sorted_latency = sorted(latencies)

    p95_index = int(
        0.95 * (len(sorted_latency) - 1)
    )

    throughput = (
        len(measured) /
        (sum(latencies) / 1000)
    )

    avg_objects = statistics.mean(
        r["objects"] for r in measured
    )

    avg_tracks = statistics.mean(
        r["tracks"] for r in measured
    )

    print("\n" + "=" * 60)
    print("TensorRT FP16 RESULTS")
    print("=" * 60)

    print(
        f"Frames            : {len(rows)}"
    )

    print(
        f"Average latency   : "
        f"{statistics.mean(latencies):.2f} ms"
    )

    print(
        f"Median latency    : "
        f"{statistics.median(latencies):.2f} ms"
    )

    print(
        f"P95 latency       : "
        f"{sorted_latency[p95_index]:.2f} ms"
    )

    print(
        f"Average inference : "
        f"{statistics.mean(inference):.2f} ms"
    )

    print(
        f"Throughput        : "
        f"{throughput:.2f} FPS"
    )

    print(
        f"Objects/frame     : "
        f"{avg_objects:.2f}"
    )

    print(
        f"Tracks/frame      : "
        f"{avg_tracks:.2f}"
    )

    print(
        f"Average RAM       : "
        f"{statistics.mean(r['ram_mb'] for r in measured):.2f} MB"
    )

    print(
        f"PyTorch VRAM      : "
        f"{statistics.mean(r['vram_mb'] for r in measured):.2f} MB"
    )

    print(
        f"\nSaved to: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()