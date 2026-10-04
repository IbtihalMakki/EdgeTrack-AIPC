from ultralytics import YOLO


def main():
    print("EdgeTrack-AIPC")
    print("Loading YOLO model...")

    model = YOLO("yolo11n.pt")

    print("Model loaded successfully.")

    # Run real-time object detection + ByteTrack tracking
    model.track(
        source=0,
        show=True,
        tracker="bytetrack.yaml",
        persist=True,
        conf=0.4
    )


if __name__ == "__main__":
    main()