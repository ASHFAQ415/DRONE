"""
Export a YOLO .pt model to a Raspberry Pi friendly format.

Run on a development machine with ultralytics installed:
    python -m pip install -r requirements-export.txt
    python scripts/export_yolo.py --weights yolov8n.pt --format onnx
"""

from argparse import ArgumentParser
from pathlib import Path


def main():
    parser = ArgumentParser(description="Export YOLO model for edge inference.")
    parser.add_argument("--weights", default="yolov8n.pt", help="Path to the source .pt model")
    parser.add_argument("--format", default="onnx", choices=["onnx", "tflite"], help="Export format")
    parser.add_argument("--imgsz", type=int, default=640, help="Square model input size")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Install export dependencies first: python -m pip install -r requirements-export.txt"
        ) from exc

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Model file not found: {weights}")

    model = YOLO(str(weights))
    exported = model.export(format=args.format, imgsz=args.imgsz, opset=args.opset, simplify=True)
    print(f"Exported model: {exported}")


if __name__ == "__main__":
    main()
