"""Inference on new images using the two trained Lab 3 checkpoints."""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "training"))
from common import NMS_IOU, MAX_DETECTIONS, SCORE_FLOOR, Predictor, letterbox, save_json

DEFAULT_WEIGHTS = {
    "fasterrcnn": LAB / "output/faster_r-cnn/fasterrcnn_seed42/best.pt",
    "yolov8s": LAB / "output/yolo/yolov8s_seed42/weights/best.pt",
}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_images(source, output, limit=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh/thư mục: {source}")
    root = source.parent if source.is_file() else source
    if root == output or root.is_relative_to(output):
        raise ValueError("Output không được chứa thư mục/ảnh nguồn.")
    candidates = [source] if source.is_file() else sorted(source.rglob("*"))
    images = [p for p in candidates if p.is_file() and p.suffix.lower() in EXTENSIONS
              and not p.resolve().is_relative_to(output)]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit phải >= 1")
        images = images[:limit]
    if not images:
        raise ValueError(f"Không có ảnh được hỗ trợ tại {source}")
    return root, images


def resolve_device(value):
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Chọn device cpu, cuda:0 hoặc auto.")
    if device.type == "cuda" and (not torch.cuda.is_available() or
            (device.index or 0) >= torch.cuda.device_count()):
        raise ValueError(f"GPU {value} không khả dụng; chọn --device cpu.")
    return device


def load_detector(family, weights, device="cpu", confidence=None):
    """Load weights without downloading pretrained models; use saved val threshold."""
    weights = Path(weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {weights}")
    run = weights.parent.parent if family == "yolov8s" else weights.parent
    config_path = run / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    size = int(config.get("size", 640))
    threshold_file = run / "confidence_selection.json"
    if confidence is None:
        if not threshold_file.is_file():
            raise FileNotFoundError(f"Thiếu {threshold_file}; chỉ định --confidence.")
        threshold = float(json.loads(threshold_file.read_text())["threshold"])
        threshold_source = str(threshold_file)
    else:
        threshold, threshold_source = float(confidence), "user"
    if not math.isfinite(threshold) or not SCORE_FLOOR <= threshold <= 1:
        raise ValueError(f"Confidence phải nằm trong [{SCORE_FLOOR}, 1].")
    device = resolve_device(str(device))
    if family == "fasterrcnn":
        from fasterrcnn_train import build_model
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        size = int(checkpoint.get("config", {}).get("size", size))
        model = build_model(pretrained=False, size=size)
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
    elif family == "yolov8s":
        from ultralytics import YOLO
        model = YOLO(str(weights)).model
    else:
        raise ValueError(f"Model không được hỗ trợ: {family}")
    predictor = Predictor(model.to(device), family, device)
    metadata = dict(model=family, weights=str(weights), confidence=threshold,
                    confidence_source=threshold_source, size=size, device=str(device),
                    precision="FP32", NMS_IoU=NMS_IOU, max_detections=MAX_DETECTIONS,
                    bbox_format="COCO xywh in original image pixels", category_id=1,
                    category_name="varroa")
    return predictor, metadata


def predict_image(predictor, image, metadata, image_id=1):
    tensor, geometry = letterbox(image.convert("RGB"), metadata["size"])
    return [r for r in predictor([tensor], [geometry], [image_id])
            if r["score"] >= metadata["confidence"]]


def draw_predictions(image, predictions, title=None):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for row in predictions:
        x, y, w, h = row["bbox"]
        draw.rectangle((x, y, x+w, y+h), outline="red", width=2)
        text = f"varroa {row['score']:.3f}"
        tx, ty = max(0, min(x, canvas.width-90)), max(0, y-14)
        draw.rectangle(draw.textbbox((tx, ty), text), fill="white")
        draw.text((tx, ty), text, fill="red")
    if title:
        panel = Image.new("RGB", (canvas.width, canvas.height+22), "white")
        panel.paste(canvas, (0, 22))
        ImageDraw.Draw(panel).text((4, 4), title, fill="black")
        return panel
    return canvas


def run_inference(source, output, model="both", device="auto", confidence=None,
                  limit=None, weights=None):
    """Write per-model COCO predictions, manifest, annotated images and comparisons."""
    output = Path(output).resolve()
    root, paths = collect_images(source, output, limit)
    families = ["fasterrcnn", "yolov8s"] if model == "both" else [model]
    checkpoints = {**DEFAULT_WEIGHTS, **(weights or {})}
    # Manifest IDs are local to this inference run, not dataset annotation IDs.
    manifest = [{"id": i+1, "file_name": str(p.relative_to(root))} for i, p in enumerate(paths)]
    detections = {}
    for family in families:
        predictor, metadata = load_detector(family, checkpoints[family], device, confidence)
        records, per_image = [], {}
        for entry, path in zip(manifest, paths):
            with Image.open(path) as source_image:
                image = source_image.convert("RGB")
            rows = predict_image(predictor, image, metadata, entry["id"])
            records.extend(rows)
            per_image[entry["id"]] = rows
            # Keep extension in the output name: a.jpg and a.png stay distinct.
            target = output / family / "images" / (entry["file_name"] + ".png")
            target.parent.mkdir(parents=True, exist_ok=True)
            draw_predictions(image, rows).save(target)
            print(f"{family}: {entry['file_name']} -> {len(rows)} detections", flush=True)
        save_json(output / family / "predictions.json", records)
        save_json(output / family / "run.json", {**metadata, "source": str(Path(source).resolve()),
                  "images": manifest, "image_count": len(paths), "detection_count": len(records)})
        detections[family] = per_image
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if len(families) == 2:
        for entry, path in zip(manifest, paths):
            with Image.open(path) as source_image:
                image = source_image.convert("RGB")
            panels = [draw_predictions(image, detections[f][entry["id"]],
                      f"{f}: {len(detections[f][entry['id']])} detections") for f in families]
            combined = Image.new("RGB", (image.width*2, image.height+22), "white")
            for index, panel in enumerate(panels):
                combined.paste(panel, (index*image.width, 0))
            target = output / "comparison" / (entry["file_name"] + ".png")
            target.parent.mkdir(parents=True, exist_ok=True)
            combined.save(target)
    print(f"Saved results: {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Ảnh hoặc thư mục ảnh (đệ quy)")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--model", choices=["both", "fasterrcnn", "yolov8s"], default="both")
    parser.add_argument("--device", default="auto", help="auto, cpu hoặc cuda:0")
    parser.add_argument("--confidence", type=float, help="Ghi đè ngưỡng validation cho model đã chọn")
    parser.add_argument("--limit", type=int, help="Giới hạn số ảnh khi chạy thử")
    parser.add_argument("--threads", type=int, default=4, help="Số CPU threads")
    parser.add_argument("--fasterrcnn-weights", type=Path, default=DEFAULT_WEIGHTS["fasterrcnn"])
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_WEIGHTS["yolov8s"])
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads phải >= 1")
    torch.set_num_threads(args.threads)
    try:
        run_inference(args.source, args.output, args.model, args.device, args.confidence,
                      args.limit, {"fasterrcnn": args.fasterrcnn_weights, "yolov8s": args.yolo_weights})
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
