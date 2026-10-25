"""Shared preprocessing and evaluation for both standalone Kaggle notebooks."""
import contextlib
import csv
import io
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset

SCORE_FLOOR = 0.001
NMS_IOU = 0.5
MAX_DETECTIONS = 100
PRIMARY_METRICS = ("mAP50_95", "AP50", "precision", "recall", "F1", "mean_IoU_TP", "FPS")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Forcing all deterministic ops makes Torchvision switch CUDA ROIAlign to a
    # different compiled implementation. Keep native detection kernels for both
    # the T4 training budget and the inference benchmark; seeds/cuDNN stay fixed.
    torch.use_deterministic_algorithms(False)
    cv2.setNumThreads(0)


def seed_worker(_):
    value = torch.initial_seed() % 2**32
    np.random.seed(value)
    random.seed(value)
    cv2.setNumThreads(0)


def letterbox(image, size=640):
    """RGB uint8 → RGB float CHW; return the actual x/y resize factors and padding."""
    width, height = image.size
    ratio = min(size / width, size / height)
    nw, nh = round(width * ratio), round(height * ratio)
    resized = cv2.resize(np.asarray(image), (nw, nh), interpolation=cv2.INTER_LINEAR)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top+nh, left:left+nw] = resized
    tensor = torch.from_numpy(canvas.transpose(2, 0, 1).copy()).float() / 255.0
    return tensor, (nw / width, nh / height, left, top, width, height)


def transform_boxes(boxes, geometry, inverse=False):
    boxes = boxes.clone().reshape(-1, 4)
    sx, sy, left, top, width, height = geometry
    if inverse:
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / sx).clamp(0, width)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / sy).clamp(0, height)
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * sx + left
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * sy + top
    return boxes


class CocoDataset(Dataset):
    def __init__(self, root, split, size=640, augment=False):
        self.root, self.size, self.augment = Path(root), size, augment
        self.annotation_path = self.root / "annotations" / f"instances_{split}.json"
        self.data = json.loads(self.annotation_path.read_text())
        self.images = sorted(self.data["images"], key=lambda x: x["id"])
        self.annotations = defaultdict(list)
        for ann in self.data["annotations"]:
            assert ann["category_id"] == 1 and ann["iscrowd"] == 0
            self.annotations[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        entry = self.images[index]
        with Image.open(self.root / entry["file_name"]) as source:
            image = source.convert("RGB")
        boxes = torch.tensor([a["bbox"] for a in self.annotations[entry["id"]]], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                x1, x2 = boxes[:, 0].clone(), boxes[:, 2].clone()
                boxes[:, 0], boxes[:, 2] = image.width - x2, image.width - x1
            for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                image = enhancer(image).enhance(random.uniform(0.9, 1.1))
        tensor, geometry = letterbox(image, self.size)
        boxes = transform_boxes(boxes, geometry)
        target = {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64),
                  "image_id": torch.tensor(entry["id"]),
                  "area": (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1]),
                  "iscrowd": torch.zeros(len(boxes), dtype=torch.int64)}
        return tensor, target, geometry


def detection_collate(batch):
    images, targets, geometry = zip(*batch)
    return list(images), list(targets), list(geometry)


def coco_metrics(annotation_path, predictions):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(annotation_path))
        if predictions:
            result = gt.loadRes(predictions)
        else:
            result = COCO()
            result.dataset = {"images": gt.dataset["images"], "categories": gt.dataset["categories"], "annotations": []}
            result.createIndex()
        evaluator = COCOeval(gt, result, "bbox")
        evaluator.params.maxDets = [1, 10, MAX_DETECTIONS]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    names = ("mAP50_95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
             "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large")
    return {name: (float(v) if v >= 0 else None) for name, v in zip(names, evaluator.stats)}


class Predictor:
    """Shared FP32 inference boundary: letterboxed tensors → original-image COCO boxes."""
    def __init__(self, model, family, device):
        self.model, self.family, self.device = model.float().eval(), family, torch.device(device)

    @torch.inference_mode()
    def __call__(self, images, geometry, ids):
        tensors = [im.to(self.device, non_blocking=True) for im in images]
        if self.family == "fasterrcnn":
            outputs = self.model(tensors)
            detections = [(o["boxes"], o["scores"]) for o in outputs]
        else:
            from ultralytics.utils.nms import non_max_suppression
            raw = self.model(torch.stack(tensors))
            raw = raw[0] if isinstance(raw, tuple) else raw
            outputs = non_max_suppression(raw, conf_thres=SCORE_FLOOR, iou_thres=NMS_IOU,
                                           max_det=MAX_DETECTIONS, nc=1)
            detections = [(o[:, :4], o[:, 4]) for o in outputs]
        records = []
        for (boxes, scores), geo, image_id in zip(detections, geometry, ids):
            boxes = transform_boxes(boxes, geo, inverse=True).cpu()
            scores = scores.detach().cpu()
            for box, score in zip(boxes.tolist(), scores.tolist()):
                x1, y1, x2, y2 = box
                if x2 > x1 and y2 > y1 and score >= SCORE_FLOOR:
                    records.append({"image_id": int(image_id), "category_id": 1,
                                    "bbox": [x1, y1, x2-x1, y2-y1], "score": score})
        return records


def predict_dataset(predictor, dataset, batch_size=8, workers=2):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=predictor.device.type == "cuda",
        multiprocessing_context="spawn" if workers else None,
        collate_fn=detection_collate, worker_init_fn=seed_worker)
    records = []
    for images, targets, geometry in loader:
        records.extend(predictor(images, geometry, [int(t["image_id"]) for t in targets]))
    return records


def matched_detections(annotation_path, predictions):
    """Greedy score-ordered, class-aware one-to-one matching at IoU=0.5."""
    data = json.loads(Path(annotation_path).read_text())
    truth = defaultdict(list)
    for ann in data["annotations"]:
        truth[ann["image_id"]].append(ann["bbox"])
    grouped = defaultdict(list)
    for p in predictions:
        grouped[p["image_id"]].append(p)
    records = []
    for image_id, preds in grouped.items():
        boxes = np.asarray(truth[image_id], dtype=float).reshape(-1, 4)
        used = np.zeros(len(boxes), dtype=bool)
        for p in sorted(preds, key=lambda x: -x["score"]):
            x,y,w,h = p["bbox"]
            if len(boxes):
                inter = np.maximum(0, np.minimum(x+w, boxes[:,0]+boxes[:,2])-np.maximum(x,boxes[:,0])) * np.maximum(
                    0, np.minimum(y+h, boxes[:,1]+boxes[:,3])-np.maximum(y,boxes[:,1]))
                ious = inter / np.maximum(w*h+boxes[:,2]*boxes[:,3]-inter, 1e-12)
                ious[used] = -1
                idx = int(ious.argmax())
                tp = bool(ious[idx] >= 0.5)
                matched_iou = float(ious[idx]) if tp else 0.0
                if tp:
                    used[idx] = True
            else:
                tp, matched_iou = False, 0.0
            records.append({"score": p["score"], "tp": int(tp), "iou": matched_iou, "image_id": image_id})
    return records, len(data["annotations"]), [im["id"] for im in data["images"] if not truth[im["id"]]], len(data["images"])


def select_threshold(annotation_path, predictions):
    records, total, _, _ = matched_detections(annotation_path, predictions)
    if not records:
        return 1.0, []
    records.sort(key=lambda r: -r["score"])
    tp, curve = 0, []
    for i, row in enumerate(records):
        tp += row["tp"]
        if i+1 < len(records) and records[i+1]["score"] == row["score"]:
            continue
        precision, recall = tp/(i+1), tp/max(total,1)
        f1 = 2*precision*recall/max(precision+recall,1e-12)
        curve.append({"threshold": row["score"], "precision": precision, "recall": recall, "f1": f1})
    best = max(curve, key=lambda r: (r["f1"], r["threshold"]))
    if best["f1"] == 0:
        return 1.0, curve
    return best["threshold"], curve


def operating_metrics(annotation_path, predictions, threshold):
    records, total, negatives, count = matched_detections(annotation_path, predictions)
    selected = [r for r in records if r["score"] >= threshold]
    tp = sum(r["tp"] for r in selected)
    fp, fn = len(selected)-tp, total-tp
    precision, recall = tp/max(tp+fp,1), tp/max(total,1)
    false_images = {r["image_id"] for r in selected} & set(negatives)
    return {"confidence_threshold": threshold, "matching_IoU": 0.5,
            "precision": precision, "recall": recall,
            "F1": 2*precision*recall/max(precision+recall,1e-12),
            "TP": tp, "FP": fp, "FN": fn,
            "mean_IoU_TP": sum(r["iou"] for r in selected)/tp if tp else None,
            "negative_image_false_alarm_rate": len(false_images)/len(negatives) if negatives else None,
            "false_positives_per_image": fp/count}


def benchmark(predictor, dataset, samples=100, warmup=20):
    """Batch 1, FP32, one GPU; excludes file decoding, includes all other inference steps."""
    indices = np.linspace(0, len(dataset)-1, min(samples, len(dataset)), dtype=int)
    originals = []
    for index in indices:
        entry = dataset.images[index]
        with Image.open(dataset.root / entry["file_name"]) as im:
            originals.append((im.convert("RGB"), entry["id"]))
    cuda = predictor.device.type == "cuda"
    def one(item):
        tensor, geo = letterbox(item[0], dataset.size)
        return predictor([tensor], [geo], [item[1]])
    for i in range(warmup):
        one(originals[i % len(originals)])
    if cuda:
        torch.cuda.synchronize(predictor.device)
        torch.cuda.reset_peak_memory_stats(predictor.device)
    elapsed = []
    for item in originals:
        if cuda:
            torch.cuda.synchronize(predictor.device)
        start = time.perf_counter()
        one(item)
        if cuda:
            torch.cuda.synchronize(predictor.device)
        elapsed.append((time.perf_counter()-start)*1000)
    return {"latency_mean_ms": float(np.mean(elapsed)), "latency_median_ms": float(np.median(elapsed)),
            "latency_p95_ms": float(np.percentile(elapsed,95)), "FPS": 1000/float(np.mean(elapsed)),
            "inference_peak_VRAM_MB": torch.cuda.max_memory_allocated(predictor.device)/1024**2 if cuda else None,
            "latency_samples": len(elapsed), "batch_size": 1, "inference_precision": "FP32",
            "timing_scope": "predecoded RGB -> letterbox -> H2D -> model -> NMS -> original coords -> CPU records; excludes disk/decode"}


def render_examples(dataset, predictions, threshold, path):
    grouped = defaultdict(list)
    for p in predictions:
        if p["score"] >= threshold:
            grouped[p["image_id"]].append(p)
    positive = [i for i in dataset.images if dataset.annotations[i["id"]]][:8]
    negative = [i for i in dataset.images if not dataset.annotations[i["id"]]][:8]
    sheet = Image.new("RGB", (640, 4*310), "white")
    for k, entry in enumerate(positive+negative):
        with Image.open(dataset.root/entry["file_name"]) as source:
            im = source.convert("RGB")
        draw = ImageDraw.Draw(im)
        for ann in dataset.annotations[entry["id"]]:
            x,y,w,h = ann["bbox"]
            draw.rectangle((x,y,x+w,y+h), outline="lime", width=2)
        for p in grouped[entry["id"]]:
            x,y,w,h = p["bbox"]
            draw.rectangle((x,y,x+w,y+h), outline="red", width=1)
        sheet.paste(im, ((k%4)*160,(k//4)*310+20))
        ImageDraw.Draw(sheet).text(((k%4)*160,(k//4)*310+3),f'id={entry["id"]}',fill="black")
    sheet.save(path)


def final_evaluation(model, family, config):
    out = Path(config["output"])
    predictor = Predictor(model.to(config["eval_device"]), family, config["eval_device"])
    val = CocoDataset(config["root"], "val", config["size"])
    test = CocoDataset(config["root"], "test", config["size"])
    val_predictions = predict_dataset(predictor, val, config["eval_batch"], config["workers"])
    threshold, curve = select_threshold(val.annotation_path, val_predictions)
    save_json(out/"validation_predictions.json", val_predictions)
    save_json(out/"confidence_selection.json", {"selected_on": "validation", "rule": "maximum F1 at IoU=0.5; ties prefer higher confidence", "threshold": threshold})
    save_json(out/"validation_PR_F1_curve.json", curve)
    # The threshold is locked before reading test predictions.
    predictions = predict_dataset(predictor, test, config["eval_batch"], config["workers"])
    save_json(out/"test_predictions.json", predictions)
    metrics = {"model": family, "seed": config["seed"],
               "validation": {**coco_metrics(val.annotation_path,val_predictions), **operating_metrics(val.annotation_path,val_predictions,threshold)},
               "test": {**coco_metrics(test.annotation_path,predictions), **operating_metrics(test.annotation_path,predictions,threshold)},
               "parameters": sum(p.numel() for p in model.parameters()),
               "benchmark": benchmark(predictor,test,config["benchmark_samples"])}
    save_json(out/"metrics.json",metrics)
    row = {"model":family,"seed":config["seed"],
           **{key:metrics["test"][key] for key in PRIMARY_METRICS if key != "FPS"},
           "FPS":metrics["benchmark"]["FPS"]}
    with (out/"comparison.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    render_examples(test,predictions,threshold,out/"test_examples.jpg")
    print(json.dumps(metrics,indent=2))
    return metrics


def prepare_gpu_smoke_config(config, runtime):
    """Prepare a tiny train/val subset for the actual two-GPU/AMP/worker preflight.

    Never samples test or changes the real training dataset/config/output.
    """
    runtime = Path(runtime)
    source = Path(config["root"])
    root = runtime / "gpu_smoke_data"
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    image_link = root / "images"
    target = (source / "images").resolve()
    if image_link.is_symlink() and image_link.resolve() != target:
        image_link.unlink()
    if not image_link.is_symlink():
        image_link.symlink_to(target, target_is_directory=True)
    for split, count in (("train", max(config["batch"], 8)), ("val", 4)):
        data = json.loads((source / "annotations" / f"instances_{split}.json").read_text())
        positives = {a["image_id"] for a in data["annotations"]}
        images = sorted(data["images"], key=lambda im:im["id"])
        take = [im for im in images if im["id"] in positives][:count//2]
        take += [im for im in images if im["id"] not in positives][:count-count//2]
        selected_ids = {im["id"] for im in take}
        if len(take) != count:
            raise ValueError(f"Not enough positive/negative images for {split} smoke test")
        data["images"] = take
        data["annotations"] = [a for a in data["annotations"] if a["image_id"] in selected_ids]
        save_json(root / "annotations" / f"instances_{split}.json", data)
    yaml = root / "data.yaml"
    yaml.write_text(f'path: {json.dumps(str(root.resolve()))}\ntrain: images/train\nval: images/val\nnames: [varroa]\nnc: 1\n')
    smoke = {**config, "root":str(root.resolve()), "data_yaml":str(yaml.resolve()),
             "output":str((runtime / "gpu_smoke_run").resolve()),
             "epochs":1, "patience":1, "warmup_epochs":0, "resume":None}
    path = runtime / "gpu_smoke_config.json"
    save_json(path, smoke)
    return path
