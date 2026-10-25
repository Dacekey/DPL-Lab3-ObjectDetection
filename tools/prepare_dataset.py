"""Build a validated, portable YOLO + COCO dataset from the original archives.

Run from any directory: python /path/to/tools/prepare_dataset.py
Only Python's standard library and Pillow are required.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

LAB = Path(__file__).resolve().parents[1]
SOURCE = LAB / "dataset"
OUT = SOURCE / "varroa_ready"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    if OUT.exists():
        raise SystemExit(f"Output already exists; refusing to overwrite: {OUT}")
    OUT.mkdir(parents=True)
    audit = {"created_utc": datetime.now(timezone.utc).isoformat(),
             "seed_for_future_training": 42, "sources": {}, "splits": {},
             "duplicate_csv_rows": [], "clipped_boxes": [],
             "removed_duplicate_boxes": [], "removed_invalid_boxes": [], "annotation_variants": [],
             "exact_image_duplicates": [], "near_duplicate_candidates": []}
    all_manifest = []
    byte_hashes, pixel_hashes, dhashes = defaultdict(list), defaultdict(list), []
    review_tiles = defaultdict(list)
    gt_global = defaultdict(list)
    for line in (SOURCE / "gt.csv").read_text().splitlines():
        t = line.split()
        gt_global[t[0]].append(t[1:])
    next_image_id, next_annotation_id = 1, 1
    for split in ("train", "val", "test"):
        archives = sorted(SOURCE.glob(f"{split}*.part"))
        if len(archives) != 1:
            raise ValueError(f"Expected one source archive for {split}")
        archive = archives[0]
        print(f"Processing {split}: {archive.name}", flush=True)
        audit["sources"][archive.name] = {
            "bytes": archive.stat().st_size,
            "sha256": sha256(archive.read_bytes()),
        }
        coco = {"info": {"description": "VarroaDataset, one-class mite detection",
                         "version": "1.0"},
                "licenses": [{"id": 1, "name": "CC BY 4.0",
                              "url": "https://creativecommons.org/licenses/by/4.0/"}],
                "images": [], "annotations": [],
                "categories": [{"id": 1, "name": "varroa", "supercategory": "mite"}]}
        stats = Counter()
        sizes, flags = Counter(), Counter()
        widths, heights, areas = [], [], []
        videos = set()
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Ambiguous duplicate ZIP member")
            # Reading every member verifies its CRC, including non-image metadata.
            bad = z.testzip()
            if bad:
                raise ValueError(f"CRC failed: {bad}")
            records = defaultdict(list)
            for line in z.read(f"{split}/gt_one.csv").decode().splitlines():
                t = line.split()
                if gt_global[f"{split}/{t[0]}"].count(t[1:]) == 0:
                    raise ValueError(f"Global/split CSV mismatch: {t[0]}")
                records[t[0]].append(t[1:])
            image_names = {n for n in names if n.lower().endswith(".png")}
            if image_names != {f"{split}/{n}" for n in records}:
                raise ValueError("Image and CSV path sets differ")
            for rel, rows in sorted(records.items()):
                source_path = f"{split}/{rel}"
                if len(rows) > 1:
                    audit["duplicate_csv_rows"].append({"path": source_path, "rows": rows,
                        "resolution": "Use canonical labels/<video>/<image>.txt; do not union alternate boxes."})
                parts = Path(rel).parts
                if len(parts) != 3 or parts[0] != "videos":
                    raise ValueError(f"Unexpected path: {rel}")
                video, filename = parts[1:]
                videos.add(video)
                label_path = f"{split}/labels/{video}/{Path(filename).stem}.txt"
                original_label = z.read(label_path).decode().splitlines()
                flag = int(original_label[0].strip())
                if flag not in (0, 1, 3):
                    raise ValueError(f"Unknown flag: {flag}")
                raw_boxes = [list(map(float, line.split())) for line in original_label[1:] if line.strip()]
                if any(len(b) != 4 or not all(math.isfinite(v) for v in b) for b in raw_boxes):
                    raise ValueError(f"Malformed box: {label_path}")
                if (flag == 0) != (len(raw_boxes) == 0):
                    raise ValueError(f"Flag/box mismatch: {label_path}")
                # Source TXT keeps subpixel precision; verify against rounded original CSV.
                rounded = [flag] + [round(v) for box in raw_boxes for v in box]
                if not any([int(v) for v in row] == rounded for row in rows):
                    raise ValueError(f"Canonical TXT does not match any CSV row: {label_path}")
                alternate = f"{split}/videos/{video}/{Path(filename).stem}.txt"
                if alternate in names and z.read(alternate) != z.read(label_path):
                    audit["annotation_variants"].append({"path": source_path,
                        "chosen": label_path, "unused": alternate,
                        "chosen_content": z.read(label_path).decode(),
                        "unused_content": z.read(alternate).decode()})
                raw = z.read(source_path)
                with Image.open(io.BytesIO(raw)) as source_im:
                    source_im.load()
                    im = source_im.convert("RGB")
                width, height = im.size
                boxes = []
                for box in raw_boxes:
                    x1, y1, x2, y2 = box
                    if x2 <= x1 or y2 <= y1:
                        audit["removed_invalid_boxes"].append({"path": source_path,
                            "box": box, "reason": "non-positive width or height"})
                        continue
                    clipped = [max(0., min(width, x1)), max(0., min(height, y1)),
                               max(0., min(width, x2)), max(0., min(height, y2))]
                    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                        audit["removed_invalid_boxes"].append({"path": source_path,
                            "box": box, "reason": "no positive area inside image"})
                        continue
                    if box != clipped:
                        audit["clipped_boxes"].append({"path": source_path, "before": box, "after": clipped})
                    if clipped in boxes:
                        audit["removed_duplicate_boxes"].append({"path": source_path, "box": clipped})
                        continue
                    boxes.append(clipped)
                if flag != 0 and not boxes:
                    raise ValueError(f"Positive image lost all valid boxes: {source_path}")
                target_rel = f"images/{split}/{video}/{filename}"
                target = OUT / target_rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                yolo_path = OUT / "labels" / split / video / f"{Path(filename).stem}.txt"
                yolo_path.parent.mkdir(parents=True, exist_ok=True)
                yolo_lines = []
                for x1, y1, x2, y2 in boxes:
                    w, h = x2 - x1, y2 - y1
                    yolo_lines.append(f"0 {(x1+x2)/(2*width):.12f} {(y1+y2)/(2*height):.12f} {w/width:.12f} {h/height:.12f}")
                    coco["annotations"].append({"id": next_annotation_id, "image_id": next_image_id,
                        "category_id": 1, "bbox": [x1, y1, w, h], "area": w*h, "iscrowd": 0})
                    next_annotation_id += 1
                    widths.append(w); heights.append(h); areas.append(w*h)
                yolo_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
                coco["images"].append({"id": next_image_id, "file_name": target_rel,
                    "width": width, "height": height, "license": 1})
                pixel_hash = sha256(f"{width},{height},RGB:".encode() + im.tobytes())
                file_hash = sha256(raw)
                thumbnail = im.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
                small = list(thumbnail.get_flattened_data() if hasattr(thumbnail, "get_flattened_data") else thumbnail.getdata())
                dh = sum((small[y*9+x] > small[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
                byte_hashes[file_hash].append(source_path)
                pixel_hashes[pixel_hash].append(source_path)
                dhashes.append((source_path, split, dh))
                all_manifest.append({"image_id": next_image_id, "split": split,
                    "file_name": target_rel, "source_path": source_path, "source_flag": flag,
                    "width": width, "height": height, "num_boxes": len(boxes),
                    "sha256": file_hash, "pixel_sha256": pixel_hash, "dhash": f"{dh:016x}"})
                next_image_id += 1
                stats["images"] += 1
                stats["boxes"] += len(boxes)
                stats["positive_images" if boxes else "negative_images"] += 1
                sizes[f"{width}x{height}"] += 1
                flags[str(flag)] += 1
                key = (split, flag)
                if len(review_tiles[key]) < 4:
                    draw = ImageDraw.Draw(im)
                    for box in boxes:
                        draw.rectangle(box, outline="red", width=2)
                    review_tiles[key].append((im.copy(), source_path))
        dump(OUT / "annotations" / f"instances_{split}.json", coco)
        audit["splits"][split] = {**stats, "source_csv_rows": sum(map(len, records.values())),
            "dimensions": dict(sizes), "source_flags": dict(flags), "video_groups": sorted(videos),
            "box_width_median": statistics.median(widths), "box_height_median": statistics.median(heights),
            "coco_area_small": sum(a < 32**2 for a in areas),
            "coco_area_medium": sum(32**2 <= a < 96**2 for a in areas),
            "coco_area_large": sum(a >= 96**2 for a in areas)}
        print(audit["splits"][split], flush=True)
    for digest, paths in pixel_hashes.items():
        if len(paths) > 1:
            cross = len({p.split('/')[0] for p in paths}) > 1
            audit["exact_image_duplicates"].append({"pixel_sha256": digest, "paths": paths, "cross_split": cross})
    # Perceptual hashes are review candidates only, never grounds for deleting labels.
    # Bucket by 4 x 16-bit chunks: distance <= 3 must share at least one chunk.
    buckets = defaultdict(list)
    candidate_pairs = set()
    for idx, (path, split, h) in enumerate(dhashes):
        candidates = set()
        for chunk in range(4):
            candidates.update(buckets[(chunk, (h >> (16*chunk)) & 65535)])
        for j in candidates:
            other, other_split, oh = dhashes[j]
            if other_split != split and (h ^ oh).bit_count() <= 3:
                candidate_pairs.add((j, idx))
        for chunk in range(4):
            buckets[(chunk, (h >> (16*chunk)) & 65535)].append(idx)
    audit["near_duplicate_candidates"] = [
        {"a": dhashes[a][0], "b": dhashes[b][0],
         "hamming_distance": (dhashes[a][2] ^ dhashes[b][2]).bit_count()}
        for a, b in sorted(candidate_pairs)]
    audit["cross_split_video_overlap"] = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        audit["cross_split_video_overlap"][f"{a}-{b}"] = sorted(
            set(audit["splits"][a]["video_groups"]) & set(audit["splits"][b]["video_groups"]))
    if any(d["cross_split"] for d in audit["exact_image_duplicates"]):
        raise ValueError("Cross-split pixel-identical images require resolution before release")
    if any(audit["cross_split_video_overlap"].values()):
        raise ValueError("Cross-split video overlap")
    with (OUT / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_manifest[0]))
        writer.writeheader(); writer.writerows(all_manifest)
    # Omit `path` so Ultralytics resolves relative paths from the absolute YAML file path.
    (OUT / "data.yaml").write_text("train: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames:\n  0: varroa\n")
    (OUT / "classes.json").write_text('{"names": ["varroa"], "yolo": {"varroa": 0}, "coco": {"varroa": 1}, "torchvision": {"background": 0, "varroa": 1}, "training_seed": 42}\n')
    previews = OUT / "quality"
    previews.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        sheet = Image.new("RGB", (640, 930), "white")
        draw = ImageDraw.Draw(sheet)
        for row, flag in enumerate((0, 1, 3)):
            for col, (im, path) in enumerate(review_tiles[(split, flag)]):
                im.thumbnail((160, 280))
                sheet.paste(im, (col*160, row*310+25))
                draw.text((col*160+2, row*310+5), f"{split} flag={flag}", fill="black")
        sheet.save(previews / f"{split}_annotations.jpg", quality=90)
    dump(OUT / "quality" / "audit.json", audit)
    print("Prepared:", OUT, flush=True)


if __name__ == "__main__":
    main()
