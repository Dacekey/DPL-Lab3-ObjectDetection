"""Independently verify all exported images, COCO boxes, YOLO labels and hashes."""
import csv
import hashlib
import json
import math
from pathlib import Path

from PIL import Image


def validate(root):
    root = Path(root).resolve()
    manifest = list(csv.DictReader((root / "manifest.csv").open()))
    assert len(manifest) == len({r["file_name"] for r in manifest})
    by_path = {r["file_name"]: r for r in manifest}
    all_ids, annotation_ids, pixels = set(), set(), {}
    result = {}
    for split in ("train", "val", "test"):
        data = json.loads((root / "annotations" / f"instances_{split}.json").read_text())
        assert data["categories"] == [{"id": 1, "name": "varroa", "supercategory": "mite"}]
        images = {im["id"]: im for im in data["images"]}
        assert len(images) == len(data["images"])
        assert not (set(images) & all_ids)
        all_ids.update(images)
        grouped = {i: [] for i in images}
        for ann in data["annotations"]:
            assert ann["id"] not in annotation_ids
            annotation_ids.add(ann["id"])
            assert ann["image_id"] in images and ann["category_id"] == 1
            assert ann["iscrowd"] == 0
            x, y, w, h = ann["bbox"]
            im = images[ann["image_id"]]
            assert all(math.isfinite(v) for v in (x,y,w,h))
            assert w > 0 and h > 0 and x >= 0 and y >= 0
            assert x+w <= im["width"] + 1e-8 and y+h <= im["height"] + 1e-8
            assert abs(ann["area"] - w*h) < 1e-8
            grouped[ann["image_id"]].append(ann)
        positive = 0
        expected_images, expected_labels = set(), set()
        for image_id, im in images.items():
            path = root / im["file_name"]
            expected_images.add(path.resolve())
            row = by_path[im["file_name"]]
            assert row["split"] == split and int(row["image_id"]) == image_id
            assert path.is_file()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
            with Image.open(path) as decoded:
                decoded.load()
                assert decoded.size == (im["width"], im["height"])
                rgb = decoded.convert("RGB")
                digest = hashlib.sha256(f'{im["width"]},{im["height"]},RGB:'.encode() + rgb.tobytes()).hexdigest()
            assert digest == row["pixel_sha256"]
            assert digest not in pixels or pixels[digest] == split, "Cross-split identical image"
            pixels[digest] = split
            label = root / "labels" / Path(im["file_name"]).relative_to("images").with_suffix(".txt")
            expected_labels.add(label.resolve())
            lines = label.read_text().splitlines()
            anns = grouped[image_id]
            assert len(lines) == len(anns) == int(row["num_boxes"])
            positive += bool(anns)
            for line, ann in zip(lines, anns):
                t = line.split()
                assert len(t) == 5 and t[0] == "0"
                xc, yc, w, h = map(float, t[1:])
                assert all(0 <= v <= 1 for v in (xc,yc,w,h)) and w > 0 and h > 0
                recovered = [(xc-w/2)*im["width"], (yc-h/2)*im["height"],
                             w*im["width"], h*im["height"]]
                assert all(abs(a-b) < 1e-6 for a,b in zip(recovered,ann["bbox"]))
        assert expected_images == {p.resolve() for p in (root / "images" / split).rglob("*.png")}
        assert expected_labels == {p.resolve() for p in (root / "labels" / split).rglob("*.txt")}
        result[split] = {"images": len(images), "boxes": len(data["annotations"]),
                         "positive_images": positive, "negative_images": len(images)-positive}
    assert len(all_ids) == len(manifest)
    result["checks"] = ["all images decoded", "source SHA-256 preserved", "COCO references and bounds",
        "YOLO to COCO round-trip < 1e-6 pixel", "empty labels for negative images",
        "unique IDs and paths", "no orphan exported images or labels", "no cross-split identical pixels"]
    (root / "quality" / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=str(Path(__file__).resolve().parents[1] / "dataset/varroa_ready"))
    validate(parser.parse_args().root)
