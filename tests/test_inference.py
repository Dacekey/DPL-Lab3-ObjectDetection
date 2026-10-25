"""Inference output and geometry regressions, without downloading weights."""
import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
import predict
from common import Predictor


def test_inference_boxes_are_in_original_pixels_and_thresholded():
    class FixedDetector(torch.nn.Module):
        def forward(self, images):
            # 160x280 image -> 640 canvas, left padding 137, y scale 640/280.
            return [{"boxes": torch.tensor([[137., 0., 503., 640.], [137., 0., 200., 100.]]),
                     "scores": torch.tensor([0.9, 0.1])}]
    predictor = Predictor(FixedDetector(), "fasterrcnn", "cpu")
    rows = predict.predict_image(predictor, Image.new("RGB", (160, 280)),
                                 {"size": 640, "confidence": .8}, image_id=9)
    assert len(rows) == 1 and rows[0]["image_id"] == 9
    assert rows[0]["bbox"] == pytest.approx([0, 0, 160, 280])


def test_folder_outputs_preserve_empty_images_and_duplicate_stems(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    Image.new("RGB", (20, 30)).save(source / "nested/a.jpg")
    Image.new("RGB", (20, 30)).save(source / "nested/a.png")
    output = source / "results"
    (output / "old").mkdir(parents=True)
    Image.new("RGB", (20, 30)).save(output / "old/ignore.png")
    class EmptyPredictor:
        def __call__(self, *args):
            return []
    monkeypatch.setattr(predict, "load_detector", lambda *a, **k:
                        (EmptyPredictor(), {"size": 640, "confidence": .8}))
    predict.run_inference(source, output)
    for family in ("fasterrcnn", "yolov8s"):
        run = json.loads((output / family / "run.json").read_text())
        assert run["image_count"] == 2 and run["detection_count"] == 0
        assert {im["file_name"] for im in run["images"]} == {"nested/a.jpg", "nested/a.png"}
        assert json.loads((output / family / "predictions.json").read_text()) == []
        assert (output / family / "images/nested/a.jpg.png").is_file()
        assert (output / family / "images/nested/a.png.png").is_file()
    with Image.open(output / "comparison/nested/a.jpg.png") as comparison:
        assert comparison.size == (40, 52)


def test_bad_source_and_overlapping_output_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict.collect_images(tmp_path / "missing", tmp_path / "out")
    with pytest.raises(ValueError, match="Output"):
        predict.collect_images(tmp_path, tmp_path)
    with pytest.raises(ValueError, match="Không có ảnh"):
        predict.collect_images(tmp_path, tmp_path / "out")
