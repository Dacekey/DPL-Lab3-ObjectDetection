# Varroa — ready for YOLOv8s and Faster R-CNN

Upload **varroa_kaggle_ready.zip** as one Kaggle Dataset. Both models use the same images and boxes.
The archive places `data.yaml`, `images/`, `labels/`, and `annotations/` directly at its root.
All 13,507 images retain their original PNG bytes and 160 × 280 resolution.

| Split | Ảnh | Có Varroa | Không có box | Bounding boxes |
|---|---:|---:|---:|---:|
| train | 8,223 | 2,553 | 5,670 | 2,903 |
| val | 1,876 | 451 | 1,425 | 655 |
| test | 3,408 | 942 | 2,466 | 1,068 |

## Files

```text
data.yaml                       # Ultralytics dataset configuration
images/{train,val,test}/<video>/*.png
labels/{train,val,test}/<video>/*.txt
annotations/instances_train.json
annotations/instances_val.json
annotations/instances_test.json
classes.json                    # Explicit category mappings and future training seed
manifest.csv                    # Image IDs, source flags, original paths and hashes
quality/                        # Audit, validation, and annotated visual samples
SHA256SUMS                      # Checksums of all package files except itself
```

## YOLOv8s

Locate the absolute path to `data.yaml` under `/kaggle/input/` and pass that path as
the Ultralytics `data` argument. The YAML intentionally omits `path` so paths resolve
from the YAML directory. One class: **0 = varroa**. Empty TXT files are negative images.
Do not upload the original ZIPs or perform another label conversion.

## Faster R-CNN ResNet50-FPN

Use this same dataset root and `annotations/instances_<split>.json` (standard COCO).
Every image entry has a `file_name` relative to the dataset root and a globally unique ID.
Box format is absolute pixel **[x, y, width, height]**; category **1 = varroa**.
For Torchvision, the training Dataset adapter converts boxes to **[x1, y1, x2, y2]**,
returns RGB float image tensors in [0,1], int64 labels of 1, and reserves label 0 for
background (`num_classes=2`). Negative targets have boxes shape (0,4) and labels shape (0,).
The annotations already contain `area` and `iscrowd=0` for COCO evaluation.
No additional annotation cleanup or split preparation is needed.

## Experiment agreement

Use the fixed original split for both models and one training seed: **42**.
Resize/letterbox and training augmentation belong to the training input pipelines;
they are not baked into these original-resolution images. Apply the same agreed
640 × 640 letterbox policy to both models and avoid a second internal resize in Faster R-CNN.
Select checkpoints and confidence thresholds on validation; reserve test for final evaluation.
Use the same COCO evaluator and map YOLO category 0 to COCO category 1.

## Annotation decisions

The source image-level flags 1 and 3 are mapped to the single detection class `varroa`,
based on their box-bearing records and visual review. The author's README does not
explain the precise distinction of flag 3; this is an explicit task-level mapping,
not a claim that the author documented the flags as identical. Source flags are preserved.
Flag 0 means no annotated objects; healthy bees are not a detection category.
Source `labels/` TXT files are canonical and retain decimal coordinate precision.
Two duplicate CSV paths are resolved once each; an alternative annotation is not
added as a second object. One zero-width source box is removed while retaining the
other valid box on the same positive training image.

See `quality/audit.json` and `quality/processing_report.md` for complete details.

## Attribution

Schurischuster Stefan & Martin Kampel (2020), **VarroaDataset**, version 1.2.0.
[Dataset and license](https://doi.org/10.5281/zenodo.4085044),
[author repository](https://github.com/schurist/VarroaDataset/).
License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Changes: annotation conversion, duplicate record resolution, and removal of one invalid box;
original images are unchanged. This package does not contain trained model results.
