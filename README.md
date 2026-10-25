# Lab 3 — Varroa Object Detection / DPL302m

So sánh Faster R-CNN ResNet50-FPN và YOLOv8s trên VarroaDataset.

- Tác giả báo cáo: **Le Khoi Nguyen**.
- Giảng viên hướng dẫn: **Phan Thi Thu Hong**.
- Subject: **Lab3-ObjectDetection-Varroa/DPL302m**.
- Báo cáo khoa học hai cột: [`report/report.pdf`](report/report.pdf), nguồn [`report/report.tex`](report/report.tex).

## Kết quả test

| Model | mAP50–95 | AP50 | Precision | Recall | F1 | FPS (T4, FP32) |
|---|---:|---:|---:|---:|---:|---:|
| Faster R-CNN | 28.78% | 83.43% | 82.00% | 78.46% | 80.19% | 14.84 |
| YOLOv8s | 21.93% | 71.16% | 80.15% | 68.82% | 74.06% | 50.51 |

Hai run dùng seed 42, cùng split/annotation và evaluator COCO. Confidence chọn riêng
trên validation; đây là so sánh hai recipe fine-tune, chưa cô lập ảnh hưởng kiến trúc.

## Cấu trúc

- `tools/`: chuẩn bị/kiểm tra dataset, sinh notebook training.
- `training/`: dataset adapter, training và evaluation dùng chung.
- `notebooks/`: hai notebook Kaggle độc lập, xem [hướng dẫn](notebooks/README.md).
- `output/`: metrics, predictions, cấu hình, log và lịch sử training đã tải từ Kaggle.
- `inference/`: [notebook và CLI](inference/README.md), ảnh minh họa từ checkpoint thật.
- `tests/`: kiểm tra preprocessing, metrics, DDP failure và inference.
- `docs/`: quyết định chọn model và audit dữ liệu.
- `report/`: LaTeX hai cột, hình, bảng và PDF đã biên dịch.

## Dữ liệu và checkpoint

Git giữ metadata, annotation và kết quả cần để kiểm chứng báo cáo. Dataset ảnh,
checkpoint, ZIP và bản sao paper PDF lớn không đưa vào lịch sử Git; các file local
vẫn giữ nguyên. Nguồn dữ liệu: [VarroaDataset v1.2.0](https://zenodo.org/records/4085044).
Xem `docs/Lab3_Dataset_Preparation_Report.md` để chuẩn bị từ archive gốc; giữ tên archive
`.part` như audit nếu dùng `tools/prepare_dataset.py`. Ảnh gốc 160 × 280, không chia lại split.

Tải artifact vào đúng vị trí để chạy inference:

```bash
kaggle kernels output mthutranhehe/lab3-yolov8 -p output/yolo
kaggle kernels output marcusnguyen007/lab3-faster-r-cnn -p output/faster_r-cnn
```

Cài dependency inference (chọn cặp torch/torchvision phù hợp CPU/CUDA):

```bash
python -m pip install -r inference/requirements.txt
python inference/predict.py --source /path/to/images --model both
```

Inference dùng cùng RGB/letterbox, normalization, NMS và threshold validation như
đánh giá; xem `inference/README.md`. Các đường dẫn trong notebook training dưới Kaggle
được tìm tự động; không cần upload source repo.

## Kiểm chứng và biên dịch báo cáo

```bash
python -m pip install pytest
python -m pytest tests -q
cd report
make
```

Test liên quan ảnh chuẩn bị sẽ skip nếu chưa tải dataset. Biên dịch PDF cần XeLaTeX,
BibTeX, font Noto Serif/Noto Sans/Noto Sans Mono; hình và bảng đã được commit nên
không cần tải dataset/checkpoint để biên dịch. Xem `report/README.md`.

Dataset thuộc tác giả gốc; mã phụ thuộc giữ giấy phép của từng framework. Báo cáo
trích dẫn dataset, paper và source triển khai tương ứng, không gán quyền sở hữu chúng
cho tác giả bài lab.
