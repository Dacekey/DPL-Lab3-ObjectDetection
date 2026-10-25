# Lab 3 — Training trên Kaggle

Hai notebook chạy độc lập, đã nhúng toàn bộ code cần thiết:

- `train_fasterrcnn_kaggle.ipynb`: Faster R-CNN ResNet50-FPN, Torchvision, COCO-pretrained.
- `train_yolov8s_kaggle.ipynb`: YOLOv8s, Ultralytics 8.3.228, COCO-pretrained.

Trong Kaggle: import notebook, chọn **GPU T4 ×2**, bật **Internet**, attach dataset
đã tạo từ `varroa_kaggle_ready.zip`, rồi **Run All**. Không cần upload các file Python
trong repository. Notebook tự tìm dataset dưới `/kaggle/input/`; nếu attach nhiều bộ
Varroa, đặt `DATASET_ROOT` ở cell cấu hình.

Mặc định: seed 42, global batch 8 (4/GPU), ảnh 640 × 640, tối đa 100 epochs,
early stopping sau 15 epochs không cải thiện COCO mAP50:95 validation. Cả hai dùng
cùng Dataset adapter, letterbox, augmentation và evaluator `pycocotools`.
Evaluation batch mặc định 4. DataLoader dùng `spawn` khi có worker để chạy cùng NCCL.
Faster R-CNN dùng SGD lr=0.005; YOLO dùng SGD lr=0.01 và EMA của Ultralytics.
Faster R-CNN chạy FP32 (`AMP=False`) sau khi run FP16 gặp loss NaN/Inf ở epoch 1,
step 841; YOLO giữ AMP. Log cũ chưa xác định phép tính gây lỗi. FP32 có thể tốn
thêm VRAM/thời gian; GPU preflight vẫn kiểm tra đúng cấu hình trước run chính.
Nếu loss hoặc gradient FP32 không hữu hạn, trainer dừng trước optimizer update,
lưu `nonfinite_rank*.json` (loss, learning rate, image ID và box của mỗi rank),
và đóng process group ngay cả khi gặp exception. Không bỏ qua batch lỗi.
Recipe khác nhau được ghi lại trong artifact; không coi đây là phép cô lập riêng kiến trúc.

Sau training, notebook chọn confidence trên validation, đánh giá test bằng checkpoint tốt nhất,
và xuất `comparison.csv`, `metrics.json`, predictions COCO, hình minh họa, checkpoint,
cấu hình, phiên bản thư viện và ZIP kết quả. Tốc độ được đo trong process riêng trên
một GPU, FP32, batch 1, cùng phạm vi đo.
`comparison.csv` chỉ có 7 metric: mAP50:95, AP50, Precision, Recall, F1, Mean IoU TP,
FPS (kèm tên model và seed). Các chỉ số bổ sung nằm trong `metrics.json`.

Trước run chính, notebook tự chạy GPU preflight: 8 ảnh train và 4 ảnh validation,
với đúng kích thước ảnh, số GPU, AMP, worker và batch đã chọn. Output nằm riêng trong
runtime, không đọc test và không ảnh hưởng pretrained weights của run chính.
Nếu bước này thất bại, sửa lỗi hiển thị trước khi chạy training đầy đủ.

Để resume một run bị gián đoạn, attach output cũ, giữ cả `best.pt` và `last.pt`, đặt
`RESUME_CHECKPOINT` thành đường dẫn `last.pt` dưới `/kaggle/input/`, giữ nguyên ngân sách
epochs/seed/kích thước/batch. Notebook sao chép best checkpoint và history vào output mới.
Không resume một run đã hoàn thành; chạy lại phần đánh giá nếu chỉ cần metrics.

Các file `training/*.py` là nguồn để bảo trì; dùng `tools/build_training_notebooks.py`
để tạo lại notebook sau khi sửa nguồn. Hai notebook có cùng bản `common.py`.

Đã kiểm tra schema notebook, cú pháp từng cell, unit checks về tọa độ/metrics/negative samples,
Faster R-CNN forward/backward và checkpoint round-trip, một epoch YOLO trên tập con nhỏ,
resume YOLO sang thư mục mới, và pipeline final evaluation cho cả hai trên CPU.
Kiểm tra bổ sung: 9 unit tests, DataLoader spawn, distributed trên 2 process CPU/Gloo
(YOLO dùng adapter CPU trong test), xuất đúng 7 metric số, và xác nhận Faster R-CNN
không resize lại canvas 640 × 640. Không ép mọi kernel deterministic để giữ CUDA
ROIAlign chuẩn; seed và cuDNN được cố định, không bảo đảm kết quả giống bit.
Các lượt kiểm tra nhỏ dùng ảnh 128 × 128, không dùng để báo cáo chất lượng model.
Chưa chạy full training hoặc xác nhận VRAM/DDP trên 2 GPU T4 trong môi trường local.
