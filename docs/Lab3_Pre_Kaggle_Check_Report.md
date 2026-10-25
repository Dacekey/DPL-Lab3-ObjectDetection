# Lab 3 — Kiểm tra trước khi upload Kaggle

Ngày kiểm tra: 23/10/2026.

**Kết luận:** các kiểm tra local bên dưới đã đạt sau khi sửa lỗi. Dataset ZIP giữ nguyên;
hai notebook đã được tạo lại từ implementation mới nhất. Có thể upload dataset và
hai notebook hiện tại. CUDA/NCCL, AMP và VRAM thực tế trên 2 T4 cần được kiểm chứng
bởi bước GPU preflight tự động trong từng notebook; chưa chạy full training local.

## 1. Dataset và file upload

| Kiểm tra | Kết quả |
|---|---|
| Giải mã toàn bộ ảnh, kích thước, hash ảnh | Đạt: 13.507 ảnh, đều 160 × 280 |
| COCO references, ID, class, diện tích và tọa độ box | Đạt: 4.626 box hợp lệ |
| Nhãn YOLO so với COCO | Đạt: sai số chuyển ngược dưới 1e-6 pixel |
| Ảnh âm tính, file nhãn rỗng | Đạt |
| Thiếu/thừa ảnh hoặc nhãn | Không phát hiện |
| Ảnh giống hệt pixel giữa các split | Không phát hiện |
| ZIP SHA-256 và CRC tất cả member | Đạt |
| SHA256SUMS của mọi file trong ZIP | Đạt |
| Nội dung ZIP so với thư mục dataset đã chuẩn bị | Khớp toàn bộ |

| Split | Ảnh | Box |
|---|---:|---:|
| Train | 8.223 | 2.903 |
| Val | 1.876 | 655 |
| Test | 3.408 | 1.068 |

File upload: `dataset/varroa_kaggle_ready.zip`, 1.159.382.243 bytes.

SHA-256:

```text
e29793c9e58035687c544b43aa293edd5031732daad7d6277dffea5175bd67b0
```

## 2. Notebook và implementation

- Cả hai notebook hợp lệ theo schema nbformat; cú pháp mọi code cell hợp lệ.
- Mã Python nhúng trong notebook khớp chính xác `training/common.py` và trainer tương ứng.
- Đã chạy cell tìm dataset và tạo YAML bằng dataset thực; đường dẫn, split, tên class đúng.
- Faster R-CNN dùng ResNet50-FPN COCO_V1, head background + Varroa; YOLO dùng YOLOv8s
  COCO-pretrained với một class. Ultralytics được pin 8.3.228; pycocotools pin 2.0.11.
- Giữ torch/torchvision do Kaggle cung cấp, ghi phiên bản thực vào artifact.
- Cùng Dataset adapter, letterbox, augmentation, split, seed 42 và evaluator COCO.
- Validation chọn checkpoint; confidence chọn trên validation rồi khóa trước test.
- Cùng score floor 0.001, NMS IoU 0.5, max detections 100, tọa độ ảnh gốc khi tính metric.
- Kiểm tra canvas Faster R-CNN: input 640 × 640 vẫn là 640 × 640 sau transform nội bộ;
  box không bị resize lần hai.

## 3. Vấn đề đã sửa trong lượt kiểm tra

1. **CSV còn nhiều metric:** `comparison.csv` được rút đúng 7 metric đã chốt:
   `mAP50_95`, `AP50`, `precision`, `recall`, `F1`, `mean_IoU_TP`, `FPS`.
   Cột tên model và seed được giữ để ghép hai run. Chi tiết bổ sung ở `metrics.json`.
2. **Precision bị ghi đè bởi FP32:** metadata benchmark trước đây dùng cùng key
   `precision` với metric Precision. Đã đổi metadata sang `inference_precision`,
   lấy riêng các metric từ kết quả test và FPS từ benchmark; có regression test bảo vệ.
3. **Worker với NCCL:** mọi DataLoader có worker dùng `spawn`, gồm train và inference.
   Đã chạy DataLoader spawn trên dữ liệu thực.
4. **ROIAlign:** không ép tất cả kernel deterministic, tránh Torchvision chuyển CUDA
   ROIAlign sang implementation compile khác. Vẫn cố định seed và cuDNN; không hứa
   kết quả giống bit giữa môi trường hoặc sau resume.
5. **AMP và scheduler Faster R-CNN:** chỉ tiến scheduler khi GradScaler không bỏ qua
   optimizer step do gradient overflow.
6. **Bảo vệ run và resume:** chặn run mới nếu output đã có best hoặc last checkpoint;
   sao chép best/history khi resume. Lưu source theo file, tránh copy theo symlink
   toàn bộ dataset của smoke test vào artifact khi chạy lại.
7. **Kiểm chứng trên Kaggle:** thêm GPU preflight trước training đầy đủ, dùng một batch
   train (mặc định 8 ảnh) và 4 ảnh validation, không sử dụng test. Cùng số GPU, kích thước
   640, AMP và worker như run chính. Output và config riêng; main run khởi tạo lại từ
   pretrained weights. Evaluation batch mặc định giảm còn 4.

## 4. Các lượt chạy đã đạt

- **9 unit tests** về geometry, ghép detection, ảnh âm tính, predictions rỗng, oracle AP,
  chọn confidence, metric CSV, worker spawn và GPU-preflight subset.
- Faster R-CNN: forward/backward với batch có ảnh dương/âm và batch toàn ảnh âm;
  checkpoint round-trip cho predictions khớp; final evaluation với worker spawn;
  CSV xuất đúng 7 metric, Precision là số.
- Faster R-CNN: hai process CPU/Gloo, hai bước optimizer, kiểm tra gradient và luồng
  rank 0 validation rồi broadcast điều khiển.
- YOLO: một epoch trên subset 4 ảnh train, COCO validation, lưu/nạp best checkpoint,
  final evaluation và CSV 7 metric; batch toàn ảnh âm chạy backward được.
- YOLO: gián đoạn sau epoch đầu, resume sang thư mục mới, chạy tiếp epoch kế tiếp.
- YOLO: hai process CPU/Gloo với adapter đổi phần device/DDP sang CPU trong harness
  kiểm thử; train, COCO validation và lưu best checkpoint thành công.

Môi trường kiểm thử model: Python 3.14, torch 2.9.1+cpu, torchvision 0.24.1+cpu,
Ultralytics 8.3.228. Subset chạy model dùng kích thước 128 để kiểm tra pipeline;
các điểm AP/FPS của subset **không phải kết quả thực nghiệm để đưa vào báo cáo lab**.

## 5. Phần chỉ xác nhận được trên Kaggle

Local không có GPU CUDA. Vì vậy chưa xác nhận được NCCL, CUDA detection kernels,
AMP và peak VRAM trên phần cứng T4 thực tế. GPU preflight sẽ thử những phần này
trước khi bắt đầu training dài. Nó kiểm tra khả năng chạy, không dự đoán accuracy
hay bảo đảm mọi epoch sau đều có cùng mức dùng VRAM.

Thiết lập Kaggle: **GPU T4 ×2**, **Internet bật**, attach dataset từ ZIP trên,
import notebook mới nhất và **Run All**. Không cần upload các file `.py` trong repo.

Nguồn đối chiếu: [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html),
[Torchvision ROIAlign source](https://github.com/pytorch/vision/blob/v0.24.1/torchvision/ops/roi_align.py),
[Ultralytics training](https://docs.ultralytics.com/modes/train/).
