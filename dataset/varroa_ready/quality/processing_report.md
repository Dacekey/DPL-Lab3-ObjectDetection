# Lab 3 — Báo cáo chuẩn bị dataset dùng chung cho Faster R-CNN và YOLOv8s

Thời điểm build (UTC): 2026-10-23T03:46:10.259730+00:00.

## Kết quả bàn giao

- File cần upload lên Kaggle: `labs/Lab-3_ObjectDetection/dataset/varroa_kaggle_ready.zip`.
- Thư mục đã giải nén: `labs/Lab-3_ObjectDetection/dataset/varroa_ready/`.
- Một bộ ảnh dùng chung, hai định dạng nhãn: YOLO TXT và COCO JSON.
- 13.507 ảnh RGB 160 × 280, 4.626 bounding boxes hợp lệ, một class `varroa`.
- Chốt **một seed huấn luyện = 42** cho bước training sau. Bước chuẩn bị dữ liệu
  không chia ngẫu nhiên, không augmentation và không phụ thuộc seed.
- Chưa train model; các kết quả kiểm tra format không phải kết quả detection thực nghiệm.

| Split | Ảnh | Có Varroa | Không có box | Bounding boxes |
|---|---:|---:|---:|---:|
| train | 8,223 | 2,553 | 5,670 | 2,903 |
| val | 1,876 | 451 | 1,425 | 655 |
| test | 3,408 | 942 | 2,466 | 1,068 |

## 1. Nguồn và kiểm tra archive

Nguồn: [VarroaDataset v1.2.0](https://doi.org/10.5281/zenodo.4085044), tác giả
Schurischuster Stefan và Martin Kampel; license CC BY 4.0.
[README của tác giả](https://github.com/schurist/VarroaDataset/) mô tả ảnh, split và tọa độ box.

Ba file `.zip` ban đầu có dung lượng 0 byte nên không được dùng. Ba file `.zip.part`
thực tế là archive ZIP đọc được; đã kiểm tra CRC toàn bộ member và giải mã tất cả ảnh
thành công. File nguồn được giữ nguyên.

| Archive nguồn | Bytes | SHA-256 |
|---|---:|---|
| `train.psFXXPr5.zip.part` | 703,410,525 | `5e3ff890f26776500f1c0780c78ced639853f5f27f5a5da3471640ed540ccc67` |
| `val.BE6npwlB.zip.part` | 163,050,592 | `78e7b73422514f2a59bbf8ad411e3a920522bdbc5e8bfd95198dbbba6c6d7bfd` |
| `test.2fv_ouGd.zip.part` | 291,715,596 | `e04f0ec0888b239a65241b29988fa3c31fdef3c053a56f912c8b24b8b2a2c57f` |

`gt.csv` toàn cục được đối chiếu với `gt_one.csv` từng split. Nhãn TXT dưới `labels/`
là nguồn tọa độ chính vì giữ độ chính xác thập phân; khi làm tròn, phải khớp một dòng CSV
tương ứng. Mỗi ảnh nguồn phải tồn tại, có annotation, giải mã được và không thiếu kích thước.

## 2. Xác định bài toán và ánh xạ nhãn

- Bài toán: phát hiện box Varroa trên ảnh crop con ong, một class foreground.
- Trường đầu của nhãn nguồn là flag ở cấp ảnh, không phải số lượng box hay class YOLO.
- Flag `0`: ảnh không có box, giữ làm negative sample.
- Flag `1` và `3`: giữ các box và gộp vào class detection `varroa`.
- YOLO: `varroa=0`. COCO/Torchvision: `varroa=1`, background Torchvision bằng 0.
- Không sinh box cho con ong khỏe và không tạo class `healthy`.

Đây là quyết định ánh xạ cho bài toán một class, dựa trên cấu trúc annotation và
kiểm tra trực quan mẫu box của cả flag 1 và 3. Tài liệu gốc chưa giải thích rõ sự khác nhau
giữa hai flag; không khẳng định ý nghĩa chi tiết đã được tác giả xác nhận. Flag nguồn
được giữ trong `manifest.csv` để truy vết. Không tự gán nhãn mới từ nội dung ảnh.

## 3. Làm sạch annotation

Train có 8.225 dòng CSV nhưng chỉ 8.223 đường dẫn ảnh khác nhau:

1. `2017-08-30_15-42-59.mp4-bee_id_10000-54690-1.png`: hai dòng có box hơi khác nhau.
   Chọn nhãn chính trong `train/labels/`, khớp dòng CSV đầu tiên, giữ một box thập phân
   `[94.4767685851, 75.7110311751, 120.495953237, 118.097122302]`.
   Không hợp nhất hai biến thể thành hai object vì chúng chỉ cùng một Varroa.
   Biến thể trong thư mục `videos/` được ghi lại trong audit.
2. `2017-08-30_15-42-59.mp4-bee_id_10001-54690-1.png`: hai dòng âm tính giống nhau,
   giữ ảnh và nhãn rỗng một lần.

Phát hiện một box suy biến trong train:

```text
Ảnh: 2017-08-30_15-42-59.mp4-bee_id_11104-105630-1.png
Box bị loại: [-0.5, 119.5, -0.5, 125.928571429]
Lý do: x1 = x2, chiều rộng bằng 0.
```

Ảnh này vẫn được giữ với box hợp lệ còn lại; không chuyển ảnh có Varroa thành ảnh âm tính.
Không có box hợp lệ nào cần clip biên; không có box giống hệt nhau trong một ảnh cần loại.
Sau xử lý, train có 2.903 box thay vì 2.905 bộ tọa độ trong CSV gốc (một biến thể
của dòng lặp và một box suy biến đã bị loại). Val/test giữ nguyên số ảnh và box.

## 4. Chuẩn hóa và xuất dữ liệu

- Giữ byte PNG gốc; không resize, nén lại, crop, cân bằng lớp hoặc tạo ảnh augmentation.
- Giữ train/val/test gốc; không trộn rồi chia ngẫu nhiên các frame video.
- Xuất ảnh ở `images/<split>/<video>/*.png`, tránh đụng tên giữa video.
- Xuất YOLO `labels/<split>/<video>/*.txt`: `0 xc/W yc/H w/W h/H`, 12 chữ số thập phân.
  Ảnh âm tính có file TXT rỗng.
- Xuất COCO `annotations/instances_<split>.json`: box pixel `[x,y,w,h]`,
  `category_id=1`, `area=w*h`, `iscrowd=0`; ảnh âm tính vẫn có image entry.
- Mỗi ảnh và annotation có ID duy nhất toàn bộ dataset; file_name tương đối với dataset root.
- Xuất `data.yaml` dùng đường dẫn tương đối, không gắn đường dẫn máy local.
- `classes.json` chứa mapping; `manifest.csv` chứa source path, flag, kích thước, số box,
  SHA-256 file, SHA-256 pixel RGB và dHash.
- Resize/letterbox 640 × 640 và augmentation sẽ thực hiện trong pipeline train sau;
  cần cấu hình Faster R-CNN tránh resize nội bộ lần nữa.

## 5. Kiểm tra trùng dữ liệu và split

- Không có nhóm video trùng tên giữa train/val/test.
- Không có ảnh trùng pixel RGB trong toàn bộ dataset (kể cả trong cùng split).
- Không có cặp cross-split dHash 64-bit với Hamming distance ≤ 3.
- Kiểm tra dHash là phép sàng lọc ảnh gần giống, không chứng minh mọi frame tương quan
  hoặc cùng phiên quay đã được loại. Ví dụ một số tên phiên train/val gần nhau;
  split được giữ theo dataset gốc, không tuyên bố độc lập hoàn toàn ở cấp phiên quay.

## 6. Kiểm tra chất lượng đầu ra

Đã chạy validator độc lập trên toàn bộ output:

- Giải mã lại toàn bộ 13.507 ảnh; xác nhận kích thước và SHA-256 khớp nguồn.
- Kiểm tra ID, tham chiếu COCO, class, diện tích, tọa độ hữu hạn, box dương và nằm trong ảnh.
- Chuyển ngược mọi box YOLO về tọa độ COCO; sai số phải nhỏ hơn `1e-6` pixel.
- Kiểm tra ảnh âm tính có nhãn rỗng và vẫn có mặt trong COCO.
- Kiểm tra không thiếu hoặc thừa ảnh/label so với annotation và manifest.
- Nạp bằng `pycocotools` và chạy COCOeval với box ground truth làm oracle prediction;
  AP50, AP75 và AP50:95 bằng 1 ở cả ba split. Đây chỉ là kiểm tra format/evaluator.
- Tạo contact sheet có box để kiểm tra trực quan: `quality/train_annotations.jpg`,
  `quality/val_annotations.jpg`, `quality/test_annotations.jpg`.
- File ZIP bàn giao được kiểm tra CRC sau khi đóng gói; checksum riêng nằm cạnh ZIP.

Kết quả máy đọc được: `quality/audit.json`, `quality/validation.json`,
`quality/coco_self_check.json` và `SHA256SUMS` trong gói dataset.

Theo diện tích trên ảnh gốc, toàn bộ box thuộc nhóm COCO small hoặc medium, không có large.
AP_large sẽ không có dữ liệu, không diễn giải giá trị -1 của COCOeval là điểm model.

## 7. Cách sử dụng trên Kaggle

Bạn chỉ cần upload **`varroa_kaggle_ready.zip`** thành một Kaggle Dataset rồi attach
vào notebook huấn luyện. Sau khi Kaggle giải nén:

- YOLOv8s đọc `data.yaml` bằng đường dẫn tuyệt đối dưới `/kaggle/input/`.
- Faster R-CNN đọc `annotations/instances_train.json`, `instances_val.json`,
  `instances_test.json` với cùng dataset root. Dataset adapter trong code train sẽ
  đổi COCO xywh sang Torchvision xyxy và trả tensor ảnh [0,1], label int64;
  đây là thao tác nạp dữ liệu, không cần chuẩn bị lại dataset.
- Dùng cùng COCO ground truth để đánh giá hai model; map YOLO class 0 sang COCO class 1.
- Huấn luyện một seed 42; chọn checkpoint/ngưỡng confidence trên val, đánh giá cuối trên test.

## 8. Khả năng tái tạo

Các script trong thư mục `tools/`:

- `prepare_dataset.py`: tạo dataset từ archive nguồn; từ chối ghi đè output đã tồn tại.
- `validate_dataset.py`: kiểm tra độc lập ảnh, manifest, COCO và nhãn YOLO.
- `write_dataset_docs.py`: tạo README và báo cáo từ audit thực tế.

Script chuẩn bị và validator dùng Python + Pillow. Kiểm tra COCOeval bổ sung dùng
`pycocotools`; không cần GPU và không chạy training.
