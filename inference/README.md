# Inference Varroa — Faster R-CNN và YOLOv8s

Chạy trên ảnh mới hoặc một thư mục ảnh, không cần annotation. Mặc định dùng hai
`best.pt` đã tải trong `../output/`. Có thể dùng notebook `inference.ipynb` hoặc CLI.

Cài dependency trong môi trường Python đang sử dụng:

```bash
python -m pip install -r inference/requirements.txt
```

Cần cặp torch/torchvision tương thích với nhau và với CPU/CUDA của máy. Các checkpoint
Kaggle gốc dùng torch 2.11.0, torchvision 0.26.0, Ultralytics 8.3.228. Kiểm tra local
trên CPU dùng torch 2.9.1 và torchvision 0.24.1; có thể có sai khác số học nhỏ với T4.

Chạy từ thư mục gốc repo Lab 3 (đổi `--source` thành ảnh/thư mục của bạn):

```bash
python inference/predict.py \
  --source dataset/varroa_ready/images/test \
  --model both --limit 8
```

Chạy riêng từng model:

```bash
python inference/predict.py --source /path/to/image.jpg --model fasterrcnn
python inference/predict.py --source /path/to/images --model yolov8s --device cpu
```

`--device auto` tự chọn CUDA nếu có, ngược lại CPU. `--output` mặc định là `inference/results`.
`--fasterrcnn-weights` và `--yolo-weights` cho phép chọn checkpoint khác. Với checkpoint
tự cung cấp, giữ `config.json` và `confidence_selection.json` tại thư mục run giống
artifact gốc; nếu không có file threshold thì truyền `--confidence` rõ ràng.

Pipeline giữ đúng cách đánh giá: RGB / 255, letterbox 640 × 640 padding 114, FP32,
NMS IoU 0.5, tối đa 100 detection/ảnh, trả box về tọa độ ảnh gốc. Faster R-CNN còn
dùng normalization nội bộ của pretrained backbone. YOLO dùng model/EMA trong checkpoint,
không gọi pipeline resize mặc định khác của `YOLO.predict()`.

Ngưỡng mặc định lấy từ validation, không chọn lại trên ảnh inference:

| Model | Confidence |
|---|---:|
| Faster R-CNN | 0.8224449754 |
| YOLOv8s | 0.2831989825 |

`--confidence 0.5` ghi đè ngưỡng cho các model được chọn. Ngưỡng hợp lệ từ 0.001 đến 1.
Hai model có thang confidence khác nhau; dùng cùng ngưỡng không bảo đảm cùng precision/recall.

Output:

- `<model>/images/`: ảnh gốc vẽ box đỏ và score; giữ cây thư mục nguồn.
- `<model>/predictions.json`: danh sách COCO detection, box `[x, y, width, height]`
  theo pixel ảnh gốc, `category_id=1` là Varroa.
- `<model>/run.json`: checkpoint, ngưỡng, device, kích thước và manifest ánh xạ `image_id`
  sang đường dẫn ảnh. ID bắt đầu từ 1 cho lần inference này, độc lập với COCO dataset ID.
  Ảnh không có detection vẫn có trong manifest.
- `comparison/`: ảnh đặt Faster R-CNN bên trái, YOLOv8s bên phải khi chọn `both`.

Ảnh demo chạy bằng checkpoint thật trên CPU được lưu trong `examples/`.
Đây là dự đoán, không kèm ground truth và không tính lại metrics từ vài ảnh demo.
Chạy lại cùng output sẽ ghi đè file trùng tên; dùng output mới để tách các lần chạy.
