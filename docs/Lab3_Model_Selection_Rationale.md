# Lab 3 – Lý do chọn Faster R-CNN ResNet50-FPN và YOLOv8s

## 1. Bối cảnh bài toán

Lab 3 yêu cầu:

> **So sánh và nhận xét kết quả của họ R-CNN và YOLO trên bộ dữ liệu Varroa cho bài toán Object Detection. Có thể chọn 1–2 phiên bản của mỗi họ.**

Để việc so sánh có ý nghĩa, hai mô hình được chọn nên:

- Đại diện rõ ràng cho hai hướng tiếp cận khác nhau trong Object Detection.
- Có khả năng fine-tune trên dataset riêng.
- Có implementation ổn định và dễ reproduce.
- Không quá cũ để kết quả bị lệch quá nhiều do khác biệt thế hệ.
- Có thể đánh giá trên cùng các metric như mAP, Precision, Recall và tốc độ inference.

Từ các tiêu chí trên, hai mô hình được lựa chọn là:

- **Faster R-CNN ResNet50-FPN**
- **YOLOv8s**

---

## 2. Vì sao chọn Faster R-CNN ResNet50-FPN?

### 2.1. Đại diện điển hình cho họ R-CNN

Faster R-CNN là một trong những kiến trúc tiêu biểu nhất của nhóm **two-stage object detectors**.

Pipeline cơ bản:

```text
Image
  ↓
Backbone
  ↓
Feature Maps
  ↓
Region Proposal Network (RPN)
  ↓
Region Proposals
  ↓
RoI Features
  ↓
Classification + Bounding Box Regression
```

So với R-CNN và Fast R-CNN, Faster R-CNN đã thay bước tạo region proposal bằng một **Region Proposal Network (RPN)** có thể học được, vì vậy phù hợp hơn để làm mô hình đại diện cho họ R-CNN trong một bài lab hiện đại.

### 2.2. ResNet50 là backbone phổ biến và đủ mạnh

ResNet50 cung cấp khả năng trích xuất đặc trưng tốt nhưng vẫn có chi phí tính toán hợp lý hơn các backbone lớn hơn.

Điều này giúp mô hình:

- Có khả năng học đặc trưng đủ tốt trên dataset Varroa.
- Không quá nặng cho mục tiêu thực nghiệm trong Lab 3.
- Dễ sử dụng pretrained weights để fine-tune.

### 2.3. FPN phù hợp với object có kích thước nhỏ

FPN (**Feature Pyramid Network**) kết hợp feature maps ở nhiều mức độ phân giải khác nhau.

Điều này đặc biệt đáng quan tâm với dataset Varroa vì đối tượng Varroa có thể chiếm diện tích nhỏ trong ảnh.

FPN giúp mô hình khai thác cả:

- Feature map có độ phân giải cao cho object nhỏ.
- Feature map sâu hơn cho thông tin ngữ nghĩa mạnh hơn.

Do đó, Faster R-CNN ResNet50-FPN là một baseline hợp lý để đánh giá khả năng phát hiện các object nhỏ.

### 2.4. Dễ triển khai và reproduce

Faster R-CNN ResNet50-FPN có implementation phổ biến trong các framework như PyTorch/Torchvision và có pretrained weights trên COCO.

Điều này giúp giảm thời gian triển khai và tập trung vào mục tiêu chính của Lab 3 là:

> **so sánh đặc điểm và hiệu năng giữa họ R-CNN và YOLO.**

---

## 3. Vì sao chọn YOLOv8s?

### 3.1. Đại diện cho hướng one-stage detector

YOLOv8 là một mô hình thuộc nhóm **one-stage object detector**.

Khác với Faster R-CNN, YOLO không cần một stage riêng để tạo region proposals trước khi thực hiện detection.

Pipeline có thể hiểu đơn giản:

```text
Image
  ↓
Backbone
  ↓
Multi-scale Features
  ↓
Detection Head
  ↓
Bounding Boxes + Classes
```

Việc chọn YOLOv8s vì vậy tạo ra một phép so sánh rõ ràng:

```text
Faster R-CNN
→ Two-stage detector

YOLOv8s
→ One-stage detector
```

Đây cũng là điểm chính để phân tích sự khác nhau về:

- Accuracy.
- Recall.
- Localization.
- Inference speed.
- Computational cost.

### 3.2. YOLOv8 có pipeline huấn luyện thuận tiện

YOLOv8 có hệ sinh thái huấn luyện và inference tương đối hoàn chỉnh.

Model có thể:

- Fine-tune trên custom dataset.
- Đánh giá mAP, Precision, Recall.
- Inference trực tiếp trên ảnh test.
- Xuất kết quả prediction thuận tiện cho việc so sánh trực quan.

Điều này phù hợp với một bài lab yêu cầu tập trung vào kết quả thực nghiệm hơn là tự triển khai toàn bộ detector từ đầu.

### 3.3. Chọn phiên bản `s` thay vì `n`, `m`, `l` hoặc `x`

YOLOv8 có nhiều kích thước model.

Việc chọn **YOLOv8s** là một điểm cân bằng giữa:

- Độ chính xác.
- Số lượng parameters.
- Tốc độ inference.
- Chi phí huấn luyện.

`YOLOv8n` nhẹ hơn nhưng có thể làm giảm khả năng detection, trong khi `m`, `l` và `x` yêu cầu nhiều tài nguyên hơn.

Vì vậy, `YOLOv8s` phù hợp hơn cho mục tiêu so sánh trong Lab 3.

---

## 4. Vì sao cặp model này phù hợp để so sánh?

Hai model thể hiện hai triết lý detection khác nhau:

| Đặc điểm | Faster R-CNN ResNet50-FPN | YOLOv8s |
|---|---|---|
| Detector type | Two-stage | One-stage |
| Region proposal | Có RPN | Không |
| Detection pipeline | Proposal → classify/refine | Predict trực tiếp |
| Multi-scale features | FPN | Có multi-scale detection |
| Điểm mạnh kỳ vọng | Accuracy, localization, small objects | Speed, efficiency |
| Tốc độ inference kỳ vọng | Chậm hơn | Nhanh hơn |
| Mức độ triển khai | Trung bình | Dễ |

Do đó, kết quả thực nghiệm có thể trả lời các câu hỏi quan trọng như:

- Faster R-CNN có cho độ chính xác cao hơn trên Varroa hay không?
- YOLOv8s có đánh đổi accuracy để đạt tốc độ inference cao hơn không?
- Mô hình nào có Recall tốt hơn?
- Mô hình nào bỏ sót Varroa nhiều hơn?
- Mô hình nào localization bounding box tốt hơn?
- Khả năng phát hiện Varroa nhỏ của hai mô hình khác nhau như thế nào?

---

## 5. Các metric nên sử dụng

Để so sánh công bằng, nên đánh giá hai mô hình trên cùng một tập test và sử dụng các metric giống nhau.

Các metric chính:

- **Precision**
- **Recall**
- **F1-score**
- **mAP@0.5**
- **mAP@0.5:0.95**
- **Inference time / FPS**
- **Number of parameters**

Trong bài toán Varroa, **Recall** là một metric đặc biệt đáng chú ý vì Recall thấp đồng nghĩa với việc mô hình bỏ sót nhiều Varroa thực sự tồn tại trong ảnh.

---

## 6. Nguyên tắc để so sánh công bằng

Hai mô hình nên sử dụng:

- Cùng dataset Varroa.
- Cùng train/validation/test split.
- Cùng input preprocessing nếu có thể.
- Cùng tập test cuối cùng.
- Cùng cách tính metric.
- Cùng pretrained initialization tương ứng nếu sử dụng transfer learning.

Không nhất thiết phải ép hai mô hình có cùng số parameters hoặc cùng thời gian train, vì đây cũng là một phần của sự khác biệt kiến trúc cần được phân tích.

---

## 7. Kết luận

Hai model được lựa chọn cho Lab 3 là:

```text
R-CNN family:
Faster R-CNN ResNet50-FPN

YOLO family:
YOLOv8s
```

Lý do chính là:

1. **Faster R-CNN ResNet50-FPN** là một đại diện mạnh và phổ biến cho hướng two-stage detection.
2. **YOLOv8s** là một đại diện thực tế, dễ fine-tune và có hiệu năng tốt cho hướng one-stage detection.
3. Hai model có sự khác biệt kiến trúc rõ ràng, giúp việc phân tích kết quả có ý nghĩa.
4. Cả hai đều có pretrained models và implementation ổn định, phù hợp với phạm vi một bài lab.
5. Cặp model này cho phép đánh giá rõ trade-off giữa **accuracy / recall / localization** và **inference speed / computational efficiency** trên dataset Varroa.
