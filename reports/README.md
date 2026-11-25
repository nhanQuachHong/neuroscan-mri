# Báo cáo huấn luyện & chia dữ liệu (BRISC 2025)

Thư mục này lưu giữ nguyên vẹn các file kết quả phi trọng số (JSON, CSV, log, biểu đồ) được nhóm bàn giao để phục vụ đối chiếu, so sánh và viết báo cáo đồ án. Các file trọng số `.pth` không đưa vào Git.

## 1. Dữ liệu & chia tập (`data_split/`)

Nguồn: `son_outputs/` (kiểm tra trùng lặp, lỗi ảnh và chia tập chính thức trên 6.000 ảnh BRISC 2025).

- `split.csv`: 5.830 ảnh sau khi loại trùng lặp (**train = 4.118**, **val = 727**, **test = 985**).
- `class_distribution.csv`, `class_distribution.png`: phân bố 4 lớp (`glioma`, `meningioma`, `no_tumor`, `pituitary`).
- `duplicates.csv`: 170 dòng trùng lặp đã phát hiện.
- `errors.csv`: 0 ảnh lỗi.
- `image_stats.csv`: thống kê kích thước/kênh ảnh.

## 2. Kết quả huấn luyện & đối chiếu

Các mô hình đều được đánh giá trên đúng cùng 985 ảnh Test từ `split.csv`:

### Bản đang gắn trong ứng dụng (`models/*/`)

| Mô hình | Chế độ train | Thư mục báo cáo | Best / Tổng epoch | Test Accuracy | Precision Macro | Recall Macro | F1 Macro | Train (phút) | Latency (batch=1, T4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **EfficientNetB0** | Pretrained ImageNet | `pretrained/efficientnet_b0/` | 20 / 25 | **97,06%** | **96,16%** | **97,51%** | **96,72%** | 14,4 | 8,22 ms |
| **ResNet50** | Pretrained ImageNet | `pretrained/resnet50/` | 9 / 14 | 95,43% | 93,98% | 96,07% | 94,65% | 7,0 | 6,68 ms |
| **Swin-Tiny** (`timm`) | From scratch | `from_scratch/swin_tiny/` | 4 / 14 | 70,46% | 71,63% | 72,31% | 71,20% | 9,4 | 12,99 ms |

### So sánh thêm bản From Scratch (`from_scratch/`)

| Mô hình | Thư mục báo cáo | Epoch tốt nhất / Tổng | Test Accuracy | Precision Macro | Recall Macro | F1 Macro | Tham số | Latency (batch=1, T4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **EfficientNetB0 (from scratch)** | `from_scratch/efficientnet_b0/` | 46 / 56 | 95,23% | 93,91% | 95,86% | 94,59% | 4,01M | 11,84 ms |
| **ResNet50 (from scratch)** | `from_scratch/resnet50/` | 15 / 15 | 90,05% | 88,79% | 91,34% | 89,44% | 23,52M | 5,84 ms |

> [!NOTE]
> Trong `from_scratch/resnet50/train_log.txt`, lượt chạy đầu tiên 60 epoch đạt Test Accuracy **95,13%** và Macro F1 **94,96%** (best epoch 53), nhưng sau đó bị ghi đè bởi lượt chạy 15 epoch (`resnet50_best.pth` và `resnet50_metrics.json` hiện phản ánh lượt 15 epoch: Accuracy 90,05%, Macro F1 89,44%). Kết quả lượt ResNet50 pretrained trước đó (Accuracy 95,43%, Macro F1 94,65%) vẫn được giữ tại `notebooks/resnet50_metrics.original.json`.
