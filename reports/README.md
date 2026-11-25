# Báo cáo huấn luyện & chia dữ liệu (BRISC 2025)

Thư mục này lưu giữ nguyên vẹn các file kết quả phi trọng số (JSON, CSV, log, biểu đồ) được nhóm bàn giao để phục vụ đối chiếu, so sánh và viết báo cáo đồ án. Các file trọng số `.pth` không đưa vào Git.

## 1. Dữ liệu & chia tập (`data_split/`)

Nguồn: `son_outputs/` (kiểm tra trùng lặp, lỗi ảnh và chia tập chính thức trên 6.000 ảnh BRISC 2025).

- `split.csv`: 5.830 ảnh sau khi loại trùng lặp (**train = 4.118**, **val = 727**, **test = 985**).
- `class_distribution.csv`, `class_distribution.png`: phân bố 4 lớp (`glioma`, `meningioma`, `no_tumor`, `pituitary`).
- `duplicates.csv`: 170 dòng trùng lặp đã phát hiện.
- `errors.csv`: 0 ảnh lỗi.
- `image_stats.csv`: thống kê kích thước/kênh ảnh.

## 2. Kết quả huấn luyện from scratch (`from_scratch/`)

Cả 3 mô hình đều được huấn luyện từ đầu (`no_pretrained: true`) trên đúng cùng `split.csv` (seed 42, ảnh 224×224, `CrossEntropyLoss` có class weights, kiểm thử trên cùng 985 ảnh Test):

| Mô hình | Thư mục app | Epoch tốt nhất / Tổng | Test Accuracy | Precision Macro | Recall Macro | F1 Macro | Tham số | Latency (batch=1, T4) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **EfficientNetB0** | `models/efficientnet_b0` | 46 / 56 | **95,23%** | 93,91% | **95,86%** | **94,59%** | 4,01M | 11,84 ms |
| **ResNet50** | `models/resnet50` | 15 / 15 | 90,05% | 88,79% | 91,34% | 89,44% | 23,52M | 5,84 ms |
| **Swin-Tiny** (`timm`) | `models/swin` | 4 / 14 | 70,46% | 71,63% | 72,31% | 71,20% | 27,52M | 12,99 ms |

> [!NOTE]
> Trong `from_scratch/resnet50/train_log.txt`, lượt chạy đầu tiên 60 epoch đạt Test Accuracy **95,13%** và Macro F1 **94,96%** (best epoch 53), nhưng sau đó bị ghi đè bởi lượt chạy 15 epoch (`resnet50_best.pth` và `resnet50_metrics.json` hiện phản ánh lượt 15 epoch: Accuracy 90,05%, Macro F1 89,44%). Kết quả lượt ResNet50 pretrained trước đó (Accuracy 95,43%, Macro F1 94,65%) vẫn được giữ tại `notebooks/resnet50_metrics.original.json`.
