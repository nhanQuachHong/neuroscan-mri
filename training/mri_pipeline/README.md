# Pipeline tiền xử lý và khung huấn luyện – Phân loại MRI não (BRISC 2025)

Phụ trách: **Lâm Minh Trí** (phần 2). Cả nhóm dùng chung bộ code này để 3 mô hình được huấn luyện
trên **cùng dữ liệu, cùng tiền xử lý, cùng cấu hình**, nhờ đó so sánh mới công bằng.

## 1. Cấu trúc thư mục

| File | Chức năng |
|---|---|
| `config.py` | Thứ tự lớp, kích thước ảnh, mean/std ImageNet, seed, cấu hình chung và gợi ý lr từng mô hình |
| `split_utils.py` | Đọc và kiểm tra `split.csv`, kiểm tra rò rỉ dữ liệu, bảng phân bố lớp |
| `transforms.py` | Resize 224×224, chuẩn hóa ImageNet, augmentation **chỉ cho Train** |
| `dataset.py` | `MRIDataset` (ảnh xám → RGB 3 kênh) và `get_loaders()` |
| `class_weights.py` | Tính Class Weight **chỉ trên tập Train** → `class_weights.json` |
| `models.py` | Tạo ResNet50 / EfficientNetB0 / Swin-Tiny với trọng số ImageNet |
| `train.py` | Khung huấn luyện chung: đóng băng → fine-tune, early stopping, checkpoint, đo thời gian, đánh giá Test |
| `check_pipeline.py` | Kiểm tra phân bố, rò rỉ, ảnh lỗi, kích thước batch, xuất ảnh minh họa augmentation |
| `make_temp_split.py` | **Tạm thời**: tự chia dữ liệu để chạy thử khi chưa có `split.csv` của Son |
| `smoke_test.py` | Chạy thử toàn bộ trên ảnh giả trong vài phút |

## 2. Quy ước chung của nhóm (bắt buộc)

**Thứ tự lớp:** `0 glioma, 1 meningioma, 2 no_tumor, 3 pituitary` (dùng cho nhãn, checkpoint, confusion matrix, demo).

**`split.csv`** (Son xuất ra):
```
filepath,label,split
classification_task/train/glioma/brisc2025_train_00001_gl_ax_t1.jpg,glioma,train
```
- `filepath` là đường dẫn **tương đối** so với `--data-root`
- `label` thuộc `glioma | meningioma | no_tumor | pituitary`
- `split` thuộc `train | val | test`

**Cấu hình huấn luyện giữ giống nhau giữa 3 mô hình:** seed 42, ảnh 224×224, batch size 32,
tối đa 30 epoch, 3 epoch đầu đóng băng backbone, early stopping patience 5 theo Val Macro F1,
CrossEntropyLoss có Class Weight, AMP, gradient clipping 1.0.

**Phần được phép khác nhau** (đặc thù từng mô hình, mặc định trong `config.MODEL_DEFAULTS`):

| Mô hình | Người train | Optimizer | lr | weight decay | Scheduler |
|---|---|---|---|---|---|
| `resnet50` | Phúc | Adam | 1e-4 | 0 | cosine |
| `efficientnet_b0` | Nhã Trân | Adam | 3e-4 | 0 | cosine |
| `swin_tiny` | Lợi | AdamW | 1e-4 | 0.05 | warmup 2 epoch + cosine |

**Tên file đầu ra** (trong `outputs/<model>/`):
`<model>_best.pth`, `<model>_history.json`, `<model>_metrics.json`, `<model>_test_predictions.csv`, `train_log.txt`.

## 3. Chạy trên Kaggle (khuyên dùng, dataset có sẵn)

1. Tạo Notebook → **Add Input** → tìm `brisc2025` → Add.
2. Settings: **Accelerator = GPU T4 x2 hoặc P100**, **Internet = On** (để tải trọng số ImageNet).
3. Upload thư mục code (hoặc `git clone` repo nhóm) rồi chạy:

```bash
%cd /kaggle/working/mri_pipeline
!pip install -q timm

# (1) Kiểm tra code chạy được (2-5 phút)
!python smoke_test.py

# (2) Khi chưa có split.csv của Son: tạo split tạm để chạy thử
!python make_temp_split.py --data-root /kaggle/input/brisc2025 --out split_temp.csv

# (3) Kiểm tra pipeline trên dữ liệu thật
!python check_pipeline.py --split-csv split.csv --data-root /kaggle/input/brisc2025 --out-dir pipeline_check
!python class_weights.py  --split-csv split.csv --out class_weights.json

# (4) Huấn luyện (mỗi bạn chạy mô hình của mình)
!python train.py --model resnet50        --split-csv split.csv --data-root /kaggle/input/brisc2025 --out-dir /kaggle/working/outputs
!python train.py --model efficientnet_b0 --split-csv split.csv --data-root /kaggle/input/brisc2025 --out-dir /kaggle/working/outputs
!python train.py --model swin_tiny       --split-csv split.csv --data-root /kaggle/input/brisc2025 --out-dir /kaggle/working/outputs
```

Trên **Colab**: tải dataset bằng `kagglehub.dataset_download("briscdataset/brisc2025")`, lấy đường dẫn
trả về làm `--data-root`, lưu `--out-dir` vào Google Drive để không mất checkpoint khi bị ngắt.

Các tham số hay dùng của `train.py`: `--batch-size 16` (khi thiếu VRAM, nhất là Swin), `--lr`,
`--weight-decay`, `--epochs`, `--class-weights none` (tắt Class Weight), `--num-workers 2`.
Xem đầy đủ: `python train.py -h`.

## 4. Gợi ý cho từng thành viên

- **Son (dữ liệu + đánh giá):** xuất `split.csv` đúng định dạng ở mục 2. Module đánh giá có thể đọc
  `<model>_test_predictions.csv` (cột `filepath,label,pred,prob_glioma,prob_meningioma,prob_no_tumor,prob_pituitary`)
  và `<model>_metrics.json` để vẽ Confusion Matrix và bảng so sánh.
- **Phúc / Nhã Trân / Lợi (train mô hình):** chỉ cần chạy `train.py --model ...`. Muốn thử siêu tham số thì
  đổi `--lr`, `--weight-decay` và đặt `--out-dir` khác nhau cho từng lần thử. **Không** sửa `transforms.py`,
  `config.py` khi chưa báo nhóm.
- **Nhân (demo):** load checkpoint như sau, dùng đúng `get_eval_transforms()`:

```python
import torch
from models import build_model
from transforms import get_eval_transforms
ckpt = torch.load("resnet50_best.pth", map_location="cpu", weights_only=False)
model, _ = build_model(ckpt["model_name"], pretrained=False)
model.load_state_dict(ckpt["state_dict"]); model.eval()
x = get_eval_transforms()(pil_image.convert("RGB")).unsqueeze(0)
probs = torch.softmax(model(x), 1)[0]          # thứ tự lớp = ckpt["classes"]
```

## 5. Ghi chú kỹ thuật

- Augmentation trên Train: RandomResizedCrop (80–100%), lật ngang, xoay ±15°, dịch nhẹ 5%,
  chỉnh sáng/tương phản ±20%. Không lật dọc. Val/Test chỉ resize và chuẩn hóa.
- BRISC có sẵn tập test chính thức 1.000 ảnh. `make_temp_split.py` giữ nguyên tập test này và chia
  tập train chính thức thành train/val 85/15. Nhóm nên thống nhất với Son có dùng cách này cho bản chính thức không.
- BRISC không có mã bệnh nhân nên không chia theo bệnh nhân được; việc loại ảnh trùng (Son) càng quan trọng.
- Checkpoint tốt nhất chọn theo **Val Macro F1**; Test chỉ được chạy **một lần** ở cuối bằng checkpoint đó.
- Thời gian dự đoán được đo ở hai kiểu: theo batch và 1 ảnh/lần (`inference_ms_per_image_single`, phù hợp để so sánh cho demo).
