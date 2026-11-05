"""Cấu hình dùng chung cho cả nhóm. KHÔNG tự ý sửa các giá trị ở đây khi train riêng,
nếu cần đổi thì báo cả nhóm để 3 mô hình được so sánh công bằng."""

# Thứ tự lớp cố định: chỉ số 0..3 dùng cho nhãn, checkpoint, confusion matrix và demo.
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SEED = 42

# Cấu hình huấn luyện chung (giữ giống nhau giữa 3 mô hình)
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 32
DEFAULT_PATIENCE = 5          # early stopping theo Val Macro F1
DEFAULT_FREEZE_EPOCHS = 3     # số epoch đầu chỉ train lớp phân loại

# Gợi ý riêng cho từng mô hình (phần đặc thù được phép khác nhau)
MODEL_DEFAULTS = {
    "resnet50":        {"optimizer": "adam",  "lr": 1e-4, "weight_decay": 0.0,  "scheduler": "cosine", "warmup_epochs": 0},
    "efficientnet_b0": {"optimizer": "adam",  "lr": 3e-4, "weight_decay": 0.0,  "scheduler": "cosine", "warmup_epochs": 0},
    "swin_tiny":       {"optimizer": "adamw", "lr": 1e-4, "weight_decay": 0.05, "scheduler": "cosine", "warmup_epochs": 2},
}

# Tên file đầu ra thống nhất (demo và module đánh giá đọc theo tên này)
CHECKPOINT_NAME = "{model}_best.pth"
HISTORY_NAME = "{model}_history.json"
METRICS_NAME = "{model}_metrics.json"
PREDICTIONS_NAME = "{model}_test_predictions.csv"
