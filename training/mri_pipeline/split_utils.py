"""Đọc và kiểm tra split.csv (không phụ thuộc PyTorch)."""
import pandas as pd

from config import CLASSES

REQUIRED_COLUMNS = {"filepath", "label", "split"}
VALID_SPLITS = {"train", "val", "test"}

# Chấp nhận vài cách viết tên lớp khác nhau rồi đưa về tên chuẩn
LABEL_ALIASES = {
    "glioma": "glioma", "glioma_tumor": "glioma", "gl": "glioma",
    "meningioma": "meningioma", "meningioma_tumor": "meningioma", "me": "meningioma",
    "no_tumor": "no_tumor", "notumor": "no_tumor", "no tumor": "no_tumor",
    "no": "no_tumor", "non_tumorous": "no_tumor", "nt": "no_tumor",
    "pituitary": "pituitary", "pituitary_tumor": "pituitary", "pi": "pituitary",
}


def normalize_label(x: str) -> str:
    key = str(x).strip().lower().replace("-", "_")
    return LABEL_ALIASES.get(key, key)


def read_split(split_csv: str) -> pd.DataFrame:
    df = pd.read_csv(split_csv)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"split.csv thiếu cột {sorted(missing)}. Cần đủ: filepath,label,split")
    df["label"] = df["label"].map(normalize_label)
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    bad_labels = set(df["label"]) - set(CLASSES)
    if bad_labels:
        raise ValueError(f"Nhãn không hợp lệ: {sorted(bad_labels)}. Chỉ chấp nhận {CLASSES}")
    bad_splits = set(df["split"]) - VALID_SPLITS
    if bad_splits:
        raise ValueError(f"Giá trị split không hợp lệ: {sorted(bad_splits)}")
    return df


def check_leakage(df: pd.DataFrame) -> dict:
    """Kiểm tra cùng một file có nằm ở nhiều tập không, và ảnh trùng tên file giữa các tập."""
    report = {}
    dup_paths = df[df.duplicated("filepath", keep=False)]
    report["duplicate_rows"] = int(len(dup_paths))
    sets = {s: set(df.loc[df.split == s, "filepath"]) for s in VALID_SPLITS}
    report["train_val_overlap"] = len(sets["train"] & sets["val"])
    report["train_test_overlap"] = len(sets["train"] & sets["test"])
    report["val_test_overlap"] = len(sets["val"] & sets["test"])
    report["ok"] = (report["duplicate_rows"] == 0 and report["train_val_overlap"] == 0
                    and report["train_test_overlap"] == 0 and report["val_test_overlap"] == 0)
    return report


def distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.crosstab(df["label"], df["split"]).reindex(index=CLASSES, fill_value=0)
    t = t.reindex(columns=[c for c in ["train", "val", "test"] if c in t.columns])
    pct = (t / t.sum(axis=0) * 100).round(1).add_suffix("_%")
    return pd.concat([t, pct], axis=1)
