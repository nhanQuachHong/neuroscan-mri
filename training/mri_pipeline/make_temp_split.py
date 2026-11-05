"""TẠM THỜI: tạo split.csv từ thư mục ảnh để Trí và các bạn train chạy thử trước
khi Son gửi split.csv chính thức (đã loại ảnh lỗi/trùng). Khi có file của Son thì
dùng file đó, KHÔNG dùng file này để báo cáo kết quả.

Mặc định:
- Giữ tập test chính thức của BRISC (thư mục .../test/...)
- Chia tập train chính thức thành train/val (85/15), stratified, seed 42
Nếu không có thư mục test chính thức thì chia toàn bộ 70/15/15.

Chạy:
    python make_temp_split.py --data-root /kaggle/input/brisc2025 --out split_temp.csv
"""
import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from config import CLASSES, SEED
from split_utils import normalize_label

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def scan(data_root: str) -> pd.DataFrame:
    rows = []
    for dirpath, _, files in os.walk(data_root):
        low = dirpath.lower()
        if "segmentation" in low or "mask" in low:
            continue  # bỏ phần phân đoạn và mask
        label = normalize_label(os.path.basename(dirpath))
        if label not in CLASSES:
            continue
        parts = [p.lower() for p in os.path.relpath(dirpath, data_root).split(os.sep)]
        official = "test" if "test" in parts else ("train" if "train" in parts else "")
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                rel = os.path.relpath(os.path.join(dirpath, f), data_root)
                rows.append({"filepath": rel.replace(os.sep, "/"), "label": label, "official": official})
    if not rows:
        raise SystemExit(f"Không tìm thấy ảnh trong thư mục lớp {CLASSES} dưới {data_root}")
    return pd.DataFrame(rows).sort_values("filepath").reset_index(drop=True)


def make_split(df: pd.DataFrame, val_size=0.15, test_size=0.15, seed=SEED) -> pd.DataFrame:
    if (df.official == "test").any() and (df.official == "train").any():
        test = df[df.official == "test"].copy()
        pool = df[df.official == "train"]
        tr, va = train_test_split(pool, test_size=val_size, stratify=pool.label, random_state=seed)
        mode = "official_test + train/val 85/15"
    else:
        tr, rest = train_test_split(df, test_size=val_size + test_size, stratify=df.label, random_state=seed)
        va, test = train_test_split(rest, test_size=test_size / (val_size + test_size),
                                    stratify=rest.label, random_state=seed)
        mode = "stratified 70/15/15"
    tr, va, test = tr.assign(split="train"), va.assign(split="val"), test.assign(split="test")
    out = pd.concat([tr, va, test])[["filepath", "label", "split"]]
    print("Cách chia:", mode)
    return out.sort_values(["split", "filepath"]).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="split_temp.csv")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    df = scan(a.data_root)
    split = make_split(df, seed=a.seed)
    split.to_csv(a.out, index=False)
    print(pd.crosstab(split.label, split.split))
    print("Đã lưu", a.out, "-", len(split), "ảnh")
