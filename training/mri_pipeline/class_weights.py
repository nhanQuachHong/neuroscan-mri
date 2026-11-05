"""Tính Class Weight CHỈ trên tập Train.

Chạy:  python class_weights.py --split-csv split.csv --out class_weights.json
"""
import argparse

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import json
import os

from config import CLASSES
from split_utils import read_split


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def compute_weights(split_csv: str):
    df = read_split(split_csv)
    train_labels = df.loc[df.split == "train", "label"].to_numpy()
    present = set(train_labels)
    if present != set(CLASSES):
        raise ValueError(f"Tập Train thiếu lớp: {sorted(set(CLASSES) - present)}")
    weights = compute_class_weight("balanced", classes=np.array(CLASSES), y=train_labels)
    counts = {c: int((train_labels == c).sum()) for c in CLASSES}
    ratio = max(counts.values()) / min(counts.values())
    return {
        "classes": CLASSES,
        "weights": [round(float(w), 6) for w in weights],
        "train_counts": counts,
        "imbalance_ratio": round(ratio, 3),
        "method": "sklearn compute_class_weight('balanced') trên tập Train",
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--out", default="class_weights.json")
    a = ap.parse_args()
    info = compute_weights(a.split_csv)
    save_json(info, a.out)
    print("Số ảnh Train mỗi lớp:", info["train_counts"])
    print("Tỉ lệ mất cân bằng (nhiều nhất / ít nhất):", info["imbalance_ratio"])
    for c, w in zip(info["classes"], info["weights"]):
        print(f"  {c:<12} weight = {w:.4f}")
    print("Đã lưu", a.out)
