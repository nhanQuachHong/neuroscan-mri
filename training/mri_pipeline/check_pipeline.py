"""Kiểm tra pipeline trước khi cả nhóm train:
1. Phân bố 4 lớp ở từng tập Train/Val/Test
2. Rò rỉ dữ liệu (cùng file nằm ở nhiều tập)
3. Tất cả ảnh mở được
4. Một batch có đúng kích thước (B, 3, 224, 224) và đã chuẩn hóa
5. Lưu ảnh minh họa augmentation (aug_preview.png) để kiểm tra bằng mắt

Chạy:
    python check_pipeline.py --split-csv split.csv --data-root /kaggle/input/brisc2025 --out-dir pipeline_check
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from config import CLASSES
from split_utils import check_leakage, distribution_table, read_split


def check_images(df, data_root, limit=None):
    bad = []
    rows = df if limit is None else df.sample(min(limit, len(df)), random_state=0)
    for p in rows["filepath"]:
        full = os.path.join(data_root, p)
        try:
            with Image.open(full) as im:
                im.verify()
        except Exception as e:  # noqa: BLE001
            bad.append((p, str(e)))
    return bad


def save_aug_preview(datasets, out_path, n_per_class=2, n_aug=4):
    from transforms import denormalize, get_eval_transforms
    train = datasets["train"]
    eval_tf = get_eval_transforms()
    fig, axes = plt.subplots(len(CLASSES) * n_per_class, n_aug + 1,
                             figsize=(2.2 * (n_aug + 1), 2.2 * len(CLASSES) * n_per_class))
    r = 0
    for ci, cname in enumerate(CLASSES):
        idxs = [i for i, l in enumerate(train.labels) if l == ci][:n_per_class]
        for idx in idxs:
            img = Image.open(train.path_of(idx)).convert("RGB")
            axes[r, 0].imshow(denormalize(eval_tf(img)).permute(1, 2, 0))
            axes[r, 0].set_title(f"{cname}\n(gốc)", fontsize=8)
            for k in range(n_aug):
                x, _ = train[idx]
                axes[r, k + 1].imshow(denormalize(x).permute(1, 2, 0))
                axes[r, k + 1].set_title(f"aug {k + 1}", fontsize=8)
            r += 1
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", default="pipeline_check")
    ap.add_argument("--check-limit", type=int, default=None,
                    help="Chỉ kiểm tra N ảnh ngẫu nhiên (mặc định kiểm tra tất cả)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    df = read_split(a.split_csv)
    print("== 1. Phân bố lớp ==")
    dist = distribution_table(df)
    print(dist.to_string())
    dist.to_csv(os.path.join(a.out_dir, "class_distribution.csv"))

    print("\n== 2. Kiểm tra rò rỉ dữ liệu ==")
    leak = check_leakage(df)
    print(leak)

    print("\n== 3. Kiểm tra ảnh mở được ==")
    bad = check_images(df, a.data_root, a.check_limit)
    print(f"Ảnh lỗi: {len(bad)}")
    for p, e in bad[:10]:
        print("  ", p, "->", e)

    print("\n== 4. Kiểm tra batch ==")
    import torch
    from dataset import get_loaders
    loaders, datasets = get_loaders(a.split_csv, a.data_root, batch_size=8, num_workers=0)
    for name in ["train", "val", "test"]:
        x, y = next(iter(loaders[name]))
        print(f"{name:<5} batch x={tuple(x.shape)} dtype={x.dtype} "
              f"mean={x.mean():.3f} std={x.std():.3f} y={y.tolist()}")
        assert x.shape[1:] == (3, 224, 224), "Sai kích thước ảnh"
    print("Train dùng augmentation:", "RandomResizedCrop" in str(datasets["train"].transform))
    print("Val/Test KHÔNG dùng augmentation:",
          "Random" not in str(datasets["val"].transform) and "Random" not in str(datasets["test"].transform))

    print("\n== 5. Ảnh minh họa augmentation ==")
    prev = os.path.join(a.out_dir, "aug_preview.png")
    save_aug_preview(datasets, prev)
    print("Đã lưu", prev)

    ok = leak["ok"] and not bad
    print("\nKẾT LUẬN:", "PIPELINE OK" if ok else "CÓ VẤN ĐỀ, xem chi tiết ở trên")
    with open(os.path.join(a.out_dir, "check_report.txt"), "w", encoding="utf-8") as f:
        f.write(dist.to_string() + "\n\n" + str(leak) + f"\n\nẢnh lỗi: {len(bad)}\n"
                + ("PIPELINE OK" if ok else "CÓ VẤN ĐỀ") + "\n")
    if not ok:
        raise SystemExit(1)
