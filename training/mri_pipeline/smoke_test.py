"""Chạy thử toàn bộ pipeline trên ảnh GIẢ (khoảng 2-5 phút trên Colab/Kaggle GPU).
Dùng để chắc chắn code chạy được trước khi train thật, KHÔNG dùng kết quả để báo cáo.

    python smoke_test.py                 # 3 mô hình, không tải trọng số pretrained
    python smoke_test.py --pretrained    # kiểm tra thêm việc tải trọng số ImageNet
    python smoke_test.py --models resnet50
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

from config import CLASSES

HERE = os.path.dirname(os.path.abspath(__file__))


def make_fake_brisc(root, n_train=20, n_test=6, size=256):
    """Tạo cấu trúc giống BRISC: classification_task/{train,test}/<lớp>/*.jpg (ảnh xám)."""
    rng = np.random.default_rng(0)
    for split, n in [("train", n_train), ("test", n_test)]:
        for ci, c in enumerate(CLASSES):
            d = os.path.join(root, "classification_task", split, c)
            os.makedirs(d, exist_ok=True)
            for i in range(n):
                img = rng.normal(60, 20, (size, size))
                yy, xx = np.ogrid[:size, :size]
                cy, cx = size // 2 + (ci - 1.5) * 30, size // 2
                img[(yy - cy) ** 2 + (xx - cx) ** 2 < (20 + ci * 8) ** 2] += 120  # "khối" khác nhau theo lớp
                Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="L").save(
                    os.path.join(d, f"brisc2025_{split}_{i:05d}_{c[:2]}_ax_t1.jpg"))
    # ảnh mask giả để chắc chắn bị bỏ qua
    os.makedirs(os.path.join(root, "segmentation_task", "train", "masks"), exist_ok=True)


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        raise SystemExit(f"LỖI khi chạy: {' '.join(cmd)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet50", "efficientnet_b0", "swin_tiny"])
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--keep", action="store_true", help="Giữ lại thư mục tạm để xem")
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="mri_smoke_")
    data, split, out = os.path.join(tmp, "data"), os.path.join(tmp, "split.csv"), os.path.join(tmp, "outputs")
    try:
        make_fake_brisc(data)
        py = sys.executable
        run([py, "make_temp_split.py", "--data-root", data, "--out", split])
        run([py, "class_weights.py", "--split-csv", split, "--out", os.path.join(tmp, "class_weights.json")])
        run([py, "check_pipeline.py", "--split-csv", split, "--data-root", data,
             "--out-dir", os.path.join(tmp, "check")])
        for m in a.models:
            cmd = [py, "train.py", "--model", m, "--split-csv", split, "--data-root", data,
                   "--out-dir", out, "--epochs", "2", "--freeze-epochs", "1", "--batch-size", "8",
                   "--num-workers", "0", "--max-batches", "3"]
            if not a.pretrained:
                cmd.append("--no-pretrained")
            run(cmd)
            for f in ["{m}_best.pth", "{m}_history.json", "{m}_metrics.json", "{m}_test_predictions.csv"]:
                p = os.path.join(out, m, f.format(m=m))
                assert os.path.exists(p), f"Thiếu file {p}"
        print("\n" + "=" * 50 + "\nSMOKE TEST THÀNH CÔNG: pipeline và khung huấn luyện chạy được với "
              + ", ".join(a.models) + "\n" + "=" * 50)
    finally:
        if a.keep:
            print("Thư mục tạm:", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
