"""Khung huấn luyện dùng chung cho ResNet50, EfficientNetB0 và Swin Transformer.

Ví dụ:
    python train.py --model resnet50        --split-csv split.csv --data-root /kaggle/input/brisc2025
    python train.py --model efficientnet_b0 --split-csv split.csv --data-root /kaggle/input/brisc2025
    python train.py --model swin_tiny       --split-csv split.csv --data-root /kaggle/input/brisc2025

Đầu ra trong outputs/<model>/:
    <model>_best.pth               checkpoint tốt nhất theo Val Macro F1
    <model>_history.json           loss/acc/F1 từng epoch
    <model>_metrics.json           chỉ số trên Test + thời gian train/dự đoán
    <model>_test_predictions.csv   dự đoán từng ảnh Test (đầu vào cho module đánh giá của Son)
    train_log.txt
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

import config as C
from class_weights import compute_weights
from dataset import get_loaders
from models import SUPPORTED, build_model, count_params, set_backbone_trainable
from utils import get_device, load_json, save_json, seed_everything


def parse_args():
    ap = argparse.ArgumentParser(description="Huấn luyện mô hình phân loại MRI não")
    ap.add_argument("--model", required=True, choices=SUPPORTED)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--epochs", type=int, default=C.DEFAULT_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=C.DEFAULT_BATCH_SIZE)
    ap.add_argument("--patience", type=int, default=C.DEFAULT_PATIENCE)
    ap.add_argument("--freeze-epochs", type=int, default=C.DEFAULT_FREEZE_EPOCHS,
                    help="Số epoch đầu chỉ train lớp phân loại (0 = không đóng băng)")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="lr khi đóng băng backbone")
    # Phần đặc thù từng mô hình, bỏ trống thì lấy theo config.MODEL_DEFAULTS
    ap.add_argument("--optimizer", choices=["adam", "adamw"])
    ap.add_argument("--lr", type=float)
    ap.add_argument("--weight-decay", type=float)
    ap.add_argument("--scheduler", choices=["cosine", "plateau", "none"])
    ap.add_argument("--warmup-epochs", type=int)
    ap.add_argument("--class-weights", default="auto",
                    help="'auto' = tính từ tập Train, 'none' = không dùng, hoặc đường dẫn class_weights.json")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--no-amp", action="store_true", help="Tắt mixed precision")
    ap.add_argument("--no-pretrained", action="store_true", help="Chỉ dùng khi chạy thử")
    ap.add_argument("--max-batches", type=int, default=0, help="Giới hạn số batch mỗi epoch (chạy thử)")
    a = ap.parse_args()
    for k, v in C.MODEL_DEFAULTS[a.model].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    return a


class Logger:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def make_optimizer(model, a, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    if a.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=a.weight_decay)
    return torch.optim.Adam(params, lr=lr, weight_decay=a.weight_decay)


def make_scheduler(opt, a, n_epochs):
    if a.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    if a.scheduler == "cosine":
        w = a.warmup_epochs

        def f(e):  # e = số epoch đã qua trong giai đoạn fine-tune
            if w > 0 and e < w:
                return (e + 1) / w
            progress = (e - w) / max(1, n_epochs - w)
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        return torch.optim.lr_scheduler.LambdaLR(opt, f)
    return None


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def run_epoch(model, loader, criterion, device, use_amp, optimizer=None, scaler=None,
              grad_clip=0.0, max_batches=0):
    train = optimizer is not None
    model.train(train)
    total_loss, n = 0.0, 0
    ys, preds, probs = [], [], []
    with torch.set_grad_enabled(train):
        for b, (x, y) in enumerate(loader):
            if max_batches and b >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            p = torch.softmax(logits.float(), dim=1)
            ys.append(y.cpu().numpy())
            probs.append(p.detach().cpu().numpy())
            preds.append(p.argmax(1).cpu().numpy())
    y_true, y_pred, y_prob = np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)
    return {
        "loss": total_loss / max(1, n),
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(C.NUM_CLASSES)),
                                   zero_division=0)),
    }, y_true, y_pred, y_prob


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def measure_latency(model, dataset, device, n=50, warmup=5):
    """Thời gian dự đoán 1 ảnh (batch=1), tính bằng ms."""
    model.eval()
    n = min(n, len(dataset))
    xs = [dataset[i][0].unsqueeze(0).to(device) for i in range(n)]
    for x in xs[:warmup]:
        model(x)
    sync(device)
    t = time.perf_counter()
    for x in xs:
        model(x)
    sync(device)
    return (time.perf_counter() - t) / n * 1000


def main():
    a = parse_args()
    seed_everything(a.seed)
    device = get_device()
    use_amp = device.type == "cuda" and not a.no_amp
    out = os.path.join(a.out_dir, a.model)
    os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "train_log.txt"))
    log(f"===== {a.model} | device={device} | AMP={use_amp} =====")
    log(f"Cấu hình: {vars(a)}")

    loaders, datasets = get_loaders(a.split_csv, a.data_root, a.batch_size, a.num_workers, a.seed)
    log("Số ảnh: " + ", ".join(f"{k}={len(v)}" for k, v in datasets.items()))

    # Class Weight: chỉ tính trên Train
    if a.class_weights == "none":
        weight = None
        log("Không dùng Class Weight")
    else:
        info = compute_weights(a.split_csv) if a.class_weights == "auto" else load_json(a.class_weights)
        assert info["classes"] == C.CLASSES, "Thứ tự lớp trong class_weights.json khác config.CLASSES"
        weight = torch.tensor(info["weights"], dtype=torch.float32, device=device)
        save_json(info, os.path.join(out, "class_weights.json"))
        log(f"Class Weight: {dict(zip(C.CLASSES, info['weights']))}")
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=a.label_smoothing)

    model, head = build_model(a.model, pretrained=not a.no_pretrained)
    model.to(device)
    n_params = count_params(model)
    log(f"Số tham số: {n_params / 1e6:.2f}M")
    scaler = make_scaler(use_amp)

    freeze = min(a.freeze_epochs, a.epochs)
    stage = None
    optimizer = scheduler = None
    history, best_f1, best_loss, best_epoch, bad_epochs = [], -1.0, float("inf"), 0, 0
    ckpt_path = os.path.join(out, C.CHECKPOINT_NAME.format(model=a.model))
    t_train = time.perf_counter()

    for epoch in range(1, a.epochs + 1):
        want = "head" if epoch <= freeze else "full"
        if want != stage:
            stage = want
            set_backbone_trainable(model, head, trainable=(stage == "full"))
            lr0 = a.head_lr if stage == "head" else a.lr
            optimizer = make_optimizer(model, a, lr0)
            scheduler = make_scheduler(optimizer, a, a.epochs - freeze) if stage == "full" else None
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log(f"-- Giai đoạn '{stage}': lr={lr0}, tham số được train={n_train / 1e6:.2f}M")

        t0 = time.perf_counter()
        tr, *_ = run_epoch(model, loaders["train"], criterion, device, use_amp, optimizer, scaler,
                           a.grad_clip, a.max_batches)
        va, *_ = run_epoch(model, loaders["val"], criterion, device, use_amp, max_batches=a.max_batches)
        lr_now = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(va["macro_f1"]) if a.scheduler == "plateau" else scheduler.step()
        dt = time.perf_counter() - t0
        history.append({"epoch": epoch, "stage": stage, "lr": lr_now, "time_sec": round(dt, 2),
                        "train_loss": tr["loss"], "train_acc": tr["acc"], "train_macro_f1": tr["macro_f1"],
                        "val_loss": va["loss"], "val_acc": va["acc"], "val_macro_f1": va["macro_f1"]})
        improved = va["macro_f1"] > best_f1 or (va["macro_f1"] == best_f1 and va["loss"] < best_loss)
        if improved:
            best_f1, best_loss, best_epoch, bad_epochs = va["macro_f1"], va["loss"], epoch, 0
            torch.save({"model_name": a.model, "state_dict": model.state_dict(), "classes": C.CLASSES,
                        "img_size": C.IMG_SIZE, "mean": C.IMAGENET_MEAN, "std": C.IMAGENET_STD,
                        "epoch": epoch, "val_macro_f1": best_f1, "args": vars(a)}, ckpt_path)
        else:
            bad_epochs += 1
        log(f"Epoch {epoch:>3}/{a.epochs} [{stage}] lr={lr_now:.2e} | "
            f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
            f"val loss {va['loss']:.4f} acc {va['acc']:.4f} F1 {va['macro_f1']:.4f} | "
            f"{dt:.0f}s {'*' if improved else ''}")
        save_json(history, os.path.join(out, C.HISTORY_NAME.format(model=a.model)))
        if stage == "full" and bad_epochs >= a.patience:
            log(f"Early stopping: Val Macro F1 không tăng sau {a.patience} epoch")
            break

    train_time = time.perf_counter() - t_train
    log(f"Tổng thời gian huấn luyện: {train_time / 60:.1f} phút | epoch tốt nhất: {best_epoch} "
        f"(Val Macro F1 = {best_f1:.4f})")

    # ===== Đánh giá trên Test bằng checkpoint tốt nhất =====
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    sync(device)
    t0 = time.perf_counter()
    te, y_true, y_pred, y_prob = run_epoch(model, loaders["test"], criterion, device, use_amp,
                                           max_batches=a.max_batches)
    sync(device)
    test_time = time.perf_counter() - t0
    latency = measure_latency(model, datasets["test"], device)
    labels = list(range(C.NUM_CLASSES))

    metrics = {
        "model": a.model,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "per_class": classification_report(y_true, y_pred, labels=labels, target_names=C.CLASSES,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classes": C.CLASSES,
        "train_time_sec": round(train_time, 1),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "test_time_sec": round(test_time, 2),
        "inference_ms_per_image_batched": round(test_time / max(1, len(y_true)) * 1000, 3),
        "inference_ms_per_image_single": round(latency, 3),
        "num_params_million": round(n_params / 1e6, 2),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device),
        "config": {k: v for k, v in vars(a).items()},
    }
    save_json(metrics, os.path.join(out, C.METRICS_NAME.format(model=a.model)))

    paths = datasets["test"].df["filepath"].tolist()[:len(y_true)]
    pred_df = pd.DataFrame({"filepath": paths, "label": [C.CLASSES[i] for i in y_true],
                            "pred": [C.CLASSES[i] for i in y_pred]})
    for i, c in enumerate(C.CLASSES):
        pred_df[f"prob_{c}"] = y_prob[:, i].round(6)
    pred_df.to_csv(os.path.join(out, C.PREDICTIONS_NAME.format(model=a.model)), index=False)

    log(f"TEST | acc {metrics['accuracy']:.4f} | precision {metrics['precision_macro']:.4f} | "
        f"recall {metrics['recall_macro']:.4f} | macro F1 {metrics['f1_macro']:.4f}")
    log(f"Thời gian dự đoán: {metrics['inference_ms_per_image_single']:.2f} ms/ảnh (batch=1)")
    log("Confusion Matrix (hàng = thật, cột = dự đoán, thứ tự " + ", ".join(C.CLASSES) + "):")
    for row in metrics["confusion_matrix"]:
        log("   " + " ".join(f"{v:>5}" for v in row))
    log(f"Đã lưu kết quả vào {out}")


if __name__ == "__main__":
    main()
