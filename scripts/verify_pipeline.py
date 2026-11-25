"""Compare installed trained checkpoints with training/mri_pipeline on CPU.

Default inputs are synthetic technical fixtures, not an MRI accuracy benchmark.
Pass --model resnet50|efficientnet_b0|swin|all and optional --image PATH repeatedly.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training/mri_pipeline"))

import numpy as np
from PIL import Image
import torch

import app
import config as training_config
from models import build_model
from transforms import get_eval_transforms

MODEL_FOLDERS = ("resnet50", "efficientnet_b0", "swin")
DISPLAY_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "no_tumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}


def fixtures():
    pixels = np.arange(93 * 127 * 3, dtype=np.uint32).reshape(93, 127, 3).astype(np.uint8)
    rgb = Image.fromarray(pixels)
    cases = [
        ("rgb_png", rgb, "PNG", {}),
        ("grayscale_png", rgb.convert("L"), "PNG", {}),
        ("rgb_jpeg", rgb, "JPEG", {}),
    ]
    exif = Image.Exif()
    exif[274] = 6
    cases.append(("exif_rotated_jpeg", rgb, "JPEG", {"exif": exif}))
    for name, image, fmt, kwargs in cases:
        output = io.BytesIO()
        image.save(output, fmt, **kwargs)
        yield name, output.getvalue()


def verify_one(model_id: str, inputs):
    folder = ROOT / "models" / model_id
    checkpoint_path = folder / "model.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("classes") != training_config.CLASSES:
        raise ValueError(f"{model_id}: Checkpoint classes differ from training config.")
    config = app.config_for(folder)
    expected_labels = [DISPLAY_LABELS[c] for c in training_config.CLASSES]
    if config["labels"] != expected_labels:
        raise ValueError(f"{model_id}: Application label order differs from training config.")
    arch = config["architecture"]
    if checkpoint.get("model_name") and checkpoint["model_name"] != arch:
        raise ValueError(f"{model_id}: Checkpoint model_name ({checkpoint['model_name']}) != config architecture ({arch}).")
    reference, _ = build_model(arch, pretrained=False)
    reference.load_state_dict(checkpoint["state_dict"], strict=True)
    reference.eval()
    transform = get_eval_transforms()
    rows = []
    for name, payload in inputs:
        # Reproduces MRIDataset.__getitem__: convert RGB without EXIF rotation.
        with Image.open(io.BytesIO(payload)) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0)
        actual_tensor = app.preprocess(payload, config)
        np.testing.assert_allclose(actual_tensor, tensor.numpy(), atol=1e-6)
        with torch.inference_mode():
            expected = reference(tensor).softmax(dim=1)[0].numpy()
        prediction = app.predict(folder, payload)
        scores = {r["label"]: r["probability"] for r in prediction["ranked"]}
        actual = np.asarray([scores[label] for label in config["labels"]])
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
        rows.append({
            "input": name,
            "max_tensor_difference": float(np.abs(actual_tensor - tensor.numpy()).max()),
            "max_probability_difference": float(np.abs(actual - expected).max()),
        })
    return {
        "model_id": model_id,
        "architecture": arch,
        "status": "passed",
        "epoch": checkpoint.get("epoch"),
        "val_macro_f1": checkpoint.get("val_macro_f1"),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "classes": checkpoint["classes"],
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="all", choices=("all", *MODEL_FOLDERS))
    parser.add_argument("--image", action="append", default=[], type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    os.environ["NEUROSCAN_DEVICE"] = "cpu"
    torch.set_num_threads(1)
    inputs = [(p.name, p.read_bytes()) for p in args.image] if args.image else list(fixtures())
    targets = MODEL_FOLDERS if args.model == "all" else (args.model,)
    results = [verify_one(model_id, inputs) for model_id in targets]
    report = {
        "status": "passed",
        "inputs": "supplied_images" if args.image else "synthetic_fixtures",
        "models": results,
        "limitation": "Technical parity only; no MRI accuracy evaluation was performed.",
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
