"""Compare the installed trained checkpoint with the supplied training code on CPU.

Default inputs are synthetic technical fixtures, not an MRI accuracy benchmark.
Pass --image PATH repeatedly to compare specific PNG/JPEG test images.
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


def fixtures():
    pixels = np.arange(93 * 127 * 3, dtype=np.uint32).reshape(93, 127, 3).astype(np.uint8)
    rgb = Image.fromarray(pixels)
    cases = [("rgb_png", rgb, "PNG", {}), ("grayscale_png", rgb.convert("L"), "PNG", {}),
             ("rgb_jpeg", rgb, "JPEG", {})]
    exif = Image.Exif()
    exif[274] = 6
    cases.append(("exif_rotated_jpeg", rgb, "JPEG", {"exif": exif}))
    for name, image, fmt, kwargs in cases:
        output = io.BytesIO()
        image.save(output, fmt, **kwargs)
        yield name, output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", default=[], type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    os.environ["NEUROSCAN_DEVICE"] = "cpu"
    torch.set_num_threads(1)
    folder = ROOT / "models/resnet50"
    checkpoint_path = folder / "model.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["classes"] != training_config.CLASSES:
        raise ValueError("Checkpoint classes differ from the supplied training config.")
    display = {"glioma": "Glioma", "meningioma": "Meningioma", "no_tumor": "No Tumor",
               "pituitary": "Pituitary Tumor"}
    config = app.config_for(folder)
    if config["labels"] != [display[c] for c in training_config.CLASSES]:
        raise ValueError("Application label order differs from training config.")
    reference, _ = build_model("resnet50", pretrained=False)
    reference.load_state_dict(checkpoint["state_dict"], strict=True)
    reference.eval()
    transform = get_eval_transforms()
    inputs = [(p.name, p.read_bytes()) for p in args.image] if args.image else list(fixtures())
    rows = []
    for name, payload in inputs:
        # This reproduces MRIDataset.__getitem__: convert RGB without EXIF rotation.
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
        rows.append({"input": name,
                     "max_tensor_difference": float(np.abs(actual_tensor - tensor.numpy()).max()),
                     "max_probability_difference": float(np.abs(actual - expected).max())})
    report = {"status": "passed", "inputs": "supplied_images" if args.image else "synthetic_fixtures",
              "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
              "classes": checkpoint["classes"], "cases": rows,
              "limitation": "Technical parity only; no MRI accuracy evaluation was performed."}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
