"""Reconstruct standard torchvision classifiers and load local state dictionaries."""
from collections.abc import Mapping
from functools import lru_cache
import os


@lru_cache(maxsize=3)
def load_model(path, modified, architecture):
    import torch
    from torchvision import models

    network = getattr(models, architecture)(weights=None, num_classes=4)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], Mapping):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, Mapping) or not checkpoint or not all(
            isinstance(k, str) and torch.is_tensor(v) for k, v in checkpoint.items()):
        raise ValueError("Checkpoint phải chứa state_dict của mô hình.")
    if all(k.startswith("module.") for k in checkpoint):
        checkpoint = {k.removeprefix("module."): v for k, v in checkpoint.items()}
    if architecture == "resnet50" and {"fc.1.weight", "fc.1.bias"} <= checkpoint.keys():
        network.fc = torch.nn.Sequential(torch.nn.Dropout(0.3), network.fc)
    network.load_state_dict(checkpoint, strict=True)
    setting = os.environ.get("NEUROSCAN_DEVICE", "auto").lower()
    if setting not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("NEUROSCAN_DEVICE phải là auto, cpu, cuda hoặc mps.")
    if setting == "auto":
        setting = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return network.eval().to(setting)


def infer(network, arr):
    import torch
    device = next(network.parameters()).device
    with torch.inference_mode():
        return network(torch.from_numpy(arr).to(device)).detach().cpu().numpy().astype(float)
