"""Check the PyTorch API with temporary random weights, not trained MRI weights."""
import asyncio
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import app
import server


async def post_image(path, payload):
    messages = []

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
             "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": path, "raw_path": path.encode(), "query_string": b"",
             "root_path": "", "headers": [], "client": ("127.0.0.1", 1),
             "server": ("127.0.0.1", 8002)}
    await server.api(scope, receive, send)
    status = next(m["status"] for m in messages if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return status, json.loads(body)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
                     "Requires requirements-pytorch.txt")
class PyTorchApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from torchvision import models
        cls.torch = torch
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name) / "resnet50"
        cls.folder.mkdir()
        cls.config = json.loads((app.ROOT / "models/resnet50/model_config.json").read_text())
        (cls.folder / "model_config.json").write_text(json.dumps(cls.config))
        with torch.random.fork_rng():
            torch.manual_seed(42)
            cls.reference = models.resnet50(weights=None, num_classes=4).eval()
        cls.reference.fc = torch.nn.Sequential(torch.nn.Dropout(0.3), cls.reference.fc)
        cls.reference.eval()
        torch.save({"model_state_dict": {"module." + k: v for k, v in
                                       cls.reference.state_dict().items()}}, cls.folder / "model.pth")
        pixels = np.arange(93 * 127 * 3, dtype=np.uint32).reshape(93, 127, 3).astype(np.uint8)
        cls.image = Image.fromarray(pixels)
        buffer = io.BytesIO()
        cls.image.save(buffer, "PNG")
        cls.payload = buffer.getvalue()

    @classmethod
    def tearDownClass(cls):
        from torch_backend import load_model
        load_model.cache_clear()
        cls.tmp.cleanup()
        cls.torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.models = patch.object(app, "MODELS", self.folder.parent)
        self.models.start()
        self.addCleanup(self.models.stop)
        self.device = patch.dict("os.environ", {"NEUROSCAN_DEVICE": "cpu"})
        self.device.start()
        self.addCleanup(self.device.stop)

    def test_wrapped_checkpoint_prediction_matches_torchvision_reference(self):
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(), transforms.Normalize(self.config["mean"], self.config["std"])])
        tensor = transform(self.image).unsqueeze(0)
        np.testing.assert_allclose(app.preprocess(self.payload, self.config), tensor.numpy(), atol=1e-6)
        with self.torch.inference_mode():
            expected = self.reference(tensor).softmax(dim=1)[0].numpy()
        status, result = asyncio.run(post_image("/api/predict/resnet50", self.payload))
        self.assertEqual(status, 200, result)
        self.assertEqual(result["mode"], "real")
        probabilities = {row["label"]: row["probability"] for row in result["ranked"]}
        np.testing.assert_allclose([probabilities[label] for label in self.config["labels"]],
                                   expected, atol=1e-5, rtol=1e-4)
        self.assertAlmostEqual(sum(probabilities.values()), 1)

    def test_corrupt_image_is_rejected(self):
        status, result = asyncio.run(post_image("/api/predict/resnet50", b"broken"))
        self.assertEqual(status, 422, result)

    def test_missing_checkpoint_is_not_replaced_by_demo_output(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(app, "MODELS", Path(directory)):
            status, result = asyncio.run(post_image("/api/predict/resnet50", self.payload))
        self.assertEqual(status, 409)
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
