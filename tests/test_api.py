"""Exercise the API using a four-output ONNX fixture, never trained weights."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import onnx
from fastapi.testclient import TestClient
from onnx import TensorProto, helper
from PIL import Image

import server


class ApiSmokeTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        override = patch.object(server.core, "MODELS", self.root)
        override.start()
        self.addCleanup(override.stop)
        self.client = TestClient(server.api)
        self.addCleanup(self.client.close)

    def fixture(self):
        folder = self.root / "resnet50"
        folder.mkdir()
        labels = list(reversed(server.core.LABELS))
        config = dict(labels=labels, channels=1, size=[2, 2], layout="NCHW",
                      resize="nearest", output_type="logits", scale=1/255,
                      mean=[0], std=[1])
        (folder / "model_config.json").write_text(json.dumps(config))
        graph = helper.make_graph(
            [helper.make_node("Flatten", ["x"], ["y"], axis=1)], "fixture",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])])
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)],
                                    ir_version=8), folder / "model.onnx")
        return folder, labels

    def image(self, fmt):
        image = Image.new("L", (2, 2))
        image.putdata([255, 0, 0, 0])
        result = io.BytesIO()
        image.save(result, fmt)
        return result.getvalue()

    def test_startup_and_missing_model(self):
        self.assertEqual(self.client.get("/api/health").json(), {"status": "ok"})
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/integration.js").status_code, 200)
        self.assertEqual(self.client.post("/api/predict/resnet50", content=self.image("PNG")).status_code, 409)
        self.assertEqual(self.client.get("/api/metrics").json(), {"models": []})

    def test_prediction_preview_comparison_and_metrics(self):
        folder, labels = self.fixture()
        png = self.image("PNG")
        for payload in (png, self.image("JPEG")):
            reply = self.client.post("/api/predict/resnet50", content=payload)
            self.assertEqual(reply.status_code, 200)
            data = reply.json()
            self.assertEqual(data["mode"], "real")
            scores = [r["probability"] for r in data["ranked"]]
            self.assertEqual(scores, sorted(scores, reverse=True))
            self.assertAlmostEqual(sum(scores), 1)
            self.assertEqual(data["ranked"][0]["label"], "No Tumor")
        resized = self.client.post("/api/preview/resnet50", content=png)
        self.assertEqual(resized.status_code, 200)
        self.assertEqual(Image.open(io.BytesIO(resized.content)).size, (2, 2))
        rows = self.client.post("/api/compare", content=png).json()["results"]
        self.assertEqual(len(rows), 3)
        self.assertEqual(sum("error" in row for row in rows), 2)
        self.assertEqual(self.client.get("/api/metrics/resnet50").status_code, 404)
        (folder / "metrics.json").write_text(json.dumps(dict(
            labels=labels, test_set_id="fixture",
            confusion_matrix=[[1,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,4]])))
        self.assertEqual(self.client.get("/api/metrics/resnet50").json()["scores"]["Accuracy"], 1)
        self.assertTrue(self.client.get("/api/metrics").json()["models"][0]["available"])

    def test_bad_image_and_incompatible_model(self):
        folder, _ = self.fixture()
        url = "/api/predict/resnet50"
        self.assertEqual(self.client.post(url, content=b"broken").status_code, 422)
        self.assertEqual(self.client.post(url, content=b"x" * (server.core.MAX_BYTES + 1)).status_code, 413)
        path = folder / "model_config.json"
        config = json.loads(path.read_text())
        config["size"] = [3, 3]
        path.write_text(json.dumps(config))
        self.assertEqual(self.client.post(url, content=self.image("PNG")).status_code, 422)


if __name__ == "__main__":
    unittest.main()
