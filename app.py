"""Local NeuroScan web application. ONNX dependencies load only for inference."""
import argparse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from math import isfinite
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
LABELS = ("Glioma", "Meningioma", "Pituitary Tumor", "No Tumor")
MAX_BYTES = 10 * 1024 * 1024


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def config_for(folder):
    path = folder / "model_config.json"
    if not path.is_file():
        raise ValueError("Thiếu model_config.json.")
    try:
        c = read_json(path)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("model_config.json không phải JSON UTF-8 hợp lệ.") from exc
    if not isinstance(c, dict):
        raise ValueError("model_config.json phải chứa một đối tượng JSON.")

    def check(ok, message):
        if not ok:
            raise ValueError(message)

    labels = c.get("labels")
    check(isinstance(labels, list) and all(isinstance(x, str) for x in labels)
          and sorted(labels) == sorted(LABELS), "labels phải chứa đúng bốn lớp, không trùng.")
    channels, size = c.get("channels"), c.get("size")
    check(type(channels) is int and channels in (1, 3), "channels phải là 1 hoặc 3.")
    check(c.get("layout") in ("NCHW", "NHWC"), "layout phải là NCHW hoặc NHWC.")
    check(c.get("output_type") in ("logits", "probabilities"),
          "output_type phải là logits hoặc probabilities.")
    check(c.get("resize") in ("bilinear", "bicubic", "nearest", "lanczos"),
          "resize không được hỗ trợ.")
    check(isinstance(size, list) and len(size) == 2
          and all(type(x) is int and 1 <= x <= 2048 for x in size),
          "size phải là [rộng, cao], mỗi chiều từ 1 đến 2048.")

    def number(x):
        return type(x) in (int, float) and abs(x) <= 3.4e38 and isfinite(x)

    check(number(c.get("scale")), "scale phải là số hữu hạn.")
    for key in ("mean", "std"):
        values = c.get(key)
        check(isinstance(values, list) and len(values) == channels and all(number(v) for v in values),
              f"{key} phải có {channels} số hữu hạn.")
    check(all(v >= 1.17549435e-38 for v in c["std"]), "std phải lớn hơn 0 trong float32.")
    for key in ("name", "input_name", "output_name"):
        value = c.get(key)
        check(value is None or isinstance(value, str) and bool(value.strip()),
              f"{key} phải là chuỗi không rỗng hoặc null.")
    architecture = c.get("architecture")
    if architecture is not None:
        check(architecture in ("resnet50", "efficientnet_b0", "swin_t", "swin_s", "swin_b",
                               "swin_v2_t", "swin_v2_s", "swin_v2_b"), "Kiến trúc PyTorch chưa được hỗ trợ.")
        check(channels == 3 and c["layout"] == "NCHW", "Mô hình torchvision cần RGB và NCHW.")
    model = folder / ("model.pth" if architecture else "model.onnx")
    check(model.is_file(), f"Thiếu {model.name}.")
    check(model.stat().st_size > 0, f"{model.name} là file rỗng.")
    return c


def registry():
    result = []
    if not MODELS.is_dir():
        return result
    for folder in sorted(MODELS.iterdir()):
        if not folder.is_dir() or not re.fullmatch(r"[a-zA-Z0-9_-]+", folder.name):
            continue
        item = {"id": folder.name, "name": folder.name, "ready": False}
        try:
            c = config_for(folder)
            item.update(name=str(c.get("name") or folder.name), ready=True)
        except ValueError as exc:
            item["error"] = str(exc)
        except OSError:
            item["error"] = "Không đọc được file cấu hình hoặc mô hình."
        result.append(item)
    return result


@lru_cache(maxsize=8)
def session(path, modified):
    import onnxruntime as ort
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def resized_image(payload, c):
    from PIL import Image, ImageOps
    with Image.open(BytesIO(payload)) as image:
        if image.format not in ("JPEG", "PNG") or image.width * image.height > 20_000_000:
            raise ValueError("Chỉ hỗ trợ JPG/PNG tối đa 20 triệu pixel.")
        image = ImageOps.exif_transpose(image).convert("RGB" if c["channels"] == 3 else "L")
        return image.resize(tuple(c["size"]), getattr(Image.Resampling, c["resize"].upper()))


def preprocess(payload, c):
    import numpy as np
    arr = np.asarray(resized_image(payload, c), dtype=np.float32)
    if c["channels"] == 1:
        arr = arr[..., None]
    mean = np.asarray(c["mean"], dtype=np.float32)
    std = np.asarray(c["std"], dtype=np.float32)
    scale = float(c["scale"])
    if (mean.shape != (c["channels"],) or std.shape != mean.shape or
            not np.isfinite(mean).all() or not np.isfinite(std).all() or
            (std <= 0).any() or not np.isfinite(scale)):
        raise ValueError("Thông số chuẩn hóa không hợp lệ.")
    arr = (arr * scale - mean) / std
    if c["layout"] == "NCHW":
        arr = arr.transpose(2, 0, 1)
    return np.ascontiguousarray(arr[None], dtype=np.float32)


def predict(folder, payload):
    import numpy as np
    c = config_for(folder)
    arr = preprocess(payload, c)
    started = time.perf_counter()
    if c.get("architecture"):
        from torch_backend import load_model, infer
        model = folder / "model.pth"
        network = load_model(str(model), model.stat().st_mtime_ns, c["architecture"])
        started = time.perf_counter()
        out = infer(network, arr)
    else:
        model = folder / "model.onnx"
        s = session(str(model), model.stat().st_mtime_ns)
        x = c.get("input_name") or s.get_inputs()[0].name
        y = c.get("output_name") or s.get_outputs()[0].name
        started = time.perf_counter()
        out = np.asarray(s.run([y], {x: arr})[0], dtype=float)
    elapsed = (time.perf_counter() - started) * 1000
    if out.shape not in ((1, 4), (4,)) or not np.isfinite(out).all():
        raise ValueError("Mô hình phải trả về bốn điểm hữu hạn cho một ảnh.")
    out = out.reshape(4)
    if c["output_type"] == "logits":
        out = np.exp(out - out.max())
        out /= out.sum()
    elif (out < 0).any() or (out > 1).any() or abs(out.sum() - 1) > .01:
        raise ValueError("Đầu ra xác suất không hợp lệ. Kiểm tra output_type.")
    ranked = sorted(zip(c["labels"], out.tolist()), key=lambda p: p[1], reverse=True)
    return {"mode": "real", "model": c.get("name") or folder.name, "inference_ms": round(elapsed, 2),
            "ranked": [{"label": k, "probability": v} for k, v in ranked]}


def metrics(folder):
    d = read_json(folder / "metrics.json")
    m = d["confusion_matrix"]
    order = d["labels"]
    if (sorted(order) != sorted(LABELS) or len(m) != 4 or any(
            len(row) != 4 or any(type(v) is not int or v < 0 for v in row) for row in m)):
        raise ValueError("Ma trận cần 4 hàng, 4 cột và các số đếm không âm.")
    if not str(d.get("test_set_id") or "").strip():
        raise ValueError("Thiếu test_set_id.")
    m = [[m[order.index(a)][order.index(b)] for b in LABELS] for a in LABELS]
    n = sum(map(sum, m))
    if not n:
        raise ValueError("Tập đánh giá rỗng.")
    ratio = lambda a, b: a / b if b else 0
    pr = [ratio(m[i][i], sum(r[i] for r in m)) for i in range(4)]
    re_ = [ratio(m[i][i], sum(m[i])) for i in range(4)]
    return {"labels": LABELS, "confusion_matrix": m, "test_set_id": d["test_set_id"], "samples": n,
            "scores": {"Accuracy": sum(m[i][i] for i in range(4)) / n,
                       "Precision Macro": sum(pr) / 4, "Recall Macro": sum(re_) / 4,
                       "F1-Score Macro": sum(ratio(2*p*r, p+r) for p, r in zip(pr, re_)) / 4}}


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, body, mime="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/", "/integration.js"):
            file = "index.html" if path == "/" else "integration.js"
            mime = "text/html" if path == "/" else "text/javascript"
            return self.respond(200, (ROOT / "web" / file).read_bytes(), mime + "; charset=utf-8")
        if path == "/api/models":
            return self.respond(200, {"models": registry()})
        match = re.fullmatch(r"/api/metrics/([a-zA-Z0-9_-]+)", path)
        if match:
            try:
                return self.respond(200, metrics(MODELS / match[1]))
            except FileNotFoundError:
                return self.respond(404, {"error": "Chưa có kết quả đánh giá cho mô hình này."})
            except (OSError, ValueError, KeyError, TypeError):
                return self.respond(422, {"error": "File đánh giá không đúng định dạng."})
        return self.respond(404, {"error": "Không tìm thấy trang."})

    def do_POST(self):
        match = re.fullmatch(r"/api/predict/([a-zA-Z0-9_-]+)", urlsplit(self.path).path)
        if not match:
            return self.respond(404, {"error": "Không tìm thấy chức năng."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= MAX_BYTES:
            return self.respond(413, {"error": "Ảnh phải có dung lượng từ 1 byte đến 10 MB."})
        folder = MODELS / match[1]
        entry = next((m for m in registry() if m["id"] == match[1]), None)
        if not entry or not entry["ready"]:
            return self.respond(409, {"error": entry["error"] if entry else "Chưa tìm thấy mô hình."})
        try:
            self.connection.settimeout(15)
            return self.respond(200, predict(folder, self.rfile.read(length)))
        except ImportError:
            return self.respond(503, {"error": "Thiếu thư viện suy luận. Cài requirements.txt trước."})
        except Exception as exc:
            print(f"Inference failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return self.respond(422, {"error": "Dự đoán thất bại. Kiểm tra ảnh, file ONNX và cấu hình; xem log máy chủ."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--check-models", action="store_true", help="Kiểm tra các gói mô hình và thoát")
    parser.add_argument("--verify-models", action="store_true", help="Chạy thử từng mô hình bằng ảnh tổng hợp")
    args = parser.parse_args()
    if args.check_models or args.verify_models:
        entries = registry()
        if not entries:
            print("Chưa có gói mô hình trong models/.")
        passed = bool(entries)
        for entry in entries:
            error = entry.get("error")
            if args.verify_models and entry["ready"]:
                try:
                    from PIL import Image
                    buffer = BytesIO()
                    Image.new("RGB", (32, 32), color=(127, 127, 127)).save(buffer, "PNG")
                    predict(MODELS / entry["id"], buffer.getvalue())
                except Exception as exc:
                    error = f"Chạy thử thất bại: {exc}"
            passed &= not error
            status = "Chạy thử mô hình thành công" if args.verify_models else "Đủ file và cấu hình hợp lệ"
            print(f'{entry["id"]}: {error or status}')
        sys.exit(0 if passed else 1)
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"NeuroScan: http://127.0.0.1:{server.server_port}", flush=True)
        server.serve_forever()
