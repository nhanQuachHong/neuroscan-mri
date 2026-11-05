"""FastAPI service for the NeuroScan web frontend."""
import re
from io import BytesIO

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

import app as core

api = FastAPI(title="NeuroScan MRI")
WEB = core.ROOT / "web"


def problem(code, message):
    return JSONResponse({"error": message}, status_code=code)


def model_for(model_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", model_id):
        return None
    return next((m for m in core.registry() if m["id"] == model_id), None)


async def image_body(request):
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > core.MAX_BYTES:
            return None
        data.extend(chunk)
    return bytes(data) or None


@api.get("/", include_in_schema=False)
def home():
    return FileResponse(WEB / "index.html")


@api.get("/integration.js", include_in_schema=False)
def javascript():
    return FileResponse(WEB / "integration.js", media_type="text/javascript")


@api.get("/api/models")
def list_models():
    return {"models": core.registry()}


def run_prediction(payload, model):
    return core.predict(core.MODELS / model["id"], payload)


@api.post("/api/predict/{model_id}")
async def predict(model_id: str, request: Request):
    model = model_for(model_id)
    if not model or not model["ready"]:
        return problem(409, model["error"] if model else "Chưa tìm thấy mô hình.")
    payload = await image_body(request)
    if payload is None:
        return problem(413, "Ảnh rỗng hoặc vượt quá 10 MB.")
    try:
        return await run_in_threadpool(run_prediction, payload, model)
    except ImportError:
        return problem(503, "Thiếu thư viện chạy mô hình. Xem README.md.")
    except Exception as exc:
        return problem(422, f"Không chạy được mô hình: {exc}")


def compare_models(payload, entries):
    rows = []
    for model in entries:
        if not model["ready"]:
            rows.append({"model": model["name"], "error": model["error"]})
            continue
        try:
            rows.append(run_prediction(payload, model))
        except Exception as exc:
            rows.append({"model": model["name"], "error": str(exc)})
    return {"results": rows}


@api.post("/api/compare")
async def compare(request: Request):
    entries = core.registry()
    slots = {"resnet50": "ResNet50", "efficientnet_b0": "EfficientNetB0", "swin": "Swin Transformer"}
    by_id = {m["id"]: m for m in entries}
    if by_id and set(by_id) <= slots.keys():
        entries = [by_id.get(k, {"id": k, "name": name, "ready": False,
                                "error": "Chưa có checkpoint."}) for k, name in slots.items()]
    if not entries:
        return problem(409, "Chưa có mô hình để so sánh.")
    if len(entries) > 3:
        return problem(422, "Chỉ đặt tối đa ba gói mô hình trong models/ khi so sánh.")
    payload = await image_body(request)
    if payload is None:
        return problem(413, "Ảnh rỗng hoặc vượt quá 10 MB.")
    return await run_in_threadpool(compare_models, payload, entries)


@api.get("/api/metrics/{model_id}")
def get_metrics(model_id: str):
    if not model_for(model_id):
        return problem(404, "Không tìm thấy mô hình.")
    try:
        return core.metrics(core.MODELS / model_id)
    except FileNotFoundError:
        return problem(404, "Chưa có kết quả đánh giá cho mô hình này.")
    except (OSError, ValueError, KeyError, TypeError):
        return problem(422, "File đánh giá không đúng định dạng.")


@api.get("/api/health")
def health():
    return {"status": "ok"}


@api.get("/api/metrics")
def metrics_overview():
    rows = []
    for item in core.registry():
        response = get_metrics(item["id"])
        if isinstance(response, JSONResponse):
            rows.append({"model": item["name"], "available": False})
        else:
            rows.append({"model": item["name"], "available": True,
                         "test_set_id": response["test_set_id"], "scores": response["scores"]})
    return {"models": rows}


@api.post("/api/preview/{model_id}")
async def preview(model_id: str, request: Request):
    model = model_for(model_id)
    if not model or not model["ready"]:
        return problem(409, model["error"] if model else "Chưa tìm thấy mô hình.")
    payload = await image_body(request)
    if payload is None:
        return problem(413, "Ảnh rỗng hoặc vượt quá 10 MB.")

    def preview_bytes():
        config = core.config_for(core.MODELS / model_id)
        image = core.resized_image(payload, config)
        buffer = BytesIO()
        image.save(buffer, "PNG")
        return buffer.getvalue()

    try:
        return Response(await run_in_threadpool(preview_bytes), media_type="image/png",
                        headers={"Cache-Control": "no-store"})
    except Exception:
        return problem(422, "Không thể đọc hoặc tiền xử lý ảnh.")
