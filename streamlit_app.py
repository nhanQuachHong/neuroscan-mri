"""NeuroScan Streamlit demo: dummy mode and locally exported ONNX models."""
from hashlib import sha256
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

from app import LABELS, MAX_BYTES, MODELS, config_for, metrics, predict, registry

DUMMY_MODELS = ("MobileNetV2", "ResNet50", "EfficientNetB0")
DEMO = "Minh họa (dummy)"
REAL = "Mô hình thật"

st.set_page_config(page_title="NeuroScan · MRI Research Studio", layout="wide")
st.title("NeuroScan")
st.caption("MRI RESEARCH STUDIO · Phân loại ảnh MRI não thành 4 lớp")


def sample_bytes():
    im = Image.new("L", (224, 224), 16)
    draw = ImageDraw.Draw(im)
    for offset in range(0, 72, 9):
        draw.ellipse((23 + offset, 10 + offset, 201 - offset, 196 - offset),
                     outline=65 + offset, width=2)
    draw.text((35, 205), "HINH MO PHONG - KHONG PHAI MRI", fill=220)
    buffer = BytesIO()
    im.save(buffer, format="PNG")
    return buffer.getvalue()


def dummy_prediction(model, payload):
    seed = int.from_bytes(sha256(model.encode() + payload).digest()[:8], "big")
    p = np.random.default_rng(seed).dirichlet(np.ones(4) * .8)
    return {"model": model, "ranked": sorted(
        [{"label": name, "probability": float(score)} for name, score in zip(LABELS, p)],
        key=lambda r: r["probability"], reverse=True)}


def dummy_metrics(model):
    offset = DUMMY_MODELS.index(model)
    diag, n = 82 + offset * 5, 100
    m = [[diag if i == j else n - diag if j == (i + 1) % 4 else 0
          for j in range(4)] for i in range(4)]
    return {"labels": LABELS, "confusion_matrix": m, "test_set_id": "Ví dụ giả lập",
            "samples": 400, "scores": dict.fromkeys(
                ("Accuracy", "Precision Macro", "Recall Macro", "F1-Score Macro"), diag / n)}


def read_image(payload):
    if not 0 < len(payload) <= MAX_BYTES:
        raise ValueError("Ảnh phải có dung lượng tối đa 10 MB.")
    with Image.open(BytesIO(payload)) as im:
        if im.format not in ("JPEG", "PNG") or im.width * im.height > 20_000_000:
            raise ValueError("Chỉ hỗ trợ JPG/PNG tối đa 20 triệu pixel.")
        return ImageOps.exif_transpose(im).convert("RGB")


mode = st.sidebar.radio("Chế độ", (DEMO, REAL))
entries = registry() if mode == REAL else [
    {"id": name, "name": name, "ready": True} for name in DUMMY_MODELS]
lookup = {m["id"]: m["name"] for m in entries}
available = [m["id"] for m in entries if m["ready"]]

if mode == DEMO:
    st.info("Chế độ minh họa: dự đoán và chỉ số đều là dữ liệu giả lập.")
else:
    st.info("Chế độ mô hình thật · Dùng cho đồ án nghiên cứu.")
    for item in entries:
        if not item["ready"]:
            st.sidebar.warning(item["name"] + ": " + item["error"])
    if not available:
        st.warning("Chưa có mô hình sẵn sàng. Thêm model.onnx và model_config.json vào models/.")

left, right = st.columns(2)
with left:
    st.subheader("Ảnh đầu vào")
    upload = st.file_uploader("Tải ảnh MRI", type=["jpg", "jpeg", "png"])
    sample = st.checkbox("Dùng hình mô phỏng") if mode == DEMO and upload is None else False
    payload = upload.getvalue() if upload is not None else sample_bytes() if sample else None
    model = st.selectbox("Mô hình dự đoán", available, format_func=lookup.get,
                         disabled=not available)
    img = None
    if payload is not None:
        try:
            img = read_image(payload)
            st.image(img, caption="Hình mô phỏng" if sample else "Ảnh đã tải", width=260)
            cfg = config_for(MODELS / model) if mode == REAL and model else None
            if mode == DEMO or cfg:
                size = cfg["size"] if cfg else [224, 224]
                method = cfg["resize"] if cfg else "bilinear"
                view = img.convert("L" if cfg and cfg["channels"] == 1 else "RGB")
                with st.expander("Xem ảnh sau đổi kích thước"):
                    st.image(view.resize(size, getattr(Image.Resampling, method.upper())),
                             caption=f"{size[0]} × {size[1]} px", width=224)
                    if cfg:
                        st.caption(f"Tensor: {cfg['layout']} · {cfg['channels']} kênh")
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            img = None
            st.error(f"Không xử lý được ảnh: {exc}")


def run_model(key):
    return dummy_prediction(key, payload) if mode == DEMO else predict(MODELS / key, payload)


def context(keys):
    return mode, tuple(keys), sha256(payload or b"").hexdigest()


with right:
    st.subheader("Kết quả phân loại")
    current = context([model])
    if st.button("Chạy dự đoán", type="primary", disabled=img is None or not model):
        st.session_state.pop("prediction", None)
        try:
            st.session_state.prediction = current, run_model(model)
        except Exception as exc:
            st.error(f"Không chạy được mô hình: {exc}")
    saved = st.session_state.get("prediction")
    if saved and saved[0] == current and img is not None:
        top = saved[1]["ranked"][:3]
        st.metric(top[0]["label"], f"{top[0]['probability']:.1%}")
        st.caption("Điểm minh họa" if mode == DEMO else "Điểm xác suất do mô hình trả về")
        st.markdown("**Top-3**")
        for item in top:
            st.progress(item["probability"], text=f"{item['label']} · {item['probability']:.1%}")
    else:
        st.caption("Chọn ảnh và mô hình để xem kết quả.")

st.divider()
comparison, assessment = st.tabs(["So sánh 3 mô hình", "Đánh giá"])
with comparison:
    selection = st.multiselect("Mô hình đem so sánh", available, default=available[:3],
                               max_selections=3, format_func=lookup.get)
    key = context(selection)
    if st.button("So sánh trên cùng ảnh", disabled=img is None or not selection):
        rows = []
        for item in selection:
            try:
                top = run_model(item)["ranked"][0]
                rows.append({"Mô hình": lookup[item], "Top-1": top["label"],
                             "Điểm": f"{top['probability']:.1%}", "Trạng thái": "Đã chạy"})
            except Exception as exc:
                rows.append({"Mô hình": lookup[item], "Top-1": "—", "Điểm": "—",
                             "Trạng thái": str(exc)})
        st.session_state.comparison = key, rows
    saved = st.session_state.get("comparison")
    if saved and saved[0] == key and img is not None:
        st.dataframe(saved[1], hide_index=True)
    if mode == REAL and len(available) < 3:
        st.caption("Có thể thử các mô hình đã nhận; thêm đủ 3 mô hình để so sánh đầy đủ.")

with assessment:
    key = st.selectbox("Mô hình đánh giá", list(lookup), format_func=lookup.get,
                       disabled=not lookup)
    if key:
        try:
            data = dummy_metrics(key) if mode == DEMO else metrics(MODELS / key)
        except FileNotFoundError:
            st.info("Chưa có metrics.json cho mô hình này.")
        except (ValueError, TypeError, OSError, KeyError) as exc:
            st.error(f"Không đọc được dữ liệu đánh giá: {exc}")
        else:
            st.caption(f"{data['test_set_id']} · {data['samples']} ảnh")
            for col, (label, value) in zip(st.columns(4), data["scores"].items()):
                col.metric(label, f"{value:.1%}")
            st.markdown("**Confusion Matrix** · hàng là nhãn thật, cột là nhãn dự đoán")
            st.dataframe(pd.DataFrame(data["confusion_matrix"], index=data["labels"],
                                      columns=data["labels"]))
