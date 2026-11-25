"""NeuroScan Streamlit demo: full proposal-aligned MRI brain tumor classification app."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

from app import LABELS, MAX_BYTES, MODELS, ROOT, config_for, metrics, predict, registry, resized_image

DUMMY_MODELS = ("ResNet50", "EfficientNetB0", "Swin Transformer")
DEMO = "Minh họa (dummy)"
REAL = "Mô hình thật"
REPORTS = ROOT / "reports"
HISTORY_MAP = {
    "resnet50": REPORTS / "pretrained/resnet50/resnet50_history.json",
    "efficientnet_b0": REPORTS / "from_scratch/efficientnet_b0/efficientnet_b0_history.json",
    "swin": REPORTS / "from_scratch/swin_tiny/swin_tiny_history.json",
}

st.set_page_config(page_title="NeuroScan · MRI Research Studio", layout="wide")
st.title("NeuroScan · Phân loại ảnh MRI não (BRISC 2025)")
st.caption("ResNet50 · EfficientNetB0 · Swin Transformer — 4 lớp: Glioma, Meningioma, Pituitary Tumor và No Tumor")


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
    p = np.random.default_rng(seed).dirichlet(np.ones(4) * 0.8)
    return {
        "mode": "dummy",
        "model": model,
        "inference_ms": round(float(np.random.default_rng(seed).uniform(8, 25)), 2),
        "ranked": sorted(
            [{"label": name, "probability": float(score)} for name, score in zip(LABELS, p)],
            key=lambda r: r["probability"],
            reverse=True,
        ),
    }


def dummy_metrics(model):
    offset = DUMMY_MODELS.index(model)
    diag, n = 82 + offset * 5, 100
    m = [[diag if i == j else n - diag if j == (i + 1) % 4 else 0
          for j in range(4)] for i in range(4)]
    per_class = [
        {"label": LABELS[i], "precision": diag / n, "recall": diag / n, "f1": diag / n, "support": n}
        for i in range(4)
    ]
    return {
        "labels": LABELS,
        "confusion_matrix": m,
        "test_set_id": "Ví dụ giả lập",
        "samples": 400,
        "scores": dict.fromkeys(("Accuracy", "Precision Macro", "Recall Macro", "F1-Score Macro"), diag / n),
        "per_class": per_class,
    }


def read_image(payload):
    if not 0 < len(payload) <= MAX_BYTES:
        raise ValueError("Ảnh phải có dung lượng tối đa 10 MB.")
    with Image.open(BytesIO(payload)) as im:
        if im.format not in ("JPEG", "PNG") or im.width * im.height > 20_000_000:
            raise ValueError("Chỉ hỗ trợ JPG/PNG tối đa 20 triệu pixel.")
        return ImageOps.exif_transpose(im).convert("RGB")


def confusion_heatmap(data):
    labels = list(data["labels"])
    matrix = data["confusion_matrix"]
    rows = []
    for i, actual in enumerate(labels):
        row_sum = sum(matrix[i]) or 1
        for j, pred in enumerate(labels):
            count = matrix[i][j]
            rate = count / row_sum
            rows.append({
                "Nhãn thực": actual,
                "Nhãn dự đoán": pred,
                "Số ảnh": count,
                "Tỷ lệ": rate,
                "Hiển thị": f"{count}\n({rate:.1%})",
            })
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(
        x=alt.X("Nhãn dự đoán:N", sort=labels, title="Nhãn dự đoán"),
        y=alt.Y("Nhãn thực:N", sort=labels, title="Nhãn thực"),
    )
    rects = base.mark_rect(cornerRadius=4, stroke="#ffffff", strokeWidth=2).encode(
        color=alt.Color("Tỷ lệ:Q", scale=alt.Scale(scheme="tealblues", domain=[0, 1]), legend=alt.Legend(format="%")),
        tooltip=["Nhãn thực", "Nhãn dự đoán", "Số ảnh", alt.Tooltip("Tỷ lệ:Q", format=".2%")],
    )
    text = base.mark_text(fontSize=12, fontWeight="bold", lineBreak="\n").encode(
        text="Hiển thị:N",
        color=alt.condition(alt.datum["Tỷ lệ"] > 0.55, alt.value("white"), alt.value("#102e42")),
    )
    return (rects + text).properties(height=360)


def load_history(model_id):
    path = HISTORY_MAP.get(model_id)
    if path and path.is_file():
        return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    return None


real_entries = registry()
real_ready_count = sum(1 for m in real_entries if m["ready"])
default_index = 1 if real_ready_count > 0 else 0

with st.sidebar:
    st.header("Cấu hình hệ thống")
    mode = st.radio("Chế độ chạy", (DEMO, REAL), index=default_index)
    if mode == REAL:
        st.caption(f"Mô hình sẵn sàng: **{real_ready_count}/{len(real_entries)}**")
        for item in real_entries:
            if item["ready"]:
                st.success(f"{item['name']}: Sẵn sàng")
            else:
                st.warning(f"{item['name']}: {item['error']}")
    else:
        st.info("Chế độ minh họa dùng điểm giả lập, không cần checkpoint.")
    st.divider()
    st.caption("Chỉ phục vụ mục đích nghiên cứu & đồ án học tập, không thay thế chẩn đoán y khoa.")

entries = real_entries if mode == REAL else [
    {"id": name, "name": name, "ready": True} for name in DUMMY_MODELS
]
lookup = {m["id"]: m["name"] for m in entries}
available = [m["id"] for m in entries if m["ready"]]

if mode == REAL and not available:
    st.warning("Chưa có mô hình sẵn sàng. Thêm `model.pth` hoặc `model.onnx` và `model_config.json` vào `models/`.")

tab_predict, tab_compare, tab_eval, tab_data = st.tabs([
    "1. Phân tích & Dự đoán",
    "2. So sánh 3 mô hình",
    "3. Đánh giá & Hiệu quả tính toán",
    "4. Dữ liệu & Quy trình",
])

with tab_predict:
    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Bước 1 · Tải ảnh MRI & tiền xử lý")
        upload = st.file_uploader("Chọn ảnh MRI não (JPG, PNG · tối đa 10 MB)", type=["jpg", "jpeg", "png"])
        use_sample = st.checkbox("Dùng hình mô phỏng") if mode == DEMO and upload is None else False
        payload = upload.getvalue() if upload is not None else sample_bytes() if use_sample else None
        selected_model = st.selectbox("Chọn mô hình phân loại", available, format_func=lookup.get,
                                      disabled=not available)
        img = None
        if payload is not None:
            try:
                img = read_image(payload)
                c_orig, c_pre = st.columns(2)
                with c_orig:
                    st.image(img, caption=f"Ảnh gốc ({img.width} × {img.height} px)", width=240)
                cfg = config_for(MODELS / selected_model) if mode == REAL and selected_model else None
                size = cfg["size"] if cfg else [224, 224]
                method = cfg["resize"] if cfg else "bilinear"
                preview_img = resized_image(payload, cfg) if cfg else img.resize(
                    size, getattr(Image.Resampling, method.upper()))
                with c_pre:
                    st.image(preview_img, caption=f"Sau tiền xử lý ({size[0]} × {size[1]} px)", width=240)
                if cfg:
                    st.caption(
                        f"Tiền xử lý: RGB {cfg['channels']} kênh · Resize {cfg['resize']} · "
                        f"Chuẩn hóa ImageNet mean={cfg['mean']}, std={cfg['std']} · Layout {cfg['layout']}"
                    )
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                img = None
                st.error(f"Không xử lý được ảnh: {exc}")

    def run_model(key):
        return dummy_prediction(key, payload) if mode == DEMO else predict(MODELS / key, payload)

    def context(keys):
        return mode, tuple(keys), sha256(payload or b"").hexdigest()

    with right:
        st.subheader("Bước 2 · Kết quả dự đoán")
        current = context([selected_model])
        if st.button("Chạy dự đoán", type="primary", width="stretch",
                     disabled=img is None or not selected_model):
            st.session_state.pop("prediction", None)
            try:
                st.session_state.prediction = current, run_model(selected_model)
            except Exception as exc:
                st.error(f"Không chạy được mô hình: {exc}")
        saved = st.session_state.get("prediction")
        if saved and saved[0] == current and img is not None:
            res = saved[1]
            top = res["ranked"]
            m1, m2, m3 = st.columns([1.4, 1, 1])
            m1.metric("Lớp dự đoán (Top-1)", top[0]["label"])
            m2.metric("Độ tin cậy", f"{top[0]['probability']:.2%}")
            m3.metric("Thời gian dự đoán", f"{res.get('inference_ms', 0):.2f} ms")
            st.markdown("**Top-3 kết quả có xác suất cao nhất**")
            for rank_idx, item in enumerate(top[:3], start=1):
                st.progress(item["probability"], text=f"#{rank_idx} {item['label']} · {item['probability']:.2%}")
            with st.expander("Xem xác suất đầy đủ 4 lớp"):
                st.dataframe(
                    pd.DataFrame([
                        {"Hạng": idx, "Lớp": r["label"], "Xác suất": f"{r['probability']:.2%}"}
                        for idx, r in enumerate(top, start=1)
                    ]),
                    hide_index=True,
                    width="stretch",
                )
            st.download_button(
                "Tải kết quả (JSON)",
                data=json.dumps(res, ensure_ascii=False, indent=2),
                file_name=f"neuroscan_{selected_model}.json",
                mime="application/json",
            )
        else:
            st.info("Tải ảnh MRI ở bên trái và nhấn **Chạy dự đoán** để xem nhãn dự đoán, độ tin cậy và Top-3.")

with tab_compare:
    st.subheader("So sánh dự đoán giữa 3 mô hình trên cùng một ảnh")
    selection = st.multiselect("Chọn các mô hình so sánh (tối đa 3)", available, default=available[:3],
                               max_selections=3, format_func=lookup.get)
    cmp_key = context(selection)
    if st.button("Chạy so sánh trên cùng ảnh", type="primary", disabled=img is None or not selection):
        rows, raw_results = [], []
        for item in selection:
            try:
                res = run_model(item)
                raw_results.append(res)
                ranked = res["ranked"]
                rows.append({
                    "Mô hình": lookup[item],
                    "Top-1 dự đoán": ranked[0]["label"],
                    "Độ tin cậy": f"{ranked[0]['probability']:.2%}",
                    "Top-2": f"{ranked[1]['label']} ({ranked[1]['probability']:.1%})",
                    "Thời gian (ms)": f"{res.get('inference_ms', 0):.2f}",
                    "Trạng thái": "Thành công",
                })
            except Exception as exc:
                rows.append({
                    "Mô hình": lookup[item],
                    "Top-1 dự đoán": "—",
                    "Độ tin cậy": "—",
                    "Top-2": "—",
                    "Thời gian (ms)": "—",
                    "Trạng thái": str(exc),
                })
        st.session_state.comparison = cmp_key, rows, raw_results

    saved_cmp = st.session_state.get("comparison")
    if saved_cmp and saved_cmp[0] == cmp_key and img is not None:
        _, rows, raw_results = saved_cmp
        st.dataframe(rows, hide_index=True, width="stretch")
        if raw_results:
            prob_rows = []
            for res in raw_results:
                for item in res["ranked"]:
                    prob_rows.append({
                        "Mô hình": res["model"],
                        "Lớp": item["label"],
                        "Xác suất": item["probability"],
                    })
            chart = alt.Chart(pd.DataFrame(prob_rows)).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X("Lớp:N", sort=list(LABELS)),
                y=alt.Y("Xác suất:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Mô hình:N", scale=alt.Scale(scheme="tealblues")),
                xOffset="Mô hình:N",
                tooltip=["Mô hình", "Lớp", alt.Tooltip("Xác suất:Q", format=".2%")],
            ).properties(height=320, title="Phân bố xác suất 4 lớp giữa các mô hình")
            st.altair_chart(chart, width="stretch")
    else:
        st.caption("Chọn ảnh MRI ở Tab 1 rồi nhấn **Chạy so sánh trên cùng ảnh** để đối chiếu cả 3 mô hình.")

with tab_eval:
    st.subheader("So sánh tổng hợp hiệu suất & hiệu quả tính toán trên tập Test")
    overview_rows = []
    chart_rows = []
    for item in entries:
        try:
            d = dummy_metrics(item["id"]) if mode == DEMO else metrics(MODELS / item["id"])
        except (FileNotFoundError, ValueError, TypeError, OSError, KeyError):
            continue
        eff = d.get("efficiency", {})
        mode_tag = {"pretrained": "Pretrained", "from_scratch": "From scratch"}.get(
            eff.get("training_mode"), "—"
        )
        overview_rows.append({
            "Mô hình": item["name"],
            "Chế độ train": mode_tag,
            "Tập Test": d["test_set_id"],
            "Accuracy": f"{d['scores']['Accuracy']:.2%}",
            "Precision Macro": f"{d['scores']['Precision Macro']:.2%}",
            "Recall Macro": f"{d['scores']['Recall Macro']:.2%}",
            "Macro F1-Score": f"{d['scores']['F1-Score Macro']:.2%}",
            "Thời gian train": f"{eff['train_time_sec'] / 60:.1f} phút" if "train_time_sec" in eff else "—",
            "Dự đoán (T4, batch=1)": f"{eff['inference_ms_per_image_single']:.2f} ms" if "inference_ms_per_image_single" in eff else "—",
            "Tham số": f"{eff['num_params_million']:.2f}M" if "num_params_million" in eff else "—",
        })
        chart_rows.append({
            "Mô hình": item["name"],
            "Accuracy": d["scores"]["Accuracy"],
            "Macro F1": d["scores"]["F1-Score Macro"],
            "Train (phút)": eff.get("train_time_sec", 0) / 60 if "train_time_sec" in eff else None,
            "Latency (ms)": eff.get("inference_ms_per_image_single"),
        })

    if overview_rows:
        st.dataframe(overview_rows, hide_index=True, width="stretch")
        cdf = pd.DataFrame(chart_rows)
        c1, c2 = st.columns(2)
        with c1:
            melted = cdf.melt(id_vars=["Mô hình"], value_vars=["Accuracy", "Macro F1"],
                              var_name="Chỉ số", value_name="Giá trị")
            score_chart = alt.Chart(melted).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X("Mô hình:N"),
                y=alt.Y("Giá trị:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Chỉ số:N", scale=alt.Scale(range=["#087f83", "#102e42"])),
                xOffset="Chỉ số:N",
                tooltip=["Mô hình", "Chỉ số", alt.Tooltip("Giá trị:Q", format=".2%")],
            ).properties(height=280, title="Accuracy & Macro F1 trên tập Test")
            st.altair_chart(score_chart, width="stretch")
        with c2:
            if cdf["Latency (ms)"].notna().any():
                lat_chart = alt.Chart(cdf.dropna(subset=["Latency (ms)"])).mark_bar(
                    cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#12aca2"
                ).encode(
                    x=alt.X("Mô hình:N"),
                    y=alt.Y("Latency (ms):Q", title="Thời gian dự đoán batch=1 (ms/ảnh, T4)"),
                    tooltip=["Mô hình", alt.Tooltip("Latency (ms):Q", format=".2f"),
                             alt.Tooltip("Train (phút):Q", format=".1f")],
                ).properties(height=280, title="Thời gian dự đoán trung bình (ms/ảnh, thấp hơn = nhanh hơn)")
                st.altair_chart(lat_chart, width="stretch")
            else:
                st.info("Chuyển sang **Mô hình thật** để xem biểu đồ thời gian huấn luyện và thời gian dự đoán.")

    st.divider()
    st.subheader("Chi tiết Confusion Matrix & chỉ số từng lớp")
    eval_key = st.selectbox("Chọn mô hình xem chi tiết", list(lookup), format_func=lookup.get, disabled=not lookup)
    if eval_key:
        try:
            data = dummy_metrics(eval_key) if mode == DEMO else metrics(MODELS / eval_key)
        except FileNotFoundError:
            st.info("Chưa có `metrics.json` cho mô hình này.")
        except (ValueError, TypeError, OSError, KeyError) as exc:
            st.error(f"Không đọc được dữ liệu đánh giá: {exc}")
        else:
            st.caption(f"{data['test_set_id']} · {data['samples']} ảnh Test")
            for col, (label, value) in zip(st.columns(4), data["scores"].items()):
                col.metric(label, f"{value:.2%}")
            eff = data.get("efficiency", {})
            if eff:
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Thời gian train", f"{eff['train_time_sec'] / 60:.1f} phút" if "train_time_sec" in eff else "—")
                e2.metric("Dự đoán (batch=1)", f"{eff['inference_ms_per_image_single']:.2f} ms" if "inference_ms_per_image_single" in eff else "—")
                e3.metric("Số tham số", f"{eff['num_params_million']:.2f}M" if "num_params_million" in eff else "—")
                e4.metric("Best / Tổng epoch", f"{eff.get('best_epoch', '—')} / {eff.get('epochs_run', '—')}")

            hm_col, pc_col = st.columns([1.2, 1])
            with hm_col:
                st.markdown("**Confusion Matrix (Heatmap)**")
                st.altair_chart(confusion_heatmap(data), width="stretch")
            with pc_col:
                st.markdown("**Chỉ số theo từng lớp**")
                if "per_class" in data:
                    st.dataframe(
                        pd.DataFrame([
                            {
                                "Lớp": c["label"],
                                "Precision": f"{c['precision']:.2%}",
                                "Recall": f"{c['recall']:.2%}",
                                "F1-Score": f"{c['f1']:.2%}",
                                "Số ảnh": c["support"],
                            }
                            for c in data["per_class"]
                        ]),
                        hide_index=True,
                        width="stretch",
                    )

            if mode == REAL:
                hist_df = load_history(eval_key)
                if hist_df is not None and not hist_df.empty:
                    st.markdown("**Quá trình huấn luyện theo Epoch (Learning Curves)**")
                    hc1, hc2 = st.columns(2)
                    with hc1:
                        loss_m = hist_df.melt(id_vars=["epoch"], value_vars=["train_loss", "val_loss"],
                                              var_name="Tập", value_name="Loss")
                        st.altair_chart(
                            alt.Chart(loss_m).mark_line(point=True).encode(
                                x=alt.X("epoch:Q", title="Epoch"),
                                y=alt.Y("Loss:Q", title="Cross-Entropy Loss"),
                                color="Tập:N",
                                tooltip=["epoch", "Tập", alt.Tooltip("Loss:Q", format=".4f")],
                            ).properties(height=260, title="Train vs Validation Loss"),
                            width="stretch",
                        )
                    with hc2:
                        f1_m = hist_df.melt(id_vars=["epoch"], value_vars=["train_macro_f1", "val_macro_f1"],
                                            var_name="Tập", value_name="Macro F1")
                        st.altair_chart(
                            alt.Chart(f1_m).mark_line(point=True).encode(
                                x=alt.X("epoch:Q", title="Epoch"),
                                y=alt.Y("Macro F1:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                                color="Tập:N",
                                tooltip=["epoch", "Tập", alt.Tooltip("Macro F1:Q", format=".2%")],
                            ).properties(height=260, title="Train vs Validation Macro F1"),
                            width="stretch",
                        )

with tab_data:
    st.subheader("Tập dữ liệu BRISC 2025 & Quy trình tiền xử lý")
    st.markdown(
        "**Quy trình:** Khám phá dữ liệu → Phân tích phân bố 4 lớp → Kiểm tra ảnh lỗi & trùng lặp → "
        "Chia Train / Validation / Test (stratified, seed = 42) → Resize 224×224 → Chuẩn hóa ImageNet → "
        "Data Augmentation chỉ trên Train → Class Weight cho Cross-Entropy Loss → Huấn luyện & so sánh 3 mô hình."
    )
    split_dir = REPORTS / "data_split"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ảnh gốc BRISC 2025", "6.000")
    m2.metric("Trùng lặp đã lọc / Lỗi", "170 / 0")
    m3.metric("Ảnh sạch sử dụng", "5.830")
    m4.metric("Train / Val / Test", "4.118 / 727 / 985")

    d1, d2 = st.columns([1.2, 1])
    dist_img = split_dir / "class_distribution.png"
    dist_csv = split_dir / "class_distribution.csv"
    with d1:
        if dist_img.is_file():
            st.image(str(dist_img), caption="Phân bố 4 lớp trên BRISC 2025", width="stretch")
    with d2:
        if dist_csv.is_file():
            st.markdown("**Bảng phân bố số lượng ảnh gốc theo lớp**")
            st.dataframe(pd.read_csv(dist_csv), hide_index=True, width="stretch")
        st.markdown(
            "- **Thứ tự nhãn thống nhất:** `glioma` (0), `meningioma` (1), `no_tumor` (2), `pituitary` (3)\n"
            "- **Augmentation (chỉ tập Train):** RandomResizedCrop (80–100%), lật ngang, xoay ±15°, dịch 5%, sáng/tương phản ±20%\n"
            "- **Tối ưu hóa:** Adam (ResNet50, EfficientNetB0) & AdamW (Swin-Tiny) với Cross-Entropy Loss có Class Weight"
        )
