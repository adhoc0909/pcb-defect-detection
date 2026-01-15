import streamlit as st
from pathlib import Path
from PIL import Image
import tempfile
import time

# ===== project imports =====
from pcbdet.models.factory import build_model, ModelSpec
from pcbdet.infer.heic import load_image_auto  # heic -> jpg 자동 처리

# ===========================
# Streamlit 기본 설정
# ===========================
st.set_page_config(
    page_title="PCB Defect Detection PoC",
    layout="wide",
)

st.title("🧪 PCB Defect Detection PoC")
st.markdown(
    """
- **Model**: YOLOv8 Baseline / SPDConv  
- **Purpose**: PoC inference & qualitative comparison  
- **Note**: Prediction images are **NOT saved**. Visualization is in-app only.
"""
)

# ===========================
# Sidebar - 설정
# ===========================
st.sidebar.header("⚙️ Settings")

model_kind = st.sidebar.radio(
    "Model type",
    options=["baseline", "spdconv"],
    index=0,
)

weights_path = st.sidebar.text_input(
    "Weights (.pt path)",
    value="/Users/leehw/Documents/likelion/pcb-defect-detection/weights/yolov8s_spdconv_best.pt",
    help="e.g. /path/to/best.pt",
)

imgsz = st.sidebar.selectbox(
    "Image size",
    options=[640, 512, 416],
    index=0,
)

conf = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.01,
    max_value=0.9,
    value=0.25,
    step=0.01,
)

iou = st.sidebar.slider(
    "IoU threshold",
    min_value=0.3,
    max_value=0.9,
    value=0.6,
    step=0.05,
)

device = st.sidebar.selectbox(
    "Device",
    options=["cuda:0", "cpu"],
    index=0,
)

run_btn = st.sidebar.button("🚀 Run Inference")

# ===========================
# Main - Image input
# ===========================
st.subheader("📤 Input Image")

uploaded = st.file_uploader(
    "Upload an image (jpg / png / heic)",
    type=["jpg", "jpeg", "png", "heic", "heif"],
)

if uploaded is not None:
    try:
        img_preview = Image.open(uploaded)
        st.image(img_preview, caption="Original Image", use_column_width=True)
    except Exception:
        st.info("Preview not available. Image will be loaded for inference.")

# ===========================
# Inference
# ===========================
if run_btn:
    if uploaded is None:
        st.warning("Please upload an image.")
        st.stop()

    if not weights_path:
        st.warning("Please provide weights path.")
        st.stop()

    weights_path = Path(weights_path)
    if not weights_path.exists():
        st.error(f"Weights not found: {weights_path}")
        st.stop()

    # 임시 파일로 저장 (Streamlit uploader 대응)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        raw_img_path = tmpdir / uploaded.name
        with open(raw_img_path, "wb") as f:
            f.write(uploaded.getbuffer())

        # HEIC/HEIF 자동 변환
        img_path = load_image_auto(raw_img_path)

        st.info(f"Running inference with **{model_kind}** model...")

        # ===========================
        # 모델 로딩 (캐시)
        # ===========================
        @st.cache_resource(show_spinner=False)
        def load_model(kind: str, weights: str):
            return build_model(ModelSpec(kind=kind, weights=str(weights)))

        model = load_model(model_kind, str(weights_path))

        # ===========================
        # Predict + Timing
        # ===========================
        t0 = time.perf_counter()

        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            save=False,       # ❌ 파일 저장 안 함
            verbose=False,
        )

        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000.0

        r = results[0]

        # ===========================
        # Visualization
        # ===========================
        st.subheader("📊 Prediction Result")

        vis_bgr = r.plot()            # numpy BGR
        vis_rgb = vis_bgr[..., ::-1]  # BGR -> RGB
        st.image(vis_rgb, caption="Prediction (in-app only)", use_column_width=True)

        # ===========================
        # Inference time
        # ===========================
        st.markdown("### ⏱️ Inference Time")
        st.metric(
            label="Single-image inference",
            value=f"{infer_ms:.2f} ms",
            delta=None,
        )

        # ===========================
        # Detection table
        # ===========================
        if len(r.boxes) == 0:
            st.info("No detections.")
        else:
            st.markdown("### 📦 Detected Objects")
            table = []
            for b in r.boxes:
                table.append({
                    "class": r.names[int(b.cls)],
                    "confidence": float(b.conf),
                    "x1": float(b.xyxy[0][0]),
                    "y1": float(b.xyxy[0][1]),
                    "x2": float(b.xyxy[0][2]),
                    "y2": float(b.xyxy[0][3]),
                })
            st.dataframe(table, use_container_width=True)
