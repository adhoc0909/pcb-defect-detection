import os
import time
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from pcbdet.models.factory import build_model, ModelSpec
from pcbdet.infer.heic import load_image_auto

# ===========================
# LLM (Optional) - OpenAI
# ===========================
def _openai_client():
    """
    Lazy import to avoid hard dependency if user doesn't use LLM.
    Requires:
      pip install openai
    Needs env:
      OPENAI_API_KEY=...
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise ImportError("LLM requires 'openai'. Install: pip install openai") from e

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it in your shell environment.")
    return OpenAI(api_key=api_key)


def summarize_detections(result):
    """
    Ultralytics Results -> compact JSON-like summary for LLM + UI.
    """
    r = result
    names = r.names

    dets = []
    for b in r.boxes:
        cls_id = int(b.cls)
        dets.append(
            {
                "class": names.get(cls_id, str(cls_id)),
                "conf": float(b.conf),
                "xyxy": [float(x) for x in b.xyxy[0].tolist()],
            }
        )
    return dets


def decide_grade(dets, fail_threshold=0.70):
    """
    Simple rule-based grading:
    - FAIL if any detection confidence >= fail_threshold
    - PASS if no detections
    - WARNING if detections exist but all below threshold
    """
    if len(dets) == 0:
        return "PASS", "No defects detected."

    max_conf = max(d["conf"] for d in dets)
    if max_conf >= fail_threshold:
        return "FAIL", f"At least one defect confidence >= {fail_threshold:.2f}."
    return "WARNING", f"Defects detected but max confidence < {fail_threshold:.2f}."


def generate_report_llm(dets, grade, rationale, model_kind, infer_ms, lang="Korean"):
    """
    LLM writes a short inspection report using structured inputs.
    """
    client = _openai_client()

    # keep prompt short & structured
    system = (
        "You are an industrial inspection assistant for PCB defect detection. "
        "Write concise, actionable inspection reports. "
        "Never invent detections not provided. "
        "If information is insufficient, say so."
    )

    user = f"""
Write an inspection report in {lang}.

Inputs:
- Model: {model_kind}
- Inference time (ms): {infer_ms:.2f}
- Decision: {grade} (rule-based)
- Rationale: {rationale}
- Detections (list): {dets}

Report requirements:
- 1) Overall assessment (PASS/WARNING/FAIL)
- 2) Detected defects summary (by class, count, notable confidence)
- 3) Possible causes (brief, plausible)
- 4) Recommended actions (3 bullets, practical)
- 5) Notes/limitations (1-2 lines)
Keep it under ~150-200 words.
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content


# ===========================
# Streamlit App
# ===========================
st.set_page_config(page_title="PCB Defect Detection PoC", layout="wide")
st.title("🧪 PCB Defect Detection PoC")

st.markdown(
    """
- **Model**: YOLOv8 Baseline / SPDConv  
- **PoC**: single-image inference + in-app visualization (no file saving)  
- **Optional**: LLM-based inspection report (OpenAI)
"""
)

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("⚙️ Settings")

model_kind = st.sidebar.radio("Model type", ["baseline", "spdconv"], index=0)

weights_path_str = st.sidebar.text_input(
    "Weights (.pt path)",
    value="/Users/leehw/Documents/likelion/pcb-defect-detection/weights/yolov8s_spdconv_best.pt",
    help="e.g. /path/to/best.pt",
)

imgsz = st.sidebar.selectbox("Image size", [640, 512, 416], index=0)

conf = st.sidebar.slider("Confidence threshold", 0.01, 0.9, 0.25, 0.01)
iou = st.sidebar.slider("IoU threshold", 0.3, 0.9, 0.6, 0.05)

device = st.sidebar.selectbox("Device", ["cuda:0", "cpu"], index=0)

# rule threshold
st.sidebar.subheader("✅ Rule-based decision")
fail_threshold = st.sidebar.slider("FAIL threshold (max conf ≥)", 0.10, 0.99, 0.70, 0.01)

# LLM options
st.sidebar.subheader("📝 LLM report (optional)")
use_llm = st.sidebar.checkbox("Generate report with LLM", value=False)
llm_lang = st.sidebar.selectbox("Report language", ["Korean", "English"], index=0)

run_btn = st.sidebar.button("🚀 Run Inference")

# ---------------------------
# Main: upload
# ---------------------------
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
        st.info("Preview not available for this file. It will be converted/loaded for inference.")

# ---------------------------
# Cached model loader
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model_cached(kind: str, weights: str):
    return build_model(ModelSpec(kind=kind, weights=weights))


# ---------------------------
# Run
# ---------------------------
if run_btn:
    if uploaded is None:
        st.warning("Please upload an image.")
        st.stop()

    if not weights_path_str.strip():
        st.warning("Please provide weights path.")
        st.stop()

    weights_path = Path(weights_path_str).expanduser().resolve()
    if not weights_path.exists():
        st.error(f"Weights not found: {weights_path}")
        st.stop()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # write upload to disk so ultralytics can read it
        raw_img_path = tmpdir / uploaded.name
        raw_img_path.write_bytes(uploaded.getbuffer())

        # HEIC -> JPG if needed
        img_path = load_image_auto(raw_img_path)

        st.info(f"Running inference with **{model_kind}** ...")

        # model
        model = load_model_cached(model_kind, str(weights_path))

        # predict + timing (no saving)
        t0 = time.perf_counter()
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            save=False,
            verbose=False,
        )
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000.0

        r = results[0]

        # visualize
        st.subheader("📊 Prediction Result")
        vis_bgr = r.plot()          # numpy (BGR)
        vis_rgb = vis_bgr[..., ::-1]
        st.image(vis_rgb, caption="Prediction (in-app only)", use_column_width=True)

        # time metric
        st.markdown("### ⏱️ Inference Time")
        st.metric("Single-image inference", f"{infer_ms:.2f} ms")

        # table + structured dets
        dets = summarize_detections(r)

        if len(dets) == 0:
            st.info("No detections.")
        else:
            st.markdown("### 📦 Detected Objects")
            # simple table
            table = [
                {
                    "class": d["class"],
                    "confidence": round(d["conf"], 4),
                    "x1": round(d["xyxy"][0], 1),
                    "y1": round(d["xyxy"][1], 1),
                    "x2": round(d["xyxy"][2], 1),
                    "y2": round(d["xyxy"][3], 1),
                }
                for d in dets
            ]
            st.dataframe(table, use_container_width=True)

        # rule-based decision
        grade, rationale = decide_grade(dets, fail_threshold=fail_threshold)
        st.markdown("### ✅ Rule-based Decision")
        st.write(f"**{grade}** — {rationale}")

        # LLM report (optional)
        if use_llm:
            st.subheader("📝 Automated Inspection Report (LLM)")
            try:
                with st.spinner("Generating report..."):
                    report = generate_report_llm(
                        dets=dets,
                        grade=grade,
                        rationale=rationale,
                        model_kind=model_kind,
                        infer_ms=infer_ms,
                        lang=llm_lang,
                    )
                st.markdown(report)
            except Exception as e:
                st.error(f"LLM report failed: {e}")
                st.info("Check that you installed `openai` and exported OPENAI_API_KEY.")
