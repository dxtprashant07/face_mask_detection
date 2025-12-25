import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS Styling
# --------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
h1 {
    color: #1f2937;
}
h3 {
    color: #374151;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    padding: 0.5em 1.2em;
}
.stButton>button:hover {
    background-color: #1d4ed8;
}
.alert-box {
    padding: 10px;
    border-radius: 8px;
    background-color: #fee2e2;
    color: #991b1b;
    font-weight: bold;
}
.success-box {
    padding: 10px;
    border-radius: 8px;
    background-color: #dcfce7;
    color: #166534;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    "<h1>😷 Face Mask Detection System</h1>"
    "<p style='color:#6b7280;'>Real-time & Image-based Surveillance using YOLO</p>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best_face_mask.onnx")

model = load_model()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("⚙️ Controls")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.6, 0.05
)

mode = st.sidebar.radio(
    "Detection Mode",
    ["📤 Upload Image", "📷 Live Webcam"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Classes**")
st.sidebar.markdown("- 😷 With Mask")
st.sidebar.markdown("- 🚫 Without Mask")
st.sidebar.markdown("- ⚠️ Mask Worn Incorrectly")

# --------------------------------------------------
# IMAGE UPLOAD MODE
# --------------------------------------------------
if mode == "📤 Upload Image":
    st.subheader("Upload an Image")

    uploaded_file = st.file_uploader(
        "Choose an image (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", use_column_width=True)

        if st.button("🔍 Run Detection"):
            with st.spinner("Analyzing image..."):
                results = model(img_np, conf=conf_threshold)[0]
                annotated = results.plot()
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            with col2:
                st.image(annotated, caption="Detection Result", use_column_width=True)

            st.markdown("### 🧾 Detection Summary")
            if len(results.boxes) == 0:
                st.info("No faces detected.")
            else:
                for box in results.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls]

                    if label == "without_mask" and conf >= 0.7:
                        st.markdown(
                            f"<div class='alert-box'>🚨 {label.upper()} ({conf:.2f})</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div class='success-box'>✅ {label} ({conf:.2f})</div>",
                            unsafe_allow_html=True
                        )

# --------------------------------------------------
# LIVE WEBCAM MODE
# --------------------------------------------------
elif mode == "📷 Live Webcam":
    st.subheader("Live Camera Detection")

    start = st.checkbox("▶️ Start Camera")

    FRAME_WINDOW = st.image([])
    cap = None

    if start:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Unable to access webcam")
        else:
            while start:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, conf=conf_threshold)[0]
                annotated = results.plot()
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                FRAME_WINDOW.image(annotated)

    if cap:
        cap.release()

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6b7280;'>"
    "Built for real-world surveillance & academic evaluation</p>",
    unsafe_allow_html=True
)
