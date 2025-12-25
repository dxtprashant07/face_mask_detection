import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide"
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    "<h1>😷 Face Mask Detection System</h1>"
    "<p style='color:#6b7280;'>YOLO-based Surveillance Detection</p>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# Load Model (cached)
# --------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best_face_mask.onnx")

model = load_model()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("⚙️ Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.5, 0.05
)

mode = st.sidebar.radio(
    "Detection Mode",
    ["📤 Upload Image", "🎥 Upload Video"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "⚠️ Webcam is **NOT supported** on Streamlit Cloud.\n\n"
    "Use local OpenCV script for live camera."
)

# --------------------------------------------------
# IMAGE MODE
# --------------------------------------------------
if mode == "📤 Upload Image":
    st.subheader("Upload an Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", use_column_width=True)

        if st.button("Run Detection"):
            with st.spinner("Running inference..."):
                results = model(img_np, conf=conf_threshold)[0]
                annotated = results.plot()
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            with col2:
                st.image(annotated, caption="Detected Output", use_column_width=True)

            st.markdown("### Detection Summary")
            if len(results.boxes) == 0:
                st.info("No faces detected.")
            else:
                for box in results.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls]

                    if label == "without_mask":
                        st.error(f"🚨 {label} ({conf:.2f})")
                    elif label == "mask_weared_incorrect":
                        st.warning(f"⚠️ {label} ({conf:.2f})")
                    else:
                        st.success(f"✅ {label} ({conf:.2f})")

# --------------------------------------------------
# VIDEO MODE
# --------------------------------------------------
elif mode == "🎥 Upload Video":
    st.subheader("Upload a Video")

    video_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov"]
    )

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        with st.spinner("Processing video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, conf=conf_threshold)[0]
                annotated = results.plot()
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                stframe.image(annotated)

        cap.release()
        os.remove(tfile.name)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6b7280;'>"
    "Academic + Real-World Surveillance Pipeline</p>",
    unsafe_allow_html=True
)
