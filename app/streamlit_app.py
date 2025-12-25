import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Face Mask Detection System",
    layout="wide"
)

st.title("😷 Face Mask Detection — Real-World Deployment")
st.markdown("YOLO-based multi-person face mask detection")

# --------------------------------------------------
# Load Model (cached)
# --------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best_face_mask.onnx")   # update path if needed

model = load_model()

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.6,
    step=0.05
)

mode = st.sidebar.radio(
    "Select Mode",
    ["Upload Image", "Live Webcam"]
)

# --------------------------------------------------
# IMAGE UPLOAD MODE
# --------------------------------------------------
if mode == "Upload Image":
    st.subheader("📤 Upload an Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Detection"):
            with st.spinner("Running inference..."):
                results = model(img_np, conf=conf_threshold)[0]
                annotated = results.plot()

                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                st.image(
                    annotated,
                    caption="Detection Result",
                    use_column_width=True
                )

                # Optional: show detection summary
                st.markdown("### Detection Summary")
                for box in results.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls]
                    st.write(f"- **{label}** ({conf:.2f})")

# --------------------------------------------------
# LIVE WEBCAM MODE
# --------------------------------------------------
elif mode == "Live Webcam":
    st.subheader("📷 Live Webcam Detection")

    start_cam = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])
    cap = None

    if start_cam:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Unable to access webcam")
        else:
            while start_cam:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, conf=conf_threshold)[0]
                annotated = results.plot()

                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(annotated)

    if cap:
        cap.release()
    cv2.destroyAllWindows()