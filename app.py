from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from number_plate_detector import (
    annotate_image,
    detect_plates,
    find_model_path,
    load_yolo_model,
)

# ===================================================
# PAGE CONFIG
# ===================================================

st.set_page_config(
    page_title="GUVI's Final Number Plate Detection",
    layout="wide"
)

st.title("GUVI's Final Number Plate Detection System")

# ===================================================
# LOAD YOLO MODEL
# ===================================================

@st.cache_resource
def cached_model(model_path_text: str):

    selected = model_path_text.strip() or None

    return load_yolo_model(selected)

# ===================================================
# IMAGE DECODER
# ===================================================

def decode_uploaded_image(file_obj):

    # Reset pointer
    file_obj.seek(0)

    # Read bytes safely
    file_bytes = np.asarray(
        bytearray(file_obj.read()),
        dtype=np.uint8
    )

    # Decode image
    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "Invalid input image."
        )

    return image

# ===================================================
# TABLE GENERATOR
# ===================================================

def rows_from_detections(detections):

    rows = []

    for idx, detection in enumerate(detections, start=1):

        x1, y1, x2, y2 = detection.box

        rows.append({
            "No.": idx,
            "Plate Text": detection.text or "Unreadable",
            "OCR Confidence": detection.ocr_confidence,
            "Detector Confidence": round(detection.detector_confidence, 3),
            "Detector": detection.detector,
            "Box": f"{x1},{y1},{x2},{y2}",
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    return rows

# ===================================================
# PREPROCESSING OUTPUTS
# ===================================================

def preprocessing_images(image_bgr, detections):

    if not detections:
        return []

    x1, y1, x2, y2 = detections[0].box

    crop = image_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        return []

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)

    thresh = cv2.adaptiveThreshold(
        bilateral,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return [
        ("Plate Crop", cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
        ("Gray", gray),
        ("Bilateral Filter", bilateral),
        ("Threshold", thresh),
    ]

# ===================================================
# SIDEBAR SETTINGS
# ===================================================

with st.sidebar:

    st.header("Settings")

    # AUTO FIND MODEL
    auto_model = find_model_path()

    # FORCE DEFAULT MODEL
    if auto_model:
        default_model = str(auto_model)
    else:
        default_model = "runs/detect/train-7/weights/best.pt"

    st.write("Detected model path:")

    st.code(default_model)

    model_path_text = st.text_input(
        "YOLO model path",
        value=default_model
    )

    conf = st.slider(
        "Detection confidence",
        min_value=0.05,
        max_value=0.90,
        value=0.25,
        step=0.05
    )

    imgsz = st.select_slider(
        "Inference image size",
        options=[320, 480, 640],
        value=480
    )

    padding = st.slider(
        "Crop padding",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )

    use_cv_fallback = st.toggle(
        "Use OpenCV fallback",
        value=False
    )

# ===================================================
# MODEL INITIALIZATION
# ===================================================

model, resolved_model_path = cached_model(model_path_text)

if resolved_model_path:

    st.sidebar.success(
        f"YOLO model loaded:\n{resolved_model_path.name}"
    )

else:

    st.sidebar.error(
        "No trained YOLO weights found."
    )

# ===================================================
# INPUT SOURCE
# ===================================================

source = st.radio(
    "Input source",
    ["Upload image", "Camera"],
    horizontal=True
)

input_file = None

if source == "Upload image":

    input_file = st.file_uploader(
        "Upload vehicle image",
        type=["jpg", "jpeg", "png"]
    )

else:

    input_file = st.camera_input(
        "Capture vehicle image"
    )

# ===================================================
# MAIN DETECTION
# ===================================================

if input_file is not None:

    try:

        image_bgr = decode_uploaded_image(input_file)

        with st.spinner("Detecting number plate..."):

            detections, metrics = detect_plates(
                image_bgr,
                model=model,
                conf=conf,
                imgsz=imgsz,
                padding=padding,
                use_cv_fallback=use_cv_fallback,
            )

        # -------------------------------------------

        annotated = annotate_image(
            image_bgr,
            detections
        )

        annotated_rgb = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )

        # -------------------------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Detected Plates",
            len(detections)
        )

        col2.metric(
            "Processing Time",
            f"{metrics.get('total_seconds', 0):.2f} sec"
        )

        col3.metric(
            "Detector",
            detections[0].detector if detections else "None"
        )

        # -------------------------------------------

        st.image(
            annotated_rgb,
            caption="Detection Result",
            use_container_width=True
        )

        # -------------------------------------------

        if detections:

            st.subheader("Preprocessing Output")

            prep_columns = st.columns(4)

            prep_images = preprocessing_images(
                image_bgr,
                detections
            )

            for column, (caption, prep_image) in zip(prep_columns, prep_images):

                column.image(
                    prep_image,
                    caption=caption,
                    use_container_width=True
                )

        # -------------------------------------------

        rows = rows_from_detections(detections)

        if rows:

            st.success("Number plate detected successfully.")

            table = pd.DataFrame(rows)

            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True
            )

            csv_data = table.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download CSV",
                csv_data,
                "number_plate_results.csv",
                "text/csv"
            )

        else:

            st.warning(
                "No number plate detected."
            )

        # -------------------------------------------

        with st.expander("Performance Details"):

            st.json(metrics)

    except Exception as exc:

        st.error(str(exc))

# ===================================================
# EMPTY SCREEN
# ===================================================

else:

    st.info(
        "Upload a vehicle image to start detection."
    )