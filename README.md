# GUVI's Final Number Plate Detection System

This project detects Indian vehicle number plates from images, applies OCR, and shows results in a Streamlit web app. The main project flow is simple: train YOLO with labeled images, then run the app.

## Project structure

- `app.py` - Streamlit application with image upload, camera capture, results table, metrics, and CSV/image downloads.
- `number_plate_detector.py` - Shared detection pipeline for YOLO, OpenCV fallback, OCR, and annotation.
- `ocr_predict.py` - EasyOCR preprocessing and text cleanup.
- `train.py` - YOLOv8 training script with dataset validation before training.
- `predict.py` - Command-line batch prediction script.
- `project_check.py` - Quick project health check for images, labels, and trained weights.
- `data.yaml` - YOLO dataset configuration.
- `requirements.txt` - Python dependencies.

## Setup

```powershell
cd D:\NumberPlateProject
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Check project assets

```powershell
python project_check.py
```

Training requires YOLO annotation `.txt` files under:
- `dataset/labels/train`
- `dataset/labels/val`

## Train YOLO

After adding YOLO-format labels, run:

```powershell
python train.py --device cpu --epochs 60 --imgsz 960 --batch 4
```

For GPU training, use:

```powershell
python train.py --device 0
```

The expected model output:

```text
runs/detect/train/weights/best.pt
```

## Run the Streamlit app

```powershell
streamlit run app.py
```

For project-quality output, use trained YOLO weights at `runs/detect/train/weights/best.pt`.

## Batch prediction

```powershell
python predict.py --source dataset/images/val --output runs/predict_cli
```

This saves annotated images and `detections.csv`.

## Simple Project Flow

1. Prepare image labels in YOLO format (`dataset/labels/train`, `dataset/labels/val`).
2. Train model with `train.py`.
3. Confirm `best.pt` exists.
4. Run `streamlit run app.py`.
5. Upload image and read detected number plate text.
