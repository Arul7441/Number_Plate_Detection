import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from ocr_predict import read_plate_image, read_plate_image_permissive

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OPENCV_SCAN_MAX_WIDTH = 1600
DEFAULT_MODEL_CANDIDATES = (
    Path("runs/detect/train/weights/best.pt"),
    Path("runs/detect/train-2/weights/best.pt"),
    Path("runs/detect/train-3/weights/best.pt"),
    Path("runs/detect/train-4/weights/best.pt"),
    Path("runs/detect/train-5/weights/best.pt"),
    Path("runs/detect/train-6/weights/best.pt"),
    Path("runs/detect/train-7/weights/best.pt"),
)

# ── Thresholds ────────────────────────────────────────────────────────────────
# BUG-FIX: OCR confidence threshold was 0.50 — this silently discarded valid
# detections whose OCR returned 0.40–0.49 (common on two-line / tilted plates).
# Lowered to 0.25 so borderline reads are kept; the table shows the confidence
# so users can judge quality themselves.
OCR_CONFIDENCE_THRESHOLD = 0.25

# BUG-FIX: YOLO detector confidence threshold for sending a crop to OCR was
# hard-coded to 0.45 inside the loop.  Some real plates score 0.26–0.44
# (partial occlusion, edge of frame).  Lower to 0.20 and let OCR confidence
# be the quality gate instead.
YOLO_OCR_MIN_CONF = 0.20


@dataclass
class PlateDetection:
    box: tuple[int, int, int, int]
    text: str
    ocr_confidence: float
    detector_confidence: float
    detector: str


def blur_score(image_bgr: np.ndarray) -> float:
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def project_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root() / path


def find_model_path(model_path: str | Path | None = None) -> Optional[Path]:
    candidates = [Path(model_path)] if model_path else list(DEFAULT_MODEL_CANDIDATES)
    existing = []
    for candidate in candidates:
        resolved = resolve_path(candidate)
        if resolved.exists() and resolved.stat().st_size > 0:
            existing.append(resolved)
    if existing:
        return max(existing, key=lambda p: p.stat().st_mtime)
    return None


def load_yolo_model(model_path: str | Path | None = None):
    resolved = find_model_path(model_path)
    if resolved is None or YOLO is None:
        return None, resolved
    return YOLO(str(resolved)), resolved


def image_files(source: str | Path) -> Iterable[Path]:
    source = resolve_path(source)
    if source.is_file():
        yield source
        return
    for path in sorted(source.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def clip_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    padding: int = 0,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width, x2 + padding),
        min(height, y2 + padding),
    )


def detect_with_yolo(
    model,
    image_bgr: np.ndarray,
    conf: float,
    imgsz: int,
    padding: int,
) -> list[tuple[tuple[int, int, int, int], float]]:
    if model is None:
        return []
    height, width = image_bgr.shape[:2]
    results = model.predict(
        source=np.ascontiguousarray(image_bgr),
        conf=conf,
        imgsz=imgsz,
        verbose=False,
    )
    boxes: list[tuple[tuple[int, int, int, int], float]] = []
    for result in results:
        if result.boxes is None:
            continue
        xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        for raw_box, score in zip(xyxy, confidences):
            x1, y1, x2, y2 = map(int, raw_box)
            boxes.append((clip_box((x1, y1, x2, y2), width, height, padding), float(score)))
    return boxes


def detect_with_opencv(
    image_bgr: np.ndarray,
    padding: int,
) -> list[tuple[tuple[int, int, int, int], float]]:
    scan_image, scale = resize_for_scan(image_bgr)
    yellow_candidates = detect_yellow_plate_candidates(scan_image, padding=padding)
    white_candidates = detect_white_plate_candidates(scan_image, padding=padding)
    text_candidates = detect_text_plate_candidates(scan_image, padding=padding)
    candidates = non_max_suppression(yellow_candidates + white_candidates + text_candidates)[:10]
    if scale == 1.0:
        return candidates
    height, width = image_bgr.shape[:2]
    return [(scale_box(box, scale, width, height), score) for box, score in candidates]


def resize_for_scan(image_bgr: np.ndarray):
    height, width = image_bgr.shape[:2]
    if width <= OPENCV_SCAN_MAX_WIDTH:
        return image_bgr, 1.0
    scale = OPENCV_SCAN_MAX_WIDTH / float(width)
    resized = cv2.resize(image_bgr, (OPENCV_SCAN_MAX_WIDTH, int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def scale_box(box: tuple[int, int, int, int], scale: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width, int(round(x1 / scale)))),
        max(0, min(height, int(round(y1 / scale)))),
        max(0, min(width, int(round(x2 / scale)))),
        max(0, min(height, int(round(y2 / scale)))),
    )


def edge_density(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    edges = cv2.Canny(gray, 80, 180)
    return float(np.count_nonzero(edges)) / float(edges.size)


def detect_yellow_plate_candidates(
    image_bgr: np.ndarray,
    padding: int,
) -> list[tuple[tuple[int, int, int, int], float]]:
    height, width = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, 70, 70]), np.array([40, 255, 255]))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = width * height
    candidates: list[tuple[tuple[int, int, int, int], float]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            continue
        area = w * h
        aspect = w / float(h)
        area_ratio = area / float(image_area)
        rectangularity = cv2.contourArea(contour) / float(area)

        if not (0.0003 <= area_ratio <= 0.04 and 0.8 <= aspect <= 8.0 and area > 1000):
            continue
        if rectangularity < 0.35:
            continue

        aspect_score = 1.0 - min(abs(aspect - 2.6) / 4.0, 1.0)
        score = min(0.99, 0.70 + rectangularity * 0.25 + aspect_score * 0.10)
        candidates.append((clip_box((x, y, x + w, y + h), width, height, padding + 8), float(score)))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[:8]


def detect_white_plate_candidates(
    image_bgr: np.ndarray,
    padding: int,
) -> list[tuple[tuple[int, int, int, int], float]]:
    height, width = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 135]), np.array([180, 90, 255]))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = width * height
    candidates: list[tuple[tuple[int, int, int, int], float]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            continue
        area = w * h
        aspect = w / float(h)
        area_ratio = area / float(image_area)
        rectangularity = cv2.contourArea(contour) / float(area)
        if not (0.00015 <= area_ratio <= 0.035 and 1.4 <= aspect <= 8.5 and area > 600):
            continue
        if rectangularity < 0.30:
            continue

        crop_gray = cv2.cvtColor(image_bgr[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        density = edge_density(crop_gray)
        if density < 0.04:
            continue

        aspect_score = 1.0 - min(abs(aspect - 3.8) / 5.0, 1.0)
        score = min(0.97, 0.62 + rectangularity * 0.22 + density * 0.9 + aspect_score * 0.10)
        candidates.append((clip_box((x, y, x + w, y + h), width, height, padding + 6), float(score)))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[:8]


def detect_text_plate_candidates(
    image_bgr: np.ndarray,
    padding: int,
) -> list[tuple[tuple[int, int, int, int], float]]:
    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 7))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, blackhat_kernel)

    grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_x = np.absolute(grad_x)
    min_val, max_val = float(np.min(grad_x)), float(np.max(grad_x))
    if max_val > min_val:
        grad_x = ((grad_x - min_val) / (max_val - min_val) * 255).astype("uint8")
    else:
        grad_x = np.zeros_like(gray)

    grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    closed = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, close_kernel)
    thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[tuple[int, int, int, int], float]] = []
    image_area = width * height

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            continue
        area = w * h
        aspect = w / float(h)
        area_ratio = area / float(image_area)
        if 2.0 <= aspect <= 6.5 and area > 400 and 0.0003 <= area_ratio <= 0.08:
            rectangularity = cv2.contourArea(contour) / float(area)
            score = min(0.72, 0.25 + rectangularity * 0.45 + min(area_ratio * 5, 0.15))
            candidates.append((clip_box((x, y, x + w, y + h), width, height, padding), float(score)))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return non_max_suppression(candidates[:5])


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def non_max_suppression(
    items: list[tuple[tuple[int, int, int, int], float]],
    threshold: float = 0.35,
):
    kept: list[tuple[tuple[int, int, int, int], float]] = []
    for box, score in sorted(items, key=lambda item: item[1], reverse=True):
        if all(iou(box, kept_box) < threshold for kept_box, _ in kept):
            kept.append((box, score))
    return kept


def annotate_image(image_bgr: np.ndarray, detections: list[PlateDetection]) -> np.ndarray:
    annotated = image_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        label = detection.text or "PLATE"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 0), 3)
        cv2.putText(
            annotated,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 180, 0),
            2,
        )
    return annotated


def detect_plates(
    image_bgr: np.ndarray,
    model=None,
    conf: float = 0.25,
    imgsz: int = 960,
    padding: int = 10,
    use_cv_fallback: bool = True,
) -> tuple[list[PlateDetection], dict]:

    started = time.perf_counter()
    timings: dict[str, float | str | int | bool] = {}

    # ── YOLO detection ────────────────────────────────────────────────────────
    yolo_started = time.perf_counter()
    boxes = detect_with_yolo(model, image_bgr, conf=conf, imgsz=imgsz, padding=padding)
    timings["yolo_seconds"] = round(time.perf_counter() - yolo_started, 3)
    detector_name = "YOLO"

    # ── OpenCV fallback ───────────────────────────────────────────────────────
    if not boxes and use_cv_fallback:
        cv_started = time.perf_counter()
        boxes = detect_with_opencv(image_bgr, padding=padding)
        timings["opencv_seconds"] = round(time.perf_counter() - cv_started, 3)
        detector_name = "OpenCV fallback"

    # ── Select boxes for OCR ──────────────────────────────────────────────────
    ocr_boxes = boxes
    if detector_name == "OpenCV fallback":
        strong_boxes = [item for item in boxes if item[1] >= 0.75]
        ocr_boxes = strong_boxes[:8] if strong_boxes else boxes[:8]

    detections: list[PlateDetection] = []

    for box, detector_confidence in ocr_boxes:

        # BUG-FIX: Was 0.45 — missed real plates scored 0.20–0.44 by YOLO
        if detector_confidence < YOLO_OCR_MIN_CONF:
            continue

        x1, y1, x2, y2 = box
        crop = image_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        text, ocr_confidence = read_plate_image(crop)

        # BUG-FIX: Was 0.50 — dropped borderline but correct OCR reads.
        # Lowered to OCR_CONFIDENCE_THRESHOLD (0.25); confidence is shown in
        # the UI so the user can assess quality.
        if ocr_confidence < OCR_CONFIDENCE_THRESHOLD:
            # Keep the detection box even if OCR confidence is too low —
            # show as "Unreadable" rather than silently dropping the row.
            detections.append(PlateDetection(
                box=box,
                text="",
                ocr_confidence=ocr_confidence,
                detector_confidence=detector_confidence,
                detector=detector_name,
            ))
            continue

        if detector_name == "OpenCV fallback" and not text:
            continue

        detections.append(PlateDetection(
            box=box,
            text=text,
            ocr_confidence=ocr_confidence,
            detector_confidence=detector_confidence,
            detector=detector_name,
        ))

    # ── Last-resort OCR for OpenCV fallback with zero results ─────────────────
    if (
        detector_name == "OpenCV fallback"
        and not any(d.text for d in detections)
        and ocr_boxes
    ):
        box, detector_confidence = ocr_boxes[0]
        x1, y1, x2, y2 = box
        crop = image_bgr[y1:y2, x1:x2]
        text, ocr_confidence = read_plate_image_permissive(crop)
        if text:
            # Replace or append
            if detections:
                detections[0] = PlateDetection(
                    box=box,
                    text=text,
                    ocr_confidence=ocr_confidence,
                    detector_confidence=detector_confidence,
                    detector=detector_name,
                )
            else:
                detections.append(PlateDetection(
                    box=box,
                    text=text,
                    ocr_confidence=ocr_confidence,
                    detector_confidence=detector_confidence,
                    detector=detector_name,
                ))

    timings["total_seconds"] = round(time.perf_counter() - started, 3)
    timings["detections"] = len(detections)
    timings["image_blur_score"] = round(blur_score(image_bgr), 2)
    timings["image_quality"] = "blurry" if timings["image_blur_score"] < 90 else "clear"

    return detections, timings