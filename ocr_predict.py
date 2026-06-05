import re
from functools import lru_cache
from pathlib import Path

import cv2
import easyocr
import numpy as np

ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
MIN_TEXT_LENGTH = 4
BLUR_THRESHOLD = 90.0
INDIAN_STATE_CODES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CG', 'DD', 'DL', 'DN', 'GA', 'GJ',
    'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP',
    'MZ', 'NL', 'OD', 'OR', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK',
    'UP', 'WB',
}

# ── ZONE-AWARE conversions ────────────────────────────────────────────────────
# These are applied ONLY in specific plate zones, NOT globally.
# Global substitution was the root cause of BL→81, AS→A5, BN→8N errors.

# Used ONLY in the RTO-number zone (digits expected):
#   Common OCR confusions where a letter looks like a digit
LETTER_TO_DIGIT = {
    'O': '0',
    'Q': '0',
    'I': '1',
    # 'L', 'B', 'S', 'G', 'Z' intentionally removed:
    # L is a valid series letter (BL, KL…), B is a valid series letter (BN, BR…)
    # S is a valid series letter (AS, TS…), applying digit-substitution destroys them.
}

# Used ONLY in the state/series zones (letters expected):
DIGIT_TO_LETTER = {
    '0': 'O',
    '1': 'I',
    # '2','5','6','8' removed — these rarely appear in state/series and
    # substituting them introduces wrong letters (e.g. 8→B corrupts '08' RTO)
}

# ── Size guards ───────────────────────────────────────────────────────────────
MIN_CROP_PX = 20
MIN_DETECTOR_CONF = 0.15


@lru_cache(maxsize=1)
def get_reader():
    return easyocr.Reader(['en'], gpu=True, verbose=False)


def preprocess(img):
    if img is None or img.size == 0:
        raise ValueError('Cannot OCR an empty image crop.')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.bilateralFilter(gray, 9, 17, 17)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        19,
        11,
    )
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh


def blur_score(img):
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def unsharp_mask(gray, amount=1.6):
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.2)
    return cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)


def yellow_plate_text_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([10, 40, 60]), np.array([45, 255, 255]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    yellow_only = cv2.bitwise_and(gray, gray, mask=mask)
    yellow_only = cv2.resize(yellow_only, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
    yellow_only = cv2.normalize(yellow_only, None, 0, 255, cv2.NORM_MINMAX)
    yellow_only = cv2.equalizeHist(yellow_only)
    thresh = cv2.threshold(yellow_only, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    return thresh


def preprocess_variants(img):
    """Return up to 5 preprocessed variants of the plate crop."""
    base = preprocess(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 3 if max(gray.shape[:2]) < 180 else 2
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    sharp = unsharp_mask(clahe, amount=1.8 if blur_score(img) < BLUR_THRESHOLD else 1.2)

    variants = [
        base,
        sharp,
        cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1],
    ]

    if len(img.shape) == 3:
        variants.append(yellow_plate_text_mask(img))

        h, w = img.shape[:2]
        if w > 0 and h > 0:
            angle = _estimate_skew(gray)
            if abs(angle) > 1.0:
                M = cv2.getRotationMatrix2D((w * scale // 2, h * scale // 2), angle, 1.0)
                deskewed = cv2.warpAffine(gray, M, (w * scale, h * scale),
                                          flags=cv2.INTER_LANCZOS4,
                                          borderMode=cv2.BORDER_REPLICATE)
                variants.append(deskewed)

    return variants[:5]


def _estimate_skew(gray_upscaled: np.ndarray) -> float:
    try:
        edges = cv2.Canny(gray_upscaled, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                minLineLength=gray_upscaled.shape[1] // 4,
                                maxLineGap=20)
        if lines is None or len(lines) == 0:
            return 0.0
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -45 < angle < 45:
                angles.append(angle)
        if not angles:
            return 0.0
        return float(np.median(angles))
    except Exception:
        return 0.0


def clean_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return re.sub(r'^IND', '', text)


def as_letters(text):
    """Convert digit-lookalikes to letters — used only in state/series zones."""
    return ''.join(DIGIT_TO_LETTER.get(char, char) for char in text)


def as_digits(text):
    """Convert letter-lookalikes to digits — used only in RTO/number zones."""
    return ''.join(LETTER_TO_DIGIT.get(char, char) for char in text)


def score_plate_text(text, confidence):
    if not text:
        return 0.0
    compact = clean_text(text)
    length_score = min(len(compact) / 10.0, 1.0)
    has_letter = any(c.isalpha() for c in compact)
    has_digit = any(c.isdigit() for c in compact)
    mix_bonus = 0.2 if has_letter and has_digit else 0.0
    return float(confidence) + length_score + mix_bonus


def format_standard_plate(state, rto, series, number):
    return f'{state} {rto} {series} {number}'


def normalize_indian_plate(text):
    """
    Parse raw OCR text into a formatted Indian number plate string.
    Returns (formatted_plate, score) or (None, 0.0).

    Zone-aware parsing:
    - state (2 chars)  → letters only  → as_letters() to fix 0→O, 1→I
    - rto   (1-2 chars)→ digits only   → as_digits()  to fix O→0, I→1
    - series(1-3 chars)→ letters only  → as_letters() — NO digit substitution
                                          for L, B, S, G because they are
                                          valid series letters
    - number(4 chars)  → digits only   → as_digits()  to fix O→0, I→1

    KEY FIX: series uses as_letters() but LETTER_TO_DIGIT no longer contains
    L, B, S, G — so 'BL', 'AS', 'BN', 'CX' are preserved correctly.
    """
    cleaned = clean_text(text)
    if len(cleaned) < 6:
        return None, 0.0

    candidates = []

    # ── Bharat series: YY BH #### XX  (e.g. 22 BH 1234 AA)
    for start in range(0, max(1, len(cleaned) - 9)):
        chunk = cleaned[start:start + 10]
        if len(chunk) < 10:
            continue
        year = as_digits(chunk[:2])
        bh = as_letters(chunk[2:4])
        number = as_digits(chunk[4:8])
        suffix = as_letters(chunk[8:10])
        if year.isdigit() and bh == 'BH' and number.isdigit() and suffix.isalpha():
            candidates.append((f'{year} BH {number} {suffix}', 4.0))

    # ── Standard format: state(2L) + RTO(1-2D) + series(1-3L) + number(4D)
    for start in range(0, len(cleaned) - 5):
        remaining = cleaned[start:]
        for rto_len in (2, 1):
            for series_len in (3, 2, 1):
                total_len = 2 + rto_len + series_len + 4
                if len(remaining) < total_len:
                    continue

                chunk = remaining[:total_len]

                # Zone-aware conversions
                state_raw  = chunk[:2]
                rto_raw    = chunk[2:2 + rto_len]
                series_raw = chunk[2 + rto_len:2 + rto_len + series_len]
                number_raw = chunk[-4:]

                state  = as_letters(state_raw)   # fix 0→O, 1→I in state
                rto    = as_digits(rto_raw)       # fix O→0, I→1 in RTO
                series = as_letters(series_raw)   # fix 0→O, 1→I in series ONLY
                number = as_digits(number_raw)    # fix O→0, I→1 in number

                # Validate zones
                if not (state.isalpha() and rto.isdigit()
                        and series.isalpha() and number.isdigit()):
                    continue

                score = 2.0

                if state in INDIAN_STATE_CODES:
                    score += 1.0

                if len(rto) == 2:
                    score += 0.3

                if len(series) in (1, 2):
                    score += 0.2

                if total_len == len(remaining):
                    score += 0.5

                if len(rto) == 1:
                    score -= 0.1

                if len(rto) == 1 and len(series) == 1:
                    score -= 0.2

                candidates.append((format_standard_plate(state, rto, series, number), score))

    if not candidates:
        return None, 0.0

    return max(candidates, key=lambda item: item[1])


def ocr_text_candidates(results):
    """
    Build all candidate text strings from EasyOCR result tokens.
    Produces row-aware joins so two-line plates (KL-01 / BL 6750)
    are combined in reading order.
    """
    candidates = []
    tokens = []

    for bbox, text, confidence in results:
        cleaned = clean_text(text)
        if len(cleaned) >= MIN_TEXT_LENGTH:
            candidates.append((cleaned, float(confidence)))
        if cleaned:
            x_center = sum(point[0] for point in bbox) / len(bbox)
            y_center = sum(point[1] for point in bbox) / len(bbox)
            tokens.append((x_center, y_center, cleaned, float(confidence)))

    if tokens:
        line_sorted = sorted(tokens, key=lambda t: (t[1], t[0]))

        if len(tokens) > 1:
            ys = [t[1] for t in tokens]
            y_range = max(ys) - min(ys) if len(ys) > 1 else 1
            row_tol = max(y_range * 0.25, 5)
            rows: list[list] = []
            for token in sorted(tokens, key=lambda t: t[1]):
                placed = False
                for row in rows:
                    if abs(token[1] - row[0][1]) <= row_tol:
                        row.append(token)
                        placed = True
                        break
                if not placed:
                    rows.append([token])
            row_texts = []
            row_confs = []
            for row in rows:
                row_sorted = sorted(row, key=lambda t: t[0])
                row_texts.append(''.join(t[2] for t in row_sorted))
                row_confs.append(sum(t[3] for t in row_sorted) / len(row_sorted))
            combined_row = ''.join(row_texts)
            avg_conf_row = sum(row_confs) / len(row_confs)
            candidates.append((combined_row, avg_conf_row))

        for ordered in (
            sorted(tokens, key=lambda t: t[0]),
            line_sorted,
        ):
            combined = ''.join(t[2] for t in ordered)
            avg_conf = sum(t[3] for t in ordered) / len(ordered)
            candidates.append((combined, avg_conf))

    return candidates


def read_plate_image(img):
    """
    Main entry: read a plate crop and return (text, confidence).
    Returns ('', 0.0) if nothing reliable is found.
    """
    if img is None or img.size == 0:
        return '', 0.0

    h, w = img.shape[:2]
    if h < MIN_CROP_PX or w < MIN_CROP_PX:
        return '', 0.0

    try:
        variants = preprocess_variants(img)
    except ValueError:
        return '', 0.0

    candidates = []
    for processed in variants:
        results = get_reader().readtext(
            processed,
            detail=1,
            paragraph=False,
            allowlist=ALLOWLIST,
            decoder='beamsearch',
        )
        for raw_text, confidence in ocr_text_candidates(results):
            normalized, plate_score = normalize_indian_plate(raw_text)
            if normalized:
                candidates.append((
                    normalized,
                    float(confidence),
                    score_plate_text(normalized, confidence) + plate_score,
                ))

    if not candidates:
        # Relaxed fallback: best alphanumeric token even if not a valid Indian plate
        relaxed = []
        for processed in variants:
            results = get_reader().readtext(
                processed,
                detail=1,
                paragraph=False,
                allowlist=ALLOWLIST,
            )
            for raw_text, confidence in ocr_text_candidates(results):
                cleaned = clean_text(raw_text)
                if (len(cleaned) >= 5
                        and any(c.isalpha() for c in cleaned)
                        and any(c.isdigit() for c in cleaned)):
                    relaxed.append((cleaned, float(confidence),
                                    score_plate_text(cleaned, confidence)))
        if not relaxed:
            return '', 0.0
        text, conf, _score = max(relaxed, key=lambda item: item[2])
        return text, round(conf, 3)

    best_text, best_confidence, _score = max(candidates, key=lambda item: item[2])
    return best_text, round(best_confidence, 3)


def read_plate(path):
    img = cv2.imread(str(Path(path)))
    text, _confidence = read_plate_image(img)
    return text or None


def read_plate_image_permissive(img):
    if img is None or img.size == 0:
        return '', 0.0
    try:
        variants = preprocess_variants(img)
    except ValueError:
        return '', 0.0
    candidates = []
    for processed in variants:
        results = get_reader().readtext(
            processed,
            detail=1,
            paragraph=False,
            allowlist=ALLOWLIST,
        )
        for raw_text, confidence in ocr_text_candidates(results):
            cleaned = clean_text(raw_text)
            if len(cleaned) >= 4:
                candidates.append((cleaned, float(confidence),
                                   score_plate_text(cleaned, confidence)))
    if not candidates:
        return '', 0.0
    text, conf, _score = max(candidates, key=lambda item: item[2])
    return text, round(conf, 3)