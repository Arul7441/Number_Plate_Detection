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
LETTER_TO_DIGIT = {
    'O': '0',
    'Q': '0',
    'D': '0',
    'I': '1',
    'L': '1',
    'T': '1',
    'Z': '2',
    'S': '5',
    'B': '8',
    'G': '6',
}
DIGIT_TO_LETTER = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '7': 'T',
    '8': 'B',
    '9': 'G',
}


@lru_cache(maxsize=1)
def get_reader():
    return easyocr.Reader(['en'], gpu=False, verbose=False)


def preprocess(img):

    if img is None or img.size == 0:
        raise ValueError('Cannot OCR an empty image crop.')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # upscale image
    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # remove noise
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # increase contrast
    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    # adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15
    )

    # morphology cleanup
    kernel = np.ones((2, 2), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel
    )

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
    mask = cv2.inRange(hsv, np.array([12, 50, 50]), np.array([45, 255, 255]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    yellow_only = cv2.bitwise_and(gray, gray, mask=mask)
    yellow_only = cv2.resize(yellow_only, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    yellow_only = cv2.normalize(yellow_only, None, 0, 255, cv2.NORM_MINMAX)
    yellow_only = cv2.equalizeHist(yellow_only)
    return cv2.threshold(yellow_only, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


def preprocess_variants(img):
    base = preprocess(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 3 if max(gray.shape[:2]) < 180 else 2
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    sharp = unsharp_mask(clahe, amount=1.8 if blur_score(img) < BLUR_THRESHOLD else 1.2)
    variants = [
        base,
        sharp,
        cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1],
    ]
    if len(img.shape) == 3:
        variants.append(yellow_plate_text_mask(img))
    return variants[:4]


def clean_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return re.sub(r'^IND', '', text)


def as_letters(text):
    return ''.join(DIGIT_TO_LETTER.get(char, char) for char in text)


def as_digits(text):
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
    cleaned = clean_text(text)
    if len(cleaned) < 8:
        return None, 0.0

    candidates = []

    # Bharat series: YY BH #### XX, for example 22 BH 1234 AA.
    for start in range(0, max(1, len(cleaned) - 9)):
        chunk = cleaned[start:start + 10]
        if len(chunk) < 10:
            continue
        year = as_digits(chunk[:2])
        bh = as_letters(chunk[2:4])
        number = as_digits(chunk[4:8])
        suffix = as_letters(chunk[8:10])
        if year.isdigit() and bh == 'BH' and number.isdigit() and suffix.isalpha():
            candidates.append((f'{year} BH {number} {suffix}', 3.0))

    # Standard format: state(2 letters) + RTO(1-2 digits) + series(1-3 letters) + number(4 digits).
    for start in range(0, len(cleaned) - 7):
        remaining = cleaned[start:]
        for rto_len in (2, 1):
            for series_len in (3, 2, 1):
                total_len = 2 + rto_len + series_len + 4
                if len(remaining) < total_len:
                    continue

                chunk = remaining[:total_len]
                state = as_letters(chunk[:2])
                rto = as_digits(chunk[2:2 + rto_len])
                series_start = 2 + rto_len
                series = as_letters(chunk[series_start:series_start + series_len])
                number = as_digits(chunk[-4:])

                if not (state.isalpha() and rto.isdigit() and series.isalpha() and number.isdigit()):
                    continue

                score = 2.0
                if state in INDIAN_STATE_CODES:
                    score += 1.0
                if len(rto) == 2:
                    score += 0.2
                if len(series) in (1, 2):
                    score += 0.1

                candidates.append((format_standard_plate(state, rto, series, number), score))

    if not candidates:
        return None, 0.0

    return max(candidates, key=lambda item: item[1])


def ocr_text_candidates(results):
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
        x_sorted = sorted(tokens, key=lambda item: item[0])
        line_sorted = sorted(tokens, key=lambda item: (item[1], item[0]))

        for ordered in (x_sorted, line_sorted):
            combined = ''.join(token for _x, _y, token, _confidence in ordered)
            confidence = sum(confidence for _x, _y, _token, confidence in ordered) / len(ordered)
            candidates.append((combined, confidence))

    return candidates


def read_plate_image(img):
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
        # Fast relaxed fallback: return best alphanumeric OCR token if strict Indian parsing fails.
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
                if len(cleaned) >= 5 and any(c.isalpha() for c in cleaned) and any(c.isdigit() for c in cleaned):
                    relaxed.append((cleaned, float(confidence), score_plate_text(cleaned, confidence)))
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
                candidates.append((cleaned, float(confidence), score_plate_text(cleaned, confidence)))
    if not candidates:
        return '', 0.0
    text, conf, _score = max(candidates, key=lambda item: item[2])
    return text, round(conf, 3)
