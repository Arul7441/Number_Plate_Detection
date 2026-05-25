import argparse
import csv
from pathlib import Path

import cv2

from number_plate_detector import (
    annotate_image,
    blur_score,
    detect_with_opencv,
    image_files,
    non_max_suppression,
    resolve_path,
)
from ocr_predict import read_plate_image


def parse_args():
    parser = argparse.ArgumentParser(description='Audit images and generate reviewable YOLO pseudo-labels.')
    parser.add_argument('--source', default='dataset/images', help='Image file or folder to scan.')
    parser.add_argument('--output', default='runs/dataset_audit', help='Folder for review images and CSV.')
    parser.add_argument('--labels-output', default='dataset/labels_auto', help='Folder for generated YOLO label txt files.')
    parser.add_argument('--max-candidates', type=int, default=3, help='Maximum candidate boxes per image.')
    parser.add_argument('--max-width', type=int, default=1400, help='Resize large images to this width for faster scanning.')
    parser.add_argument('--ocr', action='store_true', help='Run OCR on detected crops during audit. Slower but useful for spot checks.')
    parser.add_argument('--apply', action='store_true', help='Also copy pseudo-labels into dataset/labels/train and dataset/labels/val.')
    return parser.parse_args()


def yolo_line(box, width, height):
    x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f'0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}'


def split_name(image_path):
    parts = [part.lower() for part in image_path.parts]
    if 'train' in parts:
        return 'train'
    if 'val' in parts:
        return 'val'
    return 'unsplit'


def write_label(label_path, boxes, width, height):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [yolo_line(box, width, height) for box in boxes]
    label_path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def scan_image(image, max_width):
    height, width = image.shape[:2]
    if width <= max_width:
        return image, 1.0
    scale = max_width / float(width)
    resized = cv2.resize(image, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def scale_box_to_original(box, scale, width, height):
    if scale == 1.0:
        return box
    x1, y1, x2, y2 = box
    return (
        max(0, min(width, int(round(x1 / scale)))),
        max(0, min(height, int(round(y1 / scale)))),
        max(0, min(width, int(round(x2 / scale)))),
        max(0, min(height, int(round(y2 / scale)))),
    )


def main():
    args = parse_args()
    output_dir = resolve_path(args.output)
    review_dir = output_dir / 'review_images'
    crop_dir = output_dir / 'plate_crops'
    labels_root = resolve_path(args.labels_output)
    review_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)

    images = list(image_files(args.source))
    if not images:
        raise SystemExit(f'No images found in {args.source}')

    rows = []
    for index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        split = split_name(image_path)
        label_path = labels_root / split / f'{image_path.stem}.txt'
        review_path = review_dir / f'{split}_{image_path.stem}.jpg'

        if image is None:
            rows.append({
                'image': str(image_path),
                'split': split,
                'status': 'unreadable',
                'candidate_count': 0,
                'plate_text': '',
                'blur_score': 0,
                'review_image': '',
                'label_file': '',
            })
            continue

        height, width = image.shape[:2]
        if label_path.exists() and review_path.exists():
            label_lines = [line for line in label_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            rows.append({
                'image': str(image_path),
                'split': split,
                'status': 'existing',
                'candidate_count': len(label_lines),
                'plate_text': '',
                'blur_score': round(blur_score(image), 2),
                'review_image': str(review_path),
                'label_file': str(label_path),
            })
            continue

        scan, scale = scan_image(image, args.max_width)
        raw_candidates = detect_with_opencv(scan, padding=10)
        candidates = [
            (scale_box_to_original(box, scale, width, height), score)
            for box, score in raw_candidates
        ]
        candidates = non_max_suppression(candidates)[:args.max_candidates]
        accepted = []
        plate_texts = []

        for crop_index, (box, _score) in enumerate(candidates, start=1):
            x1, y1, x2, y2 = box
            crop = image[y1:y2, x1:x2]
            text, confidence = read_plate_image(crop) if args.ocr else ('', 0.0)
            if text or not args.ocr:
                accepted.append(box)
                if text:
                    plate_texts.append(f'{text} ({confidence:.2f})')
                cv2.imwrite(str(crop_dir / f'{image_path.stem}_{crop_index}.jpg'), crop)

        if not accepted and candidates:
            accepted = [candidates[0][0]]

        write_label(label_path, accepted, width, height)

        if args.apply and split in {'train', 'val'}:
            actual_label = resolve_path(Path('dataset') / 'labels' / split / f'{image_path.stem}.txt')
            write_label(actual_label, accepted, width, height)

        detections_for_image = []
        for box in accepted:
            from number_plate_detector import PlateDetection
            detections_for_image.append(PlateDetection(box, '', 0.0, 1.0, 'auto-label'))
        review_image = annotate_image(image, detections_for_image)
        cv2.imwrite(str(review_path), review_image)

        rows.append({
            'image': str(image_path),
            'split': split,
            'status': 'ok' if accepted else 'no_candidate',
            'candidate_count': len(accepted),
            'plate_text': '; '.join(plate_texts),
            'blur_score': round(blur_score(image), 2),
            'review_image': str(review_path),
            'label_file': str(label_path),
        })

        if index % 25 == 0:
            print(f'Processed {index}/{len(images)} images...')

    csv_path = output_dir / 'audit_report.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        fieldnames = ['image', 'split', 'status', 'candidate_count', 'plate_text', 'blur_score', 'review_image', 'label_file']
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_count = sum(1 for row in rows if row['status'] == 'ok')
    no_candidate_count = sum(1 for row in rows if row['status'] == 'no_candidate')
    print(f'Images scanned: {len(rows)}')
    print(f'Images with candidate labels: {ok_count}')
    print(f'Images without candidates: {no_candidate_count}')
    print(f'Review CSV: {csv_path}')
    print(f'Review images: {review_dir}')
    print(f'Pseudo-labels: {labels_root}')
    if not args.apply:
        print('Labels were not copied into dataset/labels. Review them first, then rerun with --apply if acceptable.')


if __name__ == '__main__':
    main()
