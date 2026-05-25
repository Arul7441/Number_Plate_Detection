import argparse
import csv
from pathlib import Path

import cv2

from number_plate_detector import annotate_image, detect_plates, image_files, load_yolo_model, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(description='Run number plate detection on an image or folder.')
    parser.add_argument('--source', default='dataset/images/val', help='Image file or folder to process.')
    parser.add_argument('--model', default='', help='Path to YOLO best.pt. Auto-detected when omitted.')
    parser.add_argument('--output', default='runs/predict_cli', help='Folder for annotated images and CSV log.')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO confidence threshold.')
    parser.add_argument('--imgsz', type=int, default=960, help='YOLO inference image size.')
    parser.add_argument('--no-fallback', action='store_true', help='Disable OpenCV fallback detection.')
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, model_path = load_yolo_model(args.model or None)
    if model_path:
        print(f'Using YOLO weights: {model_path}')
    elif args.no_fallback:
        print('No YOLO weights found and fallback is disabled.')
    else:
        print('No YOLO weights found. Using OpenCV fallback detector.')

    rows = []
    files = list(image_files(args.source))
    if not files:
        raise SystemExit(f'No images found in {args.source}')

    for image_path in files:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f'Skipping unreadable image: {image_path}')
            continue

        detections, metrics = detect_plates(
            image,
            model=model,
            conf=args.conf,
            imgsz=args.imgsz,
            use_cv_fallback=not args.no_fallback,
        )
        annotated = annotate_image(image, detections)
        result_path = output_dir / f'{image_path.stem}_result.jpg'
        cv2.imwrite(str(result_path), annotated)

        for idx, detection in enumerate(detections, start=1):
            rows.append({
                'image': str(image_path),
                'result_image': str(result_path),
                'plate_index': idx,
                'plate_text': detection.text,
                'ocr_confidence': detection.ocr_confidence,
                'detector_confidence': round(detection.detector_confidence, 3),
                'detector': detection.detector,
                'box': ','.join(map(str, detection.box)),
                'processing_seconds': metrics.get('total_seconds', 0),
            })
        print(f'{image_path.name}: {len(detections)} plate(s)')

    log_path = output_dir / 'detections.csv'
    with log_path.open('w', newline='', encoding='utf-8') as handle:
        fieldnames = ['image', 'result_image', 'plate_index', 'plate_text', 'ocr_confidence', 'detector_confidence', 'detector', 'box', 'processing_seconds']
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Prediction completed. Results saved to {output_dir}')
    print(f'CSV log: {log_path}')


if __name__ == '__main__':
    main()
