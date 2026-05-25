import argparse
from pathlib import Path

from ultralytics import YOLO


def count_label_files(data_root: Path):
    train_labels = list((data_root / 'labels' / 'train').glob('*.txt'))
    val_labels = list((data_root / 'labels' / 'val').glob('*.txt'))
    return len(train_labels), len(val_labels)


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for number plate detection.')
    parser.add_argument('--data', default='data.yaml', help='YOLO data.yaml path.')
    parser.add_argument('--model', default='yolov8n.pt', help='Starting model checkpoint.')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--name', default='train')
    parser.add_argument('--device', default='cpu', help='Use 0 for GPU, cpu for CPU.')
    return parser.parse_args()


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    dataset_root = project_dir / 'dataset'
    train_count, val_count = count_label_files(dataset_root)

    if train_count == 0 or val_count == 0:
        raise SystemExit(
            'YOLO label files are missing. Add .txt annotations under '\
            'dataset/labels/train and dataset/labels/val before training. '\
            f'Found train labels={train_count}, val labels={val_count}.'
        )

    model = YOLO(str(project_dir / args.model if not Path(args.model).is_absolute() else args.model))
    results = model.train(
        data=str(project_dir / args.data if not Path(args.data).is_absolute() else args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        cache=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.0005,
        close_mosaic=10,
        mosaic=0.1,
        mixup=0.0,
        copy_paste=0.0,
        degrees=3,
        translate=0.08,
        scale=0.2,
        shear=1,
        perspective=0.0,
        fliplr=0.5,
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.2,
        patience=20,
        device=args.device,
        name=args.name,
    )
    print('Training completed.')
    print(results)


if __name__ == '__main__':
    main()
