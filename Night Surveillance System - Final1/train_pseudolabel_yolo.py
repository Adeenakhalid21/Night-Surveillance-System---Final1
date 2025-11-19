#!/usr/bin/env python3
"""
Train YOLOv8 with pseudo-labels generated from a pre-trained model.

- Reads images from a dataset registered in SQLite (by dataset_name)
- Generates YOLO-format labels using a base model's predictions
- Prepares a train/val split and dataset YAML
- Fine-tunes YOLO for a few epochs

Usage (example):
  python train_pseudolabel_yolo.py \
    --dataset "COCO Train2017 Sample" \
    --weights yolov8n.pt \
    --epochs 5 --batch 16 --imgsz 640 --conf 0.5
"""
from __future__ import annotations
import os
import sys
import argparse
import random
import shutil
import sqlite3
from pathlib import Path
from typing import List

from ultralytics import YOLO

DB_PATH = 'night_surveillance.db'
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_image_paths(dataset_name: str) -> List[str]:
    conn = get_db_connection()
    row = conn.execute('SELECT dataset_id FROM datasets WHERE dataset_name=?', (dataset_name,)).fetchone()
    if not row:
        conn.close()
        raise SystemExit(f"Dataset not found: {dataset_name}")
    dataset_id = row['dataset_id']
    rows = conn.execute('SELECT image_path FROM training_data WHERE dataset_id=?', (dataset_id,)).fetchall()
    conn.close()
    imgs = [r['image_path'] for r in rows if r['image_path'] and os.path.exists(r['image_path'])]
    if not imgs:
        raise SystemExit("No images found or paths are invalid for the selected dataset.")
    return imgs


def prepare_dirs(base_dir: Path) -> dict:
    # Structure: base_dir/{images,labels}/{train,val}
    images_train = base_dir / 'images' / 'train'
    images_val = base_dir / 'images' / 'val'
    labels_train = base_dir / 'labels' / 'train'
    labels_val = base_dir / 'labels' / 'val'
    for p in [images_train, images_val, labels_train, labels_val]:
        p.mkdir(parents=True, exist_ok=True)
    return {
        'images_train': images_train,
        'images_val': images_val,
        'labels_train': labels_train,
        'labels_val': labels_val,
    }


def save_yolo_label(label_path: Path, xywhn, cls) -> None:
    with open(label_path, 'w', encoding='utf-8') as f:
        for i in range(xywhn.shape[0]):
            c = int(cls[i])
            x, y, w, h = xywhn[i].tolist()
            f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='COCO Train2017 Sample', help='Dataset name in SQLite')
    ap.add_argument('--weights', type=str, default='yolov8n.pt', help='Base weights for pseudo-labeling and training')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for pseudo-labels')
    ap.add_argument('--val_split', type=float, default=0.1, help='Fraction for validation')
    ap.add_argument('--out', type=str, default='datasets/prepared/coco_sample_pseudo', help='Output prepared dataset dir')
    ap.add_argument('--name', type=str, default='coco_sample_pseudo', help='Training run name')
    ap.add_argument('--max-images', type=int, default=0, help='Limit number of images (0 = all) for large datasets')
    ap.add_argument('--resume-labels', action='store_true', help='Skip labeling if label file already exists')
    ap.add_argument('--label-every', type=int, default=0, help='Only generate labels for every Nth image (0=all)')
    args = ap.parse_args()

    # Resolve paths
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather images
    images = fetch_image_paths(args.dataset)
    if args.max_images and args.max_images > 0:
        if args.max_images < len(images):
            print(f"Limiting to first {args.max_images} images out of {len(images)} total.")
            images = images[:args.max_images]
        else:
            print(f"max-images ({args.max_images}) >= available images ({len(images)}); using all.")
    random.shuffle(images)
    n_val = max(1, int(len(images) * args.val_split))
    val_set = set(images[:n_val])

    # Prepare folders
    dirs = prepare_dirs(out_dir)

    # Load model once
    model = YOLO(args.weights)

    print(f"Preparing pseudo-labels for {len(images)} images with conf>={args.conf}...")

    for idx, img_path in enumerate(images, start=1):
        try:
            is_val = img_path in val_set
            img_dst_dir = dirs['images_val'] if is_val else dirs['images_train']
            lbl_dst_dir = dirs['labels_val'] if is_val else dirs['labels_train']
            img_src = Path(img_path)
            img_dst = img_dst_dir / img_src.name
            label_path = lbl_dst_dir / (img_src.stem + '.txt')

            # Copy image only if not present
            if not img_dst.exists():
                shutil.copy2(img_src, img_dst)

            # Optional: skip labeling on images already labeled when resuming
            if args.resume_labels and label_path.exists():
                if idx % 500 == 0:
                    print(f"(resume) Skipped existing label for {img_src.name}")
                continue

            # Optional sampling: label only every Nth image
            if args.label_every and args.label_every > 0 and idx % args.label_every != 0:
                # Create empty label file if not present
                if not label_path.exists():
                    label_path.write_text('', encoding='utf-8')
                if idx % 200 == 0:
                    print(f"(sampling) Processed {idx}/{len(images)} images...")
                continue

            results = model.predict(source=str(img_src), imgsz=args.imgsz, conf=args.conf, verbose=False)
            if not results:
                label_path.write_text('', encoding='utf-8')
            else:
                r = results[0]
                if r.boxes is None or r.boxes.xywhn is None or len(r.boxes) == 0:
                    label_path.write_text('', encoding='utf-8')
                else:
                    xywhn = r.boxes.xywhn.cpu()
                    cls = r.boxes.cls.cpu()
                    save_yolo_label(label_path, xywhn, cls)

            if idx % 200 == 0:
                print(f"Processed {idx}/{len(images)} images...")
        except Exception as e:
            print(f"Warning: failed {img_path}: {e}")

    # Write dataset YAML
    yaml_path = out_dir / 'dataset.yaml'
    yaml_content = f"""
# Auto-generated pseudo-label dataset
path: {out_dir.as_posix()}
train: images/train
val: images/val
nc: 80
names: [person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light,
        fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow,
        elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee,
        skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard,
        tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich,
        orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed,
        dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven,
        toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush]
""".strip()
    yaml_path.write_text(yaml_content, encoding='utf-8')
    print(f"Dataset YAML written: {yaml_path}")

    # Train
    print("Starting YOLO training...")
    finetune = YOLO(args.weights)
    finetune.train(data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, name=args.name)
    print("Training complete.")


if __name__ == '__main__':
    main()
