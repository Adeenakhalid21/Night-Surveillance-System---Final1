#!/usr/bin/env python3
"""
Train YOLOv8 with pseudo-labels generated from a pre-trained model.

Features:
    - Reads images from a dataset registered in SQLite (by --dataset name)
    - Generates YOLO-format labels using a base model's predictions
    - Supports batching for faster pseudo-label inference (--infer-batch)
    - Resume capability (--resume-labels) skips already labeled images
    - Sampling (--label-every N) to sparsely label then later refine
    - Adjustable image size (--imgsz) and device selection (--device)
    - Graceful KeyboardInterrupt handling (partial progress preserved)
    - Random seed control (--seed) for reproducible shuffles

Usage (example):
    python train_pseudolabel_yolo.py \
        --dataset "COCO Train2017 Sample" \
        --weights yolov8s.pt \
        --epochs 30 --batch 32 --imgsz 640 --conf 0.4 \
        --infer-batch 8 --resume-labels --max-images 5000
"""
from __future__ import annotations
import os
import sys
import argparse
import random
import shutil
import sqlite3
from pathlib import Path
from typing import List, Iterable

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
        # Help the user by listing available datasets when paths are invalid
        try:
            conn = get_db_connection()
            ds = conn.execute('SELECT dataset_name, COUNT(*) FROM datasets d JOIN training_data t ON d.dataset_id=t.dataset_id GROUP BY d.dataset_name').fetchall()
            conn.close()
            names = ', '.join([f"{r[0]}({r[1]})" for r in ds])
            print(f"Available datasets with counts: {names}")
        except Exception:
            pass
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
    ap.add_argument('--infer-batch', type=int, default=1, help='Batch size for pseudo-label inference (>=1)')
    ap.add_argument('--device', type=str, default='', help='YOLO device id ("cpu", "0" for first CUDA GPU, blank=auto)')
    ap.add_argument('--seed', type=int, default=0, help='Random seed (0 = no fixed seed)')
    # Training speed/robustness
    ap.add_argument('--freeze', type=int, default=0, help='Freeze N layers for faster fine-tuning')
    ap.add_argument('--patience', type=int, default=5, help='Early stopping patience (epochs)')
    ap.add_argument('--cos-lr', action='store_true', help='Use cosine learning rate schedule')
    ap.add_argument('--workers', type=int, default=0, help='Dataloader workers (0 recommended on Windows/CPU)')
    ap.add_argument('--cache-images', action='store_true', help='Cache images for faster training (RAM heavy)')
    ap.add_argument('--fast-preset', action='store_true', help='Enable a CPU-friendly fast training preset')
    ap.add_argument('--path-repair-from', type=str, default='', help='Old prefix to replace if image paths broken')
    ap.add_argument('--path-repair-to', type=str, default='', help='New prefix to substitute for broken image paths')
    ap.add_argument('--allow-missing', action='store_true', help='Allow missing images (skip existence check)')
    args = ap.parse_args()

    # Resolve paths
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather images
    # Raw image paths (may be broken); fetch without existence filtering so we can repair.
    conn = get_db_connection()
    row = conn.execute('SELECT dataset_id FROM datasets WHERE dataset_name=?', (args.dataset,)).fetchone()
    if not row:
        conn.close(); raise SystemExit(f"Dataset not found: {args.dataset}")
    dataset_id = row['dataset_id']
    db_rows = conn.execute('SELECT image_path FROM training_data WHERE dataset_id=?', (dataset_id,)).fetchall()
    conn.close()
    raw_paths = [r['image_path'] for r in db_rows if r['image_path']]
    repaired = []
    for p in raw_paths:
        candidate = p
        if args.path_repair_from and args.path_repair_to and p.startswith(args.path_repair_from):
            candidate = args.path_repair_to + p[len(args.path_repair_from):]
        # Workspace-relative common case: prepend workspace folder if path starts with 'datasets/'
        if not os.path.isabs(candidate) and candidate.startswith('datasets/'):
            maybe = os.path.join('Night Surveillance System - Final1', candidate)
            if os.path.exists(maybe):
                candidate = maybe
        if args.allow_missing or os.path.exists(candidate):
            repaired.append(candidate)
    if not repaired:
        print('Path repair failed or no images found. Example original paths:', raw_paths[:5])
        if not args.allow_missing:
            print('Hint: try --allow-missing or provide --path-repair-from and --path-repair-to.')
        raise SystemExit('No usable images after path repair attempts.')
    images = repaired
    if args.seed:
        random.seed(args.seed)
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

    # Optional fast preset overrides (for CPU)
    if args.fast_preset:
        # Safer, faster defaults on CPU while keeping accuracy reasonable
        args.imgsz = min(args.imgsz, 512)
        args.infer_batch = max(args.infer_batch, 8)
        args.label_every = args.label_every or 3
        args.conf = min(args.conf, 0.4)
        args.batch = min(args.batch, 16)
        args.epochs = min(args.epochs, 15)
        args.freeze = max(args.freeze, 10)
        args.patience = min(args.patience, 3)
        args.workers = 0
        args.device = args.device or 'cpu'

    # Load model once for pseudo-labeling
    device_arg = args.device if args.device else None
    model = YOLO(args.weights)

    print(f"Preparing pseudo-labels for {len(images)} images with conf>={args.conf} (batch={args.infer_batch})...")
    total = len(images)
    infer_batch = max(1, args.infer_batch)

    try:
        for start in range(0, total, infer_batch):
            chunk = images[start:start + infer_batch]
            # Determine which need labeling (consider resume and sampling logic individually)
            to_predict = []
            per_image_meta = []  # tuples of (img_src, img_dst, label_path, is_val, should_label)
            for idx_in_chunk, img_path in enumerate(chunk, start=start + 1):
                is_val = img_path in val_set
                img_dst_dir = dirs['images_val'] if is_val else dirs['images_train']
                lbl_dst_dir = dirs['labels_val'] if is_val else dirs['labels_train']
                img_src = Path(img_path)
                img_dst = img_dst_dir / img_src.name
                label_path = lbl_dst_dir / (img_src.stem + '.txt')

                if not img_dst.exists():
                    shutil.copy2(img_src, img_dst)

                # Resume skip
                if args.resume_labels and label_path.exists():
                    per_image_meta.append((img_src, img_dst, label_path, is_val, False))
                    continue

                # Sampling skip (create empty label file)
                if args.label_every and args.label_every > 0 and idx_in_chunk % args.label_every != 0:
                    if not label_path.exists():
                        label_path.write_text('', encoding='utf-8')
                    per_image_meta.append((img_src, img_dst, label_path, is_val, False))
                    continue

                # Will label this image
                to_predict.append(str(img_src))
                per_image_meta.append((img_src, img_dst, label_path, is_val, True))

            results = []
            if to_predict:
                try:
                    results = model.predict(source=to_predict, imgsz=args.imgsz, conf=args.conf, verbose=False, device=device_arg)
                except Exception as e:
                    print(f"Batch predict failed (start={start}): {e}")

            # Assign labels
            res_iter: Iterable = iter(results) if results else iter([])
            for idx_in_chunk, meta in enumerate(per_image_meta, start=start + 1):
                img_src, img_dst, label_path, is_val, should_label = meta
                try:
                    if not should_label:
                        if args.resume_labels and idx_in_chunk % 500 == 0:
                            print(f"(resume) Skipped existing label for {img_src.name}")
                        elif args.label_every and args.label_every > 0 and idx_in_chunk % 200 == 0:
                            print(f"(sampling) Processed {idx_in_chunk}/{total} images...")
                        continue
                    r = next(res_iter, None)
                    if r is None or r.boxes is None or r.boxes.xywhn is None or len(r.boxes) == 0:
                        label_path.write_text('', encoding='utf-8')
                    else:
                        xywhn = r.boxes.xywhn.cpu()
                        cls = r.boxes.cls.cpu()
                        save_yolo_label(label_path, xywhn, cls)
                except Exception as e:
                    print(f"Warning: failed {img_src}: {e}")

            if (start + infer_batch) % 200 == 0 or (start + infer_batch) >= total:
                done = min(start + infer_batch, total)
                print(f"Processed {done}/{total} images...")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected: stopping pseudo-label generation early. Partial labels preserved.")

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

    # Train (only if epochs > 0)
    if args.epochs > 0:
        print("Starting YOLO training...")
        finetune = YOLO(args.weights)
        train_kwargs = dict(
            data=str(yaml_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            name=args.name,
            device=device_arg,
            workers=args.workers,
            patience=args.patience,
            freeze=args.freeze,
            cos_lr=args.cos_lr,
        )
        if args.cache_images:
            train_kwargs['cache'] = True
        finetune.train(**train_kwargs)
        print("Training complete.")
    else:
        print("Epochs set to 0 - skipping training phase (labels only).")


if __name__ == '__main__':
    main()
