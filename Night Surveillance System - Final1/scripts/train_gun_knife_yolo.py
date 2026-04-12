#!/usr/bin/env python3
"""Prepare and train a 2-class YOLO model for knife and gun detection.

Data sources:
- Guns: datasets/gun_dataser (custom xyxy label format)
- Knives: datasets/prepared/coco_full5k_pseudo (YOLO class 43 from COCO pseudo labels)

Outputs:
- Prepared dataset: datasets/prepared/gun_knife_binary
- Dataset yaml: datasets/prepared/gun_knife_binary/dataset.yaml
- Trained weights: runs/detect/gun_knife_binary/weights/best.pt
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def resolve_base_weights(weights_arg: str) -> str:
    if weights_arg.strip().lower() != "auto":
        path = Path(weights_arg)
        # Allow either local paths or Ultralytics aliases like yolov8n.pt.
        return str(path) if path.exists() else weights_arg

    candidates = [
        Path("runs/detect/dataset2_weapon_detection_best.pt"),
        Path("runs/detect/dataset2_weapon_detection/weights/best.pt"),
        Path("runs/detect/gun_knife_binary/weights/best.pt"),
        Path("yolov8s.pt"),
        Path("yolov8n.pt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            print(f"[train] using base weights: {candidate}")
            return str(candidate)

    print("[train] no local weights found, fallback to yolov8n.pt")
    return "yolov8n.pt"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gun-source", type=str, default="datasets/gun_dataser")
    ap.add_argument("--knife-source", type=str, default="datasets/prepared/coco_full5k_pseudo")
    ap.add_argument("--out", type=str, default="datasets/prepared/gun_knife_binary")
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights", type=str, default="auto")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--freeze", type=int, default=8)
    ap.add_argument("--name", type=str, default="gun_knife_binary")
    ap.add_argument("--prepare-only", action="store_true")
    return ap.parse_args()


def find_image_by_stem(folder: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def read_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def convert_gun_labels(gun_source: Path) -> list[tuple[Path, list[str]]]:
    images_dir = gun_source / "Images"
    labels_dir = gun_source / "Labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise SystemExit(f"Gun dataset folders not found in {gun_source}")

    samples: list[tuple[Path, list[str]]] = []
    skipped = 0

    for lbl in sorted(labels_dir.glob("*.txt")):
        img = find_image_by_stem(images_dir, lbl.stem)
        if img is None:
            skipped += 1
            continue

        lines = read_lines(lbl)
        if not lines:
            skipped += 1
            continue

        # Format: first line is object count, remaining lines are x1 y1 x2 y2.
        try:
            from PIL import Image

            w, h = Image.open(img).size
        except Exception:
            skipped += 1
            continue

        yolo_lines: list[str] = []
        for row in lines[1:]:
            parts = row.split()
            if len(parts) != 4:
                continue
            try:
                x1, y1, x2, y2 = map(float, parts)
            except ValueError:
                continue
            if x2 <= x1 or y2 <= y1:
                continue

            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            cx = clamp01(cx)
            cy = clamp01(cy)
            bw = clamp01(bw)
            bh = clamp01(bh)
            if bw <= 0 or bh <= 0:
                continue

            # Class 1 = gun
            yolo_lines.append(f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if yolo_lines:
            samples.append((img, yolo_lines))
        else:
            skipped += 1

    print(f"[prep] gun samples: {len(samples)} (skipped: {skipped})")
    return samples


def find_matching_image(images_dir: Path, split: str, stem: str) -> Path | None:
    split_dir = images_dir / split
    for ext in IMAGE_EXTS:
        p = split_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def convert_knife_labels(knife_source: Path) -> list[tuple[Path, list[str]]]:
    labels_root = knife_source / "labels"
    images_root = knife_source / "images"
    if not labels_root.exists() or not images_root.exists():
        raise SystemExit(f"Knife dataset folders not found in {knife_source}")

    samples: list[tuple[Path, list[str]]] = []

    for split in ("train", "val"):
        lbl_dir = labels_root / split
        if not lbl_dir.exists():
            continue
        for lbl in sorted(lbl_dir.glob("*.txt")):
            lines = read_lines(lbl)
            if not lines:
                continue

            knife_lines: list[str] = []
            for row in lines:
                parts = row.split()
                if len(parts) < 5:
                    continue
                if parts[0] != "43":
                    continue
                # Class 0 = knife (keep xywhn as-is)
                knife_lines.append("0 " + " ".join(parts[1:5]))

            if not knife_lines:
                continue

            img = find_matching_image(images_root, split, lbl.stem)
            if img is None:
                continue

            samples.append((img, knife_lines))

    print(f"[prep] knife samples: {len(samples)}")
    return samples


def split_samples(samples: list[tuple[Path, list[str]]], val_split: float, seed: int) -> tuple[list[tuple[Path, list[str]]], list[tuple[Path, list[str]]]]:
    rng = random.Random(seed)
    items = list(samples)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * val_split)) if items else 0
    val = items[:n_val]
    train = items[n_val:]
    return train, val


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    paths = {
        "train_img": out_dir / "images" / "train",
        "val_img": out_dir / "images" / "val",
        "train_lbl": out_dir / "labels" / "train",
        "val_lbl": out_dir / "labels" / "val",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def write_dataset_yaml(out_dir: Path) -> Path:
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "# Auto-generated knife+gun dataset",
                f"path: {out_dir.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "nc: 2",
                "names: [knife, gun]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return yaml_path


def write_split(samples: Iterable[tuple[Path, list[str]]], img_dir: Path, lbl_dir: Path, prefix: str) -> int:
    count = 0
    for i, (img_src, yolo_lines) in enumerate(samples, start=1):
        stem = f"{prefix}_{i:05d}"
        img_dst = img_dir / f"{stem}{img_src.suffix.lower()}"
        lbl_dst = lbl_dir / f"{stem}.txt"
        shutil.copy2(img_src, img_dst)
        lbl_dst.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        count += 1
    return count


def prepare_dataset(args: argparse.Namespace) -> Path:
    gun_source = Path(args.gun_source)
    knife_source = Path(args.knife_source)
    out_dir = Path(args.out)

    if out_dir.exists():
        shutil.rmtree(out_dir)

    paths = ensure_dirs(out_dir)

    gun_samples = convert_gun_labels(gun_source)
    knife_samples = convert_knife_labels(knife_source)

    if not gun_samples:
        raise SystemExit("No gun samples found after conversion.")
    if not knife_samples:
        raise SystemExit("No knife samples found in knife source dataset.")

    gun_train, gun_val = split_samples(gun_samples, args.val_split, args.seed)
    knife_train, knife_val = split_samples(knife_samples, args.val_split, args.seed + 7)

    n_gt = write_split(gun_train, paths["train_img"], paths["train_lbl"], "gun")
    n_gv = write_split(gun_val, paths["val_img"], paths["val_lbl"], "gun")
    n_kt = write_split(knife_train, paths["train_img"], paths["train_lbl"], "knife")
    n_kv = write_split(knife_val, paths["val_img"], paths["val_lbl"], "knife")

    yaml_path = write_dataset_yaml(out_dir)

    print("[prep] dataset ready")
    print(f"[prep] train gun={n_gt}, train knife={n_kt}")
    print(f"[prep] val gun={n_gv}, val knife={n_kv}")
    print(f"[prep] yaml: {yaml_path}")
    return yaml_path


def train_model(args: argparse.Namespace, yaml_path: Path) -> None:
    from ultralytics import YOLO

    base_weights = resolve_base_weights(args.weights)
    print(f"[train] base checkpoint: {base_weights}")
    model = YOLO(base_weights)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        patience=args.patience,
        freeze=args.freeze,
        cos_lr=True,
    )
    print("[train] completed")
    print(f"[train] expected best weights: runs/detect/{args.name}/weights/best.pt")


def main() -> None:
    args = parse_args()
    yaml_path = prepare_dataset(args)
    if args.prepare_only:
        print("[done] prepare-only mode enabled, skipping training")
        return
    train_model(args, yaml_path)


if __name__ == "__main__":
    main()
