#!/usr/bin/env python3
"""Prepare and fine-tune a YOLO model for object/person/weapon detection.

This script builds a 3-class dataset:
- class 0: person
- class 1: knife
- class 2: gun

Sources:
- Person samples: datasets/prepared/coco_full5k_pseudo (COCO class 0 only)
- Weapon samples: datasets/prepared/gun_knife_binary (knife=0, gun=1)

Outputs:
- Prepared dataset: datasets/prepared/object_person_weapon
- Dataset YAML: datasets/prepared/object_person_weapon/dataset.yaml
- Trained run: runs/detect/object_person_weapon_ft
- Accuracy graph: runs/detect/<name>/accuracy_graph.png
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--person-source", type=str, default="datasets/prepared/coco_full5k_pseudo")
    ap.add_argument("--weapon-source", type=str, default="datasets/prepared/gun_knife_binary")
    ap.add_argument("--out", type=str, default="datasets/prepared/object_person_weapon")
    ap.add_argument("--weights", type=str, default="auto")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--freeze", type=int, default=6)
    ap.add_argument("--name", type=str, default="object_person_weapon_ft")
    ap.add_argument("--project", type=str, default="runs/detect")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-person-train", type=int, default=5000)
    ap.add_argument("--max-person-val", type=int, default=1200)
    ap.add_argument("--prepare-only", action="store_true")
    return ap.parse_args()


def resolve_base_weights(weights_arg: str) -> str:
    if weights_arg.strip().lower() != "auto":
        p = Path(weights_arg)
        return str(p) if p.exists() else weights_arg

    candidates = [
        Path("runs/detect/object_person_weapon_ft/weights/best.pt"),
        Path("runs/detect/gun_knife_finetune_v4/weights/best.pt"),
        Path("runs/detect/gun_knife_hand_ft/weights/best.pt"),
        Path("runs/detect/dataset2_weapon_detection_best.pt"),
        Path("yolov8s.pt"),
        Path("yolov8n.pt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            print(f"[train] using base weights: {candidate}")
            return str(candidate)

    print("[train] no local weights found, fallback to yolov8n.pt")
    return "yolov8n.pt"


def find_image_by_stem(folder: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def read_non_empty_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]


def ensure_out_dirs(out_dir: Path) -> dict[str, Path]:
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
                "# Auto-generated person+weapon dataset",
                f"path: {out_dir.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "nc: 3",
                "names: [person, knife, gun]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return yaml_path


def collect_person_samples(person_source: Path, split: str, max_images: int, seed: int) -> list[tuple[Path, list[str]]]:
    labels_dir = person_source / "labels" / split
    images_dir = person_source / "images" / split
    if not labels_dir.exists() or not images_dir.exists():
        return []

    rng = random.Random(seed)
    label_files = sorted(labels_dir.glob("*.txt"))
    rng.shuffle(label_files)

    samples: list[tuple[Path, list[str]]] = []
    for lbl in label_files:
        lines = read_non_empty_lines(lbl)
        person_lines: list[str] = []
        for row in lines:
            parts = row.split()
            if len(parts) < 5:
                continue
            if parts[0] != "0":
                continue
            # Keep person as class 0.
            person_lines.append("0 " + " ".join(parts[1:5]))

        if not person_lines:
            continue

        img = find_image_by_stem(images_dir, lbl.stem)
        if img is None:
            continue

        samples.append((img, person_lines))
        if max_images > 0 and len(samples) >= max_images:
            break

    return samples


def collect_weapon_samples(weapon_source: Path, split: str) -> list[tuple[Path, list[str]]]:
    labels_dir = weapon_source / "labels" / split
    images_dir = weapon_source / "images" / split
    if not labels_dir.exists() or not images_dir.exists():
        return []

    samples: list[tuple[Path, list[str]]] = []
    for lbl in sorted(labels_dir.glob("*.txt")):
        lines = read_non_empty_lines(lbl)
        mapped_lines: list[str] = []
        for row in lines:
            parts = row.split()
            if len(parts) < 5:
                continue
            cls = parts[0]
            if cls == "0":
                # knife: 0 -> 1
                mapped_lines.append("1 " + " ".join(parts[1:5]))
            elif cls == "1":
                # gun: 1 -> 2
                mapped_lines.append("2 " + " ".join(parts[1:5]))

        if not mapped_lines:
            continue

        img = find_image_by_stem(images_dir, lbl.stem)
        if img is None:
            continue

        samples.append((img, mapped_lines))

    return samples


def write_split(samples: list[tuple[Path, list[str]]], img_dir: Path, lbl_dir: Path, prefix: str) -> int:
    count = 0
    for idx, (img_src, yolo_lines) in enumerate(samples, start=1):
        stem = f"{prefix}_{idx:05d}"
        img_dst = img_dir / f"{stem}{img_src.suffix.lower()}"
        lbl_dst = lbl_dir / f"{stem}.txt"
        shutil.copy2(img_src, img_dst)
        lbl_dst.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        count += 1
    return count


def prepare_dataset(args: argparse.Namespace) -> Path:
    person_source = Path(args.person_source)
    weapon_source = Path(args.weapon_source)
    out_dir = Path(args.out)

    if out_dir.exists():
        shutil.rmtree(out_dir)

    paths = ensure_out_dirs(out_dir)

    person_train = collect_person_samples(person_source, "train", args.max_person_train, args.seed)
    person_val = collect_person_samples(person_source, "val", args.max_person_val, args.seed + 3)
    weapon_train = collect_weapon_samples(weapon_source, "train")
    weapon_val = collect_weapon_samples(weapon_source, "val")

    if not person_train:
        raise SystemExit(f"No person samples found in {person_source}")
    if not weapon_train:
        raise SystemExit(f"No weapon samples found in {weapon_source}")

    n_pt = write_split(person_train, paths["train_img"], paths["train_lbl"], "person")
    n_pv = write_split(person_val, paths["val_img"], paths["val_lbl"], "person")
    n_wt = write_split(weapon_train, paths["train_img"], paths["train_lbl"], "weapon")
    n_wv = write_split(weapon_val, paths["val_img"], paths["val_lbl"], "weapon")

    yaml_path = write_dataset_yaml(out_dir)

    print("[prep] combined dataset ready")
    print(f"[prep] train person={n_pt}, train weapon={n_wt}")
    print(f"[prep] val person={n_pv}, val weapon={n_wv}")
    print(f"[prep] yaml: {yaml_path}")

    return yaml_path


def pick_metric_key(header: list[str], candidates: list[str]) -> str | None:
    lower_to_original = {h.strip().lower(): h for h in header}
    for key in candidates:
        found = lower_to_original.get(key.lower())
        if found:
            return found
    return None


def read_metric_series(csv_path: Path, column_name: str) -> list[float]:
    values: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = (row.get(column_name) or "").strip()
            if not raw:
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
    return values


def save_accuracy_graph(run_dir: Path) -> Path:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv not found at {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

    precision_key = pick_metric_key(header, ["metrics/precision(B)", "metrics/precision"])
    recall_key = pick_metric_key(header, ["metrics/recall(B)", "metrics/recall"])
    map50_key = pick_metric_key(header, ["metrics/mAP50(B)", "metrics/mAP50"])
    map5095_key = pick_metric_key(header, ["metrics/mAP50-95(B)", "metrics/mAP50-95"])

    if not all([precision_key, recall_key, map50_key, map5095_key]):
        raise RuntimeError("Could not find expected metric columns in results.csv")

    precision = read_metric_series(csv_path, precision_key)
    recall = read_metric_series(csv_path, recall_key)
    map50 = read_metric_series(csv_path, map50_key)
    map5095 = read_metric_series(csv_path, map5095_key)

    n_points = max(len(precision), len(recall), len(map50), len(map5095))
    epochs = list(range(1, n_points + 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    if precision:
        ax.plot(epochs[: len(precision)], precision, marker="o", linewidth=2, label="Precision")
    if recall:
        ax.plot(epochs[: len(recall)], recall, marker="o", linewidth=2, label="Recall")
    if map50:
        ax.plot(epochs[: len(map50)], map50, marker="o", linewidth=2, label="mAP@50")
    if map5095:
        ax.plot(epochs[: len(map5095)], map5095, marker="o", linewidth=2, label="mAP@50-95")

    ax.set_title("Object-Person-Weapon Fine-Tuning Metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_graph = run_dir / "accuracy_graph.png"
    fig.tight_layout()
    fig.savefig(out_graph, dpi=160)
    plt.close(fig)
    return out_graph


def train_model(args: argparse.Namespace, yaml_path: Path) -> None:
    from ultralytics import YOLO

    base_weights = resolve_base_weights(args.weights)
    print(f"[train] base checkpoint: {base_weights}")
    model = YOLO(base_weights)

    result = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        project=args.project,
        patience=args.patience,
        freeze=args.freeze,
        cos_lr=True,
        plots=False,
    )

    run_dir = Path(getattr(result, "save_dir", Path(args.project) / args.name))
    graph_path = save_accuracy_graph(run_dir)

    print("\nFine-tuning finished.")
    print(f"Run directory : {run_dir}")
    print(f"Best weights  : {run_dir / 'weights' / 'best.pt'}")
    print(f"Results CSV   : {run_dir / 'results.csv'}")
    print(f"Accuracy graph: {graph_path}")


def main() -> None:
    args = parse_args()
    yaml_path = prepare_dataset(args)
    if args.prepare_only:
        print("[done] prepare-only mode enabled, skipping training")
        return
    train_model(args, yaml_path)


if __name__ == "__main__":
    main()
