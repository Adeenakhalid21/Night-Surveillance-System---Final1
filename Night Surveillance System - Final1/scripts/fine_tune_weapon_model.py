#!/usr/bin/env python3
"""Fine-tune weapon detection model and save accuracy graphs.

Default behavior:
- Base weights: runs/detect/dataset2_weapon_detection_best.pt
- Data yaml: datasets/prepared/gun_knife_binary/dataset.yaml
- Output run: runs/detect/gun_knife_finetune_v2
- Accuracy graph: runs/detect/<name>/accuracy_graph.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights",
        type=str,
        default="runs/detect/dataset2_weapon_detection_best.pt",
        help="Base checkpoint to fine-tune from.",
    )
    ap.add_argument(
        "--data",
        type=str,
        default="datasets/prepared/gun_knife_binary/dataset.yaml",
        help="YOLO dataset YAML path.",
    )
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--freeze", type=int, default=4)
    ap.add_argument("--name", type=str, default="gun_knife_finetune_v2")
    ap.add_argument("--project", type=str, default="runs/detect")
    ap.add_argument(
        "--graph-only-run-dir",
        type=str,
        default="",
        help="If provided, skip training and generate accuracy_graph.png from run_dir/results.csv.",
    )
    return ap.parse_args()


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

    precision_key = pick_metric_key(header, [
        "metrics/precision(B)",
        "metrics/precision",
        "metrics/precision(B)_val",
    ])
    recall_key = pick_metric_key(header, [
        "metrics/recall(B)",
        "metrics/recall",
        "metrics/recall(B)_val",
    ])
    map50_key = pick_metric_key(header, [
        "metrics/mAP50(B)",
        "metrics/mAP_0.5",
        "metrics/mAP50",
    ])
    map5095_key = pick_metric_key(header, [
        "metrics/mAP50-95(B)",
        "metrics/mAP_0.5:0.95",
        "metrics/mAP50-95",
    ])

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

    ax.set_title("Weapon Fine-Tuning Accuracy Metrics")
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


def prepare_runtime_data_yaml(data_yaml: Path) -> Path:
    """Create a temporary YAML that always points to the current dataset folder.

    This prevents training failures caused by stale absolute paths from another machine.
    """
    lines = data_yaml.read_text(encoding="utf-8").splitlines()
    dataset_root = data_yaml.parent.resolve().as_posix()
    replaced = False
    out_lines: list[str] = []

    for line in lines:
        if line.strip().startswith("path:"):
            out_lines.append(f"path: {dataset_root}")
            replaced = True
        else:
            out_lines.append(line)

    if not replaced:
        out_lines.insert(0, f"path: {dataset_root}")

    runtime_yaml = data_yaml.with_name(f"{data_yaml.stem}.runtime.yaml")
    runtime_yaml.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return runtime_yaml


def main() -> None:
    args = parse_args()

    if args.graph_only_run_dir:
        run_dir = Path(args.graph_only_run_dir)
        graph_path = save_accuracy_graph(run_dir)
        print(f"Graph generated: {graph_path}")
        return

    weights = Path(args.weights)
    data = Path(args.data)

    if not weights.exists():
        raise FileNotFoundError(f"Base weights not found: {weights}")
    if not data.exists():
        raise FileNotFoundError(f"Data yaml not found: {data}")

    runtime_data = prepare_runtime_data_yaml(data)

    model = YOLO(str(weights))
    run_dir = Path(args.project) / args.name

    try:
        result = model.train(
            data=str(runtime_data),
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
        run_dir = Path(getattr(result, "save_dir", run_dir))
    except Exception as exc:
        # Some environments have a broken scipy install used by Ultralytics plotting.
        # If training artifacts were saved, continue to graph generation.
        if "scipy" not in str(exc).lower() or not (run_dir / "results.csv").exists():
            raise

    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    graph_path = save_accuracy_graph(run_dir)

    print("\nFine-tuning finished.")
    print(f"Run directory : {run_dir}")
    print(f"Best weights  : {best_pt}")
    print(f"Last weights  : {last_pt}")
    print(f"Results CSV   : {run_dir / 'results.csv'}")
    print(f"Accuracy graph: {graph_path}")
    print(f"Runtime YAML  : {runtime_data}")


if __name__ == "__main__":
    main()
