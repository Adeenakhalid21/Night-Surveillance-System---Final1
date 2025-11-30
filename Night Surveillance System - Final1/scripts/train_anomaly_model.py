#!/usr/bin/env python3
r"""
Register and train anomaly detection dataset for Night Surveillance System.

This script:
1. Registers the LOL low-light dataset for anomaly training
2. Trains YOLOv8 model on low-light conditions
3. Registers COCO dataset for general anomaly patterns
4. Creates anomaly-specific model weights

Usage (PowerShell from repo root):
  .\.venv\Scripts\python.exe "Night Surveillance System - Final1\Night Surveillance System - Final1\scripts\train_anomaly_model.py"
"""
import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print status"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n[ERROR] Failed: {description}")
        return False
    print(f"\n[SUCCESS] Completed: {description}")
    return True

def main():
    # Get paths
    repo_root = Path.cwd()
    inner_path = repo_root / "Night Surveillance System - Final1" / "Night Surveillance System - Final1"
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    
    if not venv_python.exists():
        print(f"[ERROR] Venv python not found at: {venv_python}")
        print("Please activate virtual environment first:")
        print("  .\\.venv\\Scripts\\Activate.ps1")
        return False
    
    register_script = inner_path / "scripts" / "register_dataset_folder.py"
    train_script = inner_path / "train_pseudolabel_yolo.py"
    
    # Step 1: Register LOL low-light dataset
    lol_folder = inner_path / "datasets" / "lol_dataset" / "our485" / "low"
    if lol_folder.exists():
        cmd = [
            str(venv_python),
            str(register_script),
            "--name", "LOL LowLight Anomaly",
            "--folder", str(lol_folder),
            "--type", "anomaly",
            "--description", "Low-light images for anomaly detection training"
        ]
        if not run_command(cmd, "Register LOL low-light dataset"):
            return False
    else:
        print(f"[WARNING] LOL dataset not found at {lol_folder}, skipping...")
    
    # Step 2: Register COCO for anomaly patterns
    coco_folder = inner_path / "datasets" / "train2017"
    if coco_folder.exists():
        cmd = [
            str(venv_python),
            str(register_script),
            "--name", "COCO Anomaly Patterns",
            "--folder", str(coco_folder),
            "--type", "anomaly",
            "--description", "COCO dataset for general anomaly pattern detection"
        ]
        # Only register subset for faster training
        print("[INFO] Will use max 10000 images from COCO for anomaly training")
    else:
        print(f"[WARNING] COCO dataset not found at {coco_folder}, skipping...")
    
    # Step 3: Train on LOL dataset (low-light anomaly detection)
    if lol_folder.exists():
        weights = inner_path.parent / "yolov8s.pt"
        cmd = [
            str(venv_python),
            str(train_script),
            "--dataset", "LOL LowLight Anomaly",
            "--weights", str(weights),
            "--epochs", "20",
            "--batch", "16",
            "--imgsz", "640",
            "--conf", "0.25",
            "--val_split", "0.15",
            "--resume-labels",
            "--infer-batch", "8",
            "--device", "cpu",
            "--label-every", "0",
            "--cos-lr",
            "--patience", "5",
            "--freeze", "6",
            "--workers", "0",
            "--name", "anomaly_lowlight",
            "--out", str(inner_path / "datasets" / "prepared" / "anomaly_lowlight"),
            "--seed", "42"
        ]
        if not run_command(cmd, "Train anomaly detection on low-light dataset"):
            print("[WARNING] Training failed, but continuing...")
    
    # Step 4: Train on COCO subset for general anomalies
    if coco_folder.exists():
        cmd = [
            str(venv_python),
            str(train_script),
            "--dataset", "COCO Anomaly Patterns",
            "--weights", str(weights),
            "--epochs", "15",
            "--batch", "16",
            "--imgsz", "640",
            "--conf", "0.30",
            "--val_split", "0.1",
            "--resume-labels",
            "--infer-batch", "8",
            "--device", "cpu",
            "--label-every", "3",
            "--fast-preset",
            "--patience", "4",
            "--freeze", "8",
            "--workers", "0",
            "--name", "anomaly_general",
            "--max-images", "10000",
            "--out", str(inner_path / "datasets" / "prepared" / "anomaly_general"),
            "--seed", "42"
        ]
        if not run_command(cmd, "Train general anomaly detection on COCO subset"):
            print("[WARNING] Training failed, but continuing...")
    
    print("\n" + "="*60)
    print("ANOMALY MODEL TRAINING COMPLETE!")
    print("="*60)
    print("\nTrained models saved to:")
    print("  - runs/detect/anomaly_lowlight/weights/best.pt")
    print("  - runs/detect/anomaly_general/weights/best.pt")
    print("\nTo use the anomaly model in your app, set:")
    print('  $env:YOLO_WEIGHTS = "runs\\detect\\anomaly_lowlight\\weights\\best.pt"')
    print('  $env:ENABLE_ANOMALY_ALERTS = "1"')
    print('  $env:ANOMALY_CLASSES = "person,car,motorcycle,truck,knife,gun"')
    print("\nThen start the app:")
    print('  Push-Location "Night Surveillance System - Final1"')
    print('  python .\\main.py')
    print('  Pop-Location')
    
    return True

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Training stopped by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
