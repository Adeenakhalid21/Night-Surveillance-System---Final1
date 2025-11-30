#!/usr/bin/env python3
r"""Register a folder of images as a dataset in the local SQLite DB.

Usage (PowerShell from repo root):
    # Register COCO sample (2000 images)
    python "Night Surveillance System - Final1\Night Surveillance System - Final1\scripts\register_dataset_folder.py" `
        --name "COCO Train2017 Sample 2000" `
        --folder "Night Surveillance System - Final1\Night Surveillance System - Final1\datasets\coco_train2017_sample_sample2000"

    # Register LOL low-light dataset (our485/low)
    python "Night Surveillance System - Final1\Night Surveillance System - Final1\scripts\register_dataset_folder.py" `
        --name "LOL LowLight" `
        --folder "Night Surveillance System - Final1\Night Surveillance System - Final1\datasets\lol_dataset\our485\low"

This script:
  1. Ensures a row exists in `datasets` with the given name.
  2. Inserts image paths into `training_data` if not already present.

Notes:
  - Only uses local SQLite (`night_surveillance.db`). Training script also uses SQLite.
    - Image paths stored relative to repo root when possible ("datasets/...").
  - Supported extensions mirror `train_pseudolabel_yolo.py`.
"""
from __future__ import annotations
import argparse, os, sqlite3
from pathlib import Path

DB_PATH = 'night_surveillance.db'
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_dataset(conn, name: str, dtype: str, description: str) -> int:
    row = conn.execute('SELECT dataset_id FROM datasets WHERE dataset_name=?', (name,)).fetchone()
    if row:
        return row['dataset_id']
    conn.execute('''
        INSERT INTO datasets (dataset_name, dataset_type, description, total_samples, file_path, is_active)
        VALUES (?, ?, ?, 0, NULL, 1)
    ''', (name, dtype, description))
    conn.commit()
    row = conn.execute('SELECT dataset_id FROM datasets WHERE dataset_name=?', (name,)).fetchone()
    return row['dataset_id']

def normalize_path(p: Path, repo_root: Path) -> str:
    try:
        p_abs = p.resolve()
        repo_abs = repo_root.resolve()
        rel = p_abs.relative_to(repo_abs)
        # Prefer relative paths starting with datasets/ for consistency
        return str(rel).replace('\\', '/')
    except Exception:
        return str(p_abs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, help='Dataset name to register')
    ap.add_argument('--folder', required=True, help='Folder containing images')
    ap.add_argument('--type', default='image', help='Dataset type label')
    ap.add_argument('--description', default='Imported folder dataset', help='Description')
    ap.add_argument('--dry-run', action='store_true', help='List images and exit without inserting')
    args = ap.parse_args()

    repo_root = Path.cwd()  # Expect running from repo root
    folder = Path(args.folder)
    if not folder.exists():
        raise SystemExit(f'Folder does not exist: {folder}')

    # Collect images
    images = [p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise SystemExit('No images found with supported extensions.')

    print(f'Found {len(images)} images. Connecting to DB...')
    conn = connect()

    # Ensure tables (mirrors main.py init subset needed here)
    conn.execute('''CREATE TABLE IF NOT EXISTS datasets (
        dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_name TEXT UNIQUE NOT NULL,
        dataset_type TEXT NOT NULL,
        description TEXT,
        created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        total_samples INTEGER DEFAULT 0,
        file_path TEXT,
        is_active BOOLEAN DEFAULT 1
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS training_data (
        sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER NOT NULL,
        image_path TEXT NOT NULL,
        annotation_path TEXT,
        label TEXT,
        category TEXT,
        is_validated BOOLEAN DEFAULT 0,
        added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
    )''')
    conn.commit()

    dataset_id = ensure_dataset(conn, args.name, args.type, args.description)
    print(f'Dataset id: {dataset_id}')

    if args.dry_run:
        for p in images[:10]:
            print('Sample image:', p)
        print('(dry-run) Skipping inserts.')
        return

    inserted = 0
    for p in images:
        norm = normalize_path(p, repo_root)
        # Avoid duplicates
        row = conn.execute('SELECT 1 FROM training_data WHERE dataset_id=? AND image_path=?', (dataset_id, norm)).fetchone()
        if row:
            continue
        conn.execute('INSERT INTO training_data (dataset_id, image_path, category) VALUES (?, ?, ?)', (dataset_id, norm, None))
        inserted += 1
        if inserted % 500 == 0:
            print(f'Inserted {inserted} images...')
    conn.commit()

    # Update total_samples
    conn.execute('UPDATE datasets SET total_samples=(SELECT COUNT(*) FROM training_data WHERE dataset_id=?) WHERE dataset_id=?', (dataset_id, dataset_id))
    conn.commit()
    final_count = conn.execute('SELECT total_samples FROM datasets WHERE dataset_id=?', (dataset_id,)).fetchone()['total_samples']
    conn.close()
    print(f'Completed. Added {inserted} new images. Total now: {final_count}.')
    print('Next: run train_pseudolabel_yolo.py with --dataset "{0}"'.format(args.name))

if __name__ == '__main__':
    main()
