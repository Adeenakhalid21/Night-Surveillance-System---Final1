#!/usr/bin/env python3
"""Dataset Import Utility (LOL & VisDrone) for Night Surveillance System.

Imports low-light (LOL) and VisDrone datasets into the SQLite database.

Usage:
  python import_lol_dataset.py <source_path> [dataset_name]
    - Imports LOL dataset by default
  python import_lol_dataset.py visdrone <source_path> [dataset_name]
    - Imports VisDrone dataset (prefix 'visdrone')
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
from typing import List, Dict, Optional

DATABASE = 'night_surveillance.db'
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def _insert_dataset(conn: sqlite3.Connection, name: str, dtype: str, description: str, total: int, path: str) -> int:
    existing = conn.execute('SELECT dataset_id FROM datasets WHERE dataset_name=?', (name,)).fetchone()
    if existing:
        dataset_id = existing['dataset_id']
        conn.execute('UPDATE datasets SET total_samples=?, file_path=?, description=? WHERE dataset_id=?',
                     (total, path, description, dataset_id))
        conn.execute('DELETE FROM training_data WHERE dataset_id=?', (dataset_id,))
        return dataset_id
    cur = conn.cursor()
    cur.execute('INSERT INTO datasets (dataset_name, dataset_type, description, total_samples, file_path, is_active)'
                ' VALUES (?, ?, ?, ?, ?, ?)', (name, dtype, description, total, path, 1))
    return cur.lastrowid


def _bulk_insert_samples(conn: sqlite3.Connection, dataset_id: int, samples: List[Dict[str, Optional[str]]]) -> None:
    for i, s in enumerate(samples, start=1):
        conn.execute('INSERT INTO training_data (dataset_id, image_path, annotation_path, label, category, is_validated)'
                     ' VALUES (?, ?, ?, ?, ?, ?)',
                     (dataset_id, s['image_path'], s.get('annotation_path'), s['label'], s['category'], 1))
        if i % 500 == 0:
            print(f"💾 Inserted {i}/{len(samples)} samples...")


def import_lol_dataset(source_path: str, dataset_name: str = 'LOL Dataset') -> bool:
    if not os.path.isdir(source_path):
        print(f"❌ Path not found: {source_path}")
        return False
    target_dir = 'datasets/lol_dataset'
    os.makedirs(target_dir, exist_ok=True)
    samples: List[Dict[str, Optional[str]]] = []
    copied = 0
    for root, _, files in os.walk(source_path):
        rel = os.path.relpath(root, source_path)
        if rel == '.':
            rel = ''
        dest_dir = os.path.join(target_dir, rel)
        os.makedirs(dest_dir, exist_ok=True)
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                src = os.path.join(root, f)
                dst = os.path.join(dest_dir, f)
                shutil.copy2(src, dst)
                copied += 1
                cat = 'low_light'
                low_rel = rel.lower()
                if any(x in low_rel for x in ('high', 'normal')):
                    cat = 'normal_light'
                samples.append({'image_path': dst, 'annotation_path': None, 'label': cat, 'category': cat})
                if copied % 100 == 0:
                    print(f"📋 Copied {copied} images...")
    print(f"✅ Copied {copied} images total")
    conn = get_db_connection()
    dataset_id = _insert_dataset(conn, dataset_name, 'enhancement',
                                 'LOL Dataset for low-light image enhancement and detection training',
                                 len(samples), target_dir)
    _bulk_insert_samples(conn, dataset_id, samples)
    conn.commit(); conn.close()
    print('🎉 LOL import complete!')
    return True


def import_visdrone_dataset(source_path: str, dataset_name: str = 'VisDrone Dataset') -> bool:
    if not os.path.isdir(source_path):
        print(f"❌ Path not found: {source_path}")
        return False
    target_dir = 'datasets/visdrone_dataset'
    os.makedirs(target_dir, exist_ok=True)
    images_root = source_path
    annotations_root = None
    for root, dirs, _ in os.walk(source_path):
        if 'images' in dirs:
            images_root = os.path.join(root, 'images')
        if 'annotations' in dirs:
            annotations_root = os.path.join(root, 'annotations')
    samples: List[Dict[str, Optional[str]]] = []
    copied = 0
    for root, _, files in os.walk(images_root):
        rel = os.path.relpath(root, images_root)
        if rel == '.':
            rel = ''
        dest_dir = os.path.join(target_dir, 'images', rel)
        os.makedirs(dest_dir, exist_ok=True)
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                src = os.path.join(root, f)
                dst = os.path.join(dest_dir, f)
                shutil.copy2(src, dst)
                copied += 1
                ann_path = None
                if annotations_root:
                    base = os.path.splitext(f)[0]
                    for ext in ('.txt', '.xml', '.json'):
                        cand = os.path.join(annotations_root, rel, base + ext)
                        if os.path.exists(cand):
                            ann_dest = os.path.join(target_dir, 'annotations', rel)
                            os.makedirs(ann_dest, exist_ok=True)
                            ann_file = os.path.join(ann_dest, base + ext)
                            shutil.copy2(cand, ann_file)
                            ann_path = ann_file
                            break
                samples.append({'image_path': dst, 'annotation_path': ann_path})
                if copied % 100 == 0:
                    print(f"📋 Copied {copied} images...")
    print(f"✅ Copied {copied} images total")
    # Categorize
    categorized: List[Dict[str, Optional[str]]] = []
    for s in samples:
        rel_dir = os.path.relpath(os.path.dirname(s['image_path']), os.path.join(target_dir, 'images'))
        rel_low = rel_dir.lower()
        cat = 'drone_surveillance'
        if 'train' in rel_low:
            cat = 'training'
        elif 'val' in rel_low or 'validation' in rel_low:
            cat = 'validation'
        elif 'test' in rel_low:
            cat = 'testing'
        categorized.append({'image_path': s['image_path'], 'annotation_path': s['annotation_path'], 'label': cat, 'category': cat})
    conn = get_db_connection()
    dataset_id = _insert_dataset(conn, dataset_name, 'detection',
                                 'VisDrone dataset for drone-based object detection, tracking & surveillance',
                                 len(categorized), target_dir)
    _bulk_insert_samples(conn, dataset_id, categorized)
    conn.commit(); conn.close()
    print('🎉 VisDrone import complete!')
    return True


def list_datasets() -> None:
    conn = get_db_connection()
    rows = conn.execute('SELECT dataset_id, dataset_name, dataset_type, total_samples, file_path FROM datasets ORDER BY created_date DESC').fetchall()
    print('\n📋 Current Datasets')
    print('-' * 80)
    for r in rows:
        print(f"ID {r['dataset_id']}: {r['dataset_name']} ({r['dataset_type']}) | Samples {r['total_samples']} | Path {r['file_path']}")
    conn.close()


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python import_lol_dataset.py <source_path> [dataset_name]')
        print('       python import_lol_dataset.py visdrone <source_path> [dataset_name]')
        list_datasets()
        return
    args = sys.argv[1:]
    visdrone_mode = args[0].lower() == 'visdrone'
    if visdrone_mode:
        if len(args) < 2:
            print('Specify VisDrone source path')
            return
        source = args[1]
        name = args[2] if len(args) > 2 else 'VisDrone Dataset'
        import_visdrone_dataset(source, name)
    else:
        source = args[0]
        name = args[1] if len(args) > 1 else 'LOL Dataset'
        import_lol_dataset(source, name)
    list_datasets()


if __name__ == '__main__':
    main()