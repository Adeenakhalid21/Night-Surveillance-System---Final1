#!/usr/bin/env python3
"""
Dataset Import Utility for Night Surveillance System
This script helps import external datasets like the LOL Dataset into the surveillance system.
"""

import os
import shutil
import sqlite3
import sys
from pathlib import Path

# Database configuration
DATABASE = 'night_surveillance.db'

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def import_visdrone_dataset(source_path, target_name="VisDrone Dataset"):
    """Import VisDrone Dataset into the surveillance system"""
    
    print(f"🔄 Importing VisDrone Dataset from: {source_path}")
    
    # Validate source path
    if not os.path.exists(source_path):
        print(f"❌ Error: Source path does not exist: {source_path}")
        return False
    
    # Create target directory in datasets folder
    target_dir = f"datasets/visdrone_dataset"
    os.makedirs(target_dir, exist_ok=True)
    
    # VisDrone typical structure: images/ and annotations/
    images_dir = None
    annotations_dir = None
    
    # Look for common VisDrone directory structures
    for root, dirs, files in os.walk(source_path):
        if 'images' in dirs or any('image' in d.lower() for d in dirs):
            images_dir = next((os.path.join(root, d) for d in dirs if 'image' in d.lower()), None)
        if 'annotations' in dirs or any('annotation' in d.lower() for d in dirs):
            annotations_dir = next((os.path.join(root, d) for d in dirs if 'annotation' in d.lower()), None)
        
        # If no subdirectories, check if this directory contains images directly
        if not images_dir and any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
            images_dir = root
    
    if not images_dir:
        images_dir = source_path  # Fallback to source path
    
    print(f"📁 Images directory: {images_dir}")
    if annotations_dir:
        print(f"📝 Annotations directory: {annotations_dir}")
    
    # Count files for progress tracking
    total_files = 0
    for root, dirs, files in os.walk(images_dir):
        total_files += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
    
    print(f"📊 Found {total_files} image files to import")
    
    # Copy dataset files
    copied_files = 0
    training_samples = []
    
    try:
        for root, dirs, files in os.walk(images_dir):
            # Get relative path from images directory
            rel_path = os.path.relpath(root, images_dir)
            if rel_path == '.':
                rel_path = ''
            
            # Create corresponding directory in target
            target_subdir = os.path.join(target_dir, 'images', rel_path)
            os.makedirs(target_subdir, exist_ok=True)
            
            # Copy image files
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    source_file = os.path.join(root, file)
                    target_file = os.path.join(target_subdir, file)
                    
                    # Copy file
                    shutil.copy2(source_file, target_file)
                    copied_files += 1
                    
                    # Look for corresponding annotation file
                    annotation_path = None
                    if annotations_dir:
                        # Common annotation file extensions for VisDrone
                        base_name = os.path.splitext(file)[0]
                        for ext in ['.txt', '.xml', '.json']:
                            potential_annotation = os.path.join(annotations_dir, rel_path, base_name + ext)
                            if os.path.exists(potential_annotation):
                                # Copy annotation file
                                target_annotation = os.path.join(target_dir, 'annotations', rel_path, base_name + ext)
                                os.makedirs(os.path.dirname(target_annotation), exist_ok=True)
                                shutil.copy2(potential_annotation, target_annotation)
                                annotation_path = target_annotation
                                break
                    
                    # Determine category based on directory structure or filename
                    category = "drone_surveillance"
                    if "train" in rel_path.lower():
                        category = "training"
                    elif "val" in rel_path.lower() or "validation" in rel_path.lower():
                        category = "validation"
                    elif "test" in rel_path.lower():
                        category = "testing"
                    
                    # Add to training samples list
                    training_samples.append({
                        'image_path': target_file,
                        'annotation_path': annotation_path,
                        'label': category,
                        'category': category
                    })
                    
                    # Progress update
                    if copied_files % 100 == 0:
                        print(f"📋 Copied {copied_files}/{total_files} files...")
        
        print(f"✅ Successfully copied {copied_files} files")
        
        # Add dataset to database
        conn = get_db_connection()
        
        # Check if dataset already exists
        existing = conn.execute('SELECT * FROM datasets WHERE dataset_name = ?', (target_name,)).fetchone()
        
        if existing:
            print(f"⚠️  Dataset '{target_name}' already exists. Updating...")
            dataset_id = existing['dataset_id']
            
            # Update dataset
            conn.execute('''
                UPDATE datasets 
                SET total_samples = ?, file_path = ?, description = ?
                WHERE dataset_id = ?
            ''', (len(training_samples), target_dir, 
                  "VisDrone Dataset for drone-based object detection, tracking and surveillance", 
                  dataset_id))
            
            # Clear existing training data
            conn.execute('DELETE FROM training_data WHERE dataset_id = ?', (dataset_id,))
        else:
            print(f"➕ Creating new dataset '{target_name}'")
            
            # Insert new dataset
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO datasets (dataset_name, dataset_type, description, total_samples, file_path, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (target_name, 'detection', 
                  "VisDrone Dataset for drone-based object detection, tracking and surveillance",
                  len(training_samples), target_dir, 1))
            
            dataset_id = cursor.lastrowid
        
        # Insert training data samples
        print("📊 Adding training samples to database...")
        for i, sample in enumerate(training_samples):
            conn.execute('''
                INSERT INTO training_data (dataset_id, image_path, annotation_path, label, category, is_validated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (dataset_id, sample['image_path'], sample['annotation_path'], 
                  sample['label'], sample['category'], 1))
            
            if (i + 1) % 500 == 0:
                print(f"💾 Added {i + 1}/{len(training_samples)} training samples...")
        
        conn.commit()
        conn.close()
        
        print(f"🎉 Successfully imported VisDrone Dataset!")
        print(f"📈 Dataset Statistics:")
        print(f"   - Total samples: {len(training_samples)}")
        print(f"   - Storage path: {target_dir}")
        print(f"   - Database ID: {dataset_id}")
        print(f"   - Categories: {len(set(s['category'] for s in training_samples))}")
        print(f"   - Annotations: {len([s for s in training_samples if s['annotation_path']])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during import: {str(e)}")
        return False
    """Import LOL (Low-Light) Dataset into the surveillance system"""
    
    print(f"🔄 Importing LOL Dataset from: {source_path}")
    
    # Validate source path
    if not os.path.exists(source_path):
        print(f"❌ Error: Source path does not exist: {source_path}")
        return False
    
    # Create target directory in datasets folder
    target_dir = f"datasets/lol_dataset"
    os.makedirs(target_dir, exist_ok=True)
    
    # Count files for progress tracking
    total_files = 0
    for root, dirs, files in os.walk(source_path):
        total_files += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
    
    print(f"📁 Found {total_files} image files to import")
    
    # Copy dataset files
    copied_files = 0
    training_samples = []
    
    try:
        for root, dirs, files in os.walk(source_path):
            # Get relative path from source
            rel_path = os.path.relpath(root, source_path)
            if rel_path == '.':
                rel_path = ''
            
            # Create corresponding directory in target
            target_subdir = os.path.join(target_dir, rel_path)
            os.makedirs(target_subdir, exist_ok=True)
            
            # Copy image files
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    source_file = os.path.join(root, file)
                    target_file = os.path.join(target_subdir, file)
                    
                    # Copy file
                    shutil.copy2(source_file, target_file)
                    copied_files += 1
                    
                    # Determine category based on directory structure
                    category = "low_light"
                    if "high" in rel_path.lower() or "normal" in rel_path.lower():
                        category = "normal_light"
                    elif "low" in rel_path.lower() or "dark" in rel_path.lower():
                        category = "low_light"
                    
                    # Add to training samples list
                    training_samples.append({
                        'image_path': target_file,
                        'label': category,
                        'category': category,
                        'annotation_path': None
                    })
                    
                    # Progress update
                    if copied_files % 100 == 0:
                        print(f"📋 Copied {copied_files}/{total_files} files...")
        
        print(f"✅ Successfully copied {copied_files} files")
        
        # Add dataset to database
        conn = get_db_connection()
        
        # Check if dataset already exists
        existing = conn.execute('SELECT * FROM datasets WHERE dataset_name = ?', (target_name,)).fetchone()
        
        if existing:
            print(f"⚠️  Dataset '{target_name}' already exists. Updating...")
            dataset_id = existing['dataset_id']
            
            # Update dataset
            conn.execute('''
                UPDATE datasets 
                SET total_samples = ?, file_path = ?, description = ?
                WHERE dataset_id = ?
            ''', (len(training_samples), target_dir, 
                  "LOL Dataset for low-light image enhancement and object detection training", 
                  dataset_id))
            
            # Clear existing training data
            conn.execute('DELETE FROM training_data WHERE dataset_id = ?', (dataset_id,))
        else:
            print(f"➕ Creating new dataset '{target_name}'")
            
            # Insert new dataset
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO datasets (dataset_name, dataset_type, description, total_samples, file_path, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (target_name, 'enhancement', 
                  "LOL Dataset for low-light image enhancement and object detection training",
                  len(training_samples), target_dir, 1))
            
            dataset_id = cursor.lastrowid
        
        # Insert training data samples
        print("📊 Adding training samples to database...")
        for i, sample in enumerate(training_samples):
            conn.execute('''
                INSERT INTO training_data (dataset_id, image_path, annotation_path, label, category, is_validated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (dataset_id, sample['image_path'], sample['annotation_path'], 
                  sample['label'], sample['category'], 1))
            
            if (i + 1) % 500 == 0:
                print(f"💾 Added {i + 1}/{len(training_samples)} training samples...")
        
        conn.commit()
        conn.close()
        
        print(f"🎉 Successfully imported LOL Dataset!")
        print(f"📈 Dataset Statistics:")
        print(f"   - Total samples: {len(training_samples)}")
        print(f"   - Storage path: {target_dir}")
        print(f"   - Database ID: {dataset_id}")
        print(f"   - Categories: {len(set(s['category'] for s in training_samples))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during import: {str(e)}")
        return False

def list_datasets():
    """List all datasets in the system"""
    conn = get_db_connection()
    datasets = conn.execute('SELECT * FROM datasets ORDER BY created_date DESC').fetchall()
    
    print("\n📋 Current Datasets:")
    print("-" * 80)
    for ds in datasets:
        print(f"ID: {ds['dataset_id']} | {ds['dataset_name']} ({ds['dataset_type']})")
        print(f"   📁 {ds['file_path']} | 📊 {ds['total_samples']} samples")
        print(f"   📝 {ds['description']}")
        print("-" * 80)
    
    conn.close()

def main():
    """Main function"""
    print("🔧 LOL Dataset Import Utility")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage: python import_lol_dataset.py <source_path> [dataset_name]")
        print("\nExample:")
        print('python import_lol_dataset.py "C:\\Users\\HP\\Downloads\\LOL Dataset\\lol_dataset" "LOL Enhancement Dataset"')
        print("\nCurrent datasets:")
        list_datasets()
        return
    
    source_path = sys.argv[1]
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else "LOL Dataset"
    
    # Import the dataset
    success = import_lol_dataset(source_path, dataset_name)
    
    if success:
        print(f"\n✅ Import completed successfully!")
        print(f"🌐 You can now view the dataset at: http://127.0.0.1:5000/datasets")
        print(f"📊 Analytics available at: http://127.0.0.1:5000/dataset_analytics")
        
        # Show updated dataset list
        print("\n" + "=" * 50)
        list_datasets()
    else:
        print(f"\n❌ Import failed!")

if __name__ == '__main__':
    main()