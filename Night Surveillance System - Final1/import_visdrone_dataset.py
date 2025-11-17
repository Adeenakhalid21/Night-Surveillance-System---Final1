#!/usr/bin/env python3
"""
VisDrone Dataset Import Utility for Night Surveillance System
This script imports the VisDrone dataset for drone-based object detection and tracking.
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
    os.makedirs(f"{target_dir}/images", exist_ok=True)
    os.makedirs(f"{target_dir}/annotations", exist_ok=True)
    
    # VisDrone typical structure: look for images and annotations
    images_dir = None
    annotations_dir = None
    
    # Look for common VisDrone directory structures
    print("🔍 Analyzing VisDrone dataset structure...")
    for root, dirs, files in os.walk(source_path):
        # Check for image directories
        for d in dirs:
            if any(keyword in d.lower() for keyword in ['image', 'img', 'jpg']):
                images_dir = os.path.join(root, d)
                print(f"📁 Found images directory: {d}")
                break
        
        # Check for annotation directories  
        for d in dirs:
            if any(keyword in d.lower() for keyword in ['annotation', 'label', 'gt']):
                annotations_dir = os.path.join(root, d)
                print(f"📝 Found annotations directory: {d}")
                break
        
        # If no subdirectories, check if this directory contains images directly
        if not images_dir:
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if len(image_files) > 10:  # Threshold to consider it an image directory
                images_dir = root
                print(f"📁 Using root as images directory: {len(image_files)} images found")
    
    if not images_dir:
        print("❌ No images directory found. Please check the dataset structure.")
        return False
    
    print(f"📂 Images source: {images_dir}")
    if annotations_dir:
        print(f"📝 Annotations source: {annotations_dir}")
    else:
        print("⚠️  No annotations directory found - importing images only")
    
    # Count files for progress tracking
    total_files = 0
    for root, dirs, files in os.walk(images_dir):
        total_files += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
    
    print(f"📊 Found {total_files} image files to import")
    
    if total_files == 0:
        print("❌ No image files found to import")
        return False
    
    # Copy dataset files
    copied_files = 0
    copied_annotations = 0
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
                            # Try different possible annotation paths
                            potential_paths = [
                                os.path.join(annotations_dir, rel_path, base_name + ext),
                                os.path.join(annotations_dir, base_name + ext),
                                # VisDrone sometimes uses slightly different naming
                                os.path.join(annotations_dir, rel_path, file.replace('.jpg', ext).replace('.png', ext)),
                            ]
                            
                            for potential_annotation in potential_paths:
                                if os.path.exists(potential_annotation):
                                    # Copy annotation file
                                    target_annotation = os.path.join(target_dir, 'annotations', rel_path, base_name + ext)
                                    os.makedirs(os.path.dirname(target_annotation), exist_ok=True)
                                    shutil.copy2(potential_annotation, target_annotation)
                                    annotation_path = target_annotation
                                    copied_annotations += 1
                                    break
                            
                            if annotation_path:
                                break
                    
                    # Determine category based on directory structure or filename
                    category = "drone_surveillance"
                    label = "drone_data"
                    
                    if "train" in rel_path.lower() or "train" in file.lower():
                        category = "training"
                        label = "training_data"
                    elif "val" in rel_path.lower() or "validation" in rel_path.lower():
                        category = "validation" 
                        label = "validation_data"
                    elif "test" in rel_path.lower():
                        category = "testing"
                        label = "test_data"
                    elif any(keyword in rel_path.lower() for keyword in ['sequence', 'seq']):
                        category = "sequence_data"
                        label = "video_sequence"
                    
                    # Add to training samples list
                    training_samples.append({
                        'image_path': target_file,
                        'annotation_path': annotation_path,
                        'label': label,
                        'category': category
                    })
                    
                    # Progress update
                    if copied_files % 200 == 0:
                        print(f"📋 Copied {copied_files}/{total_files} files... ({copied_annotations} annotations)")
        
        print(f"✅ Successfully copied {copied_files} images and {copied_annotations} annotations")
        
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
                  "VisDrone Dataset for drone-based object detection, tracking and aerial surveillance", 
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
                  "VisDrone Dataset for drone-based object detection, tracking and aerial surveillance",
                  len(training_samples), target_dir, 1))
            
            dataset_id = cursor.lastrowid
        
        # Insert training data samples
        print("📊 Adding training samples to database...")
        batch_size = 1000
        for i in range(0, len(training_samples), batch_size):
            batch = training_samples[i:i + batch_size]
            for sample in batch:
                conn.execute('''
                    INSERT INTO training_data (dataset_id, image_path, annotation_path, label, category, is_validated)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (dataset_id, sample['image_path'], sample['annotation_path'], 
                      sample['label'], sample['category'], 1))
            
            conn.commit()
            print(f"💾 Added {min(i + batch_size, len(training_samples))}/{len(training_samples)} training samples...")
        
        conn.close()
        
        # Analysis
        categories = list(set(s['category'] for s in training_samples))
        annotations_count = len([s for s in training_samples if s['annotation_path']])
        
        print(f"🎉 Successfully imported VisDrone Dataset!")
        print(f"📈 Dataset Statistics:")
        print(f"   - Total images: {len(training_samples)}")
        print(f"   - Total annotations: {annotations_count}")
        print(f"   - Storage path: {target_dir}")
        print(f"   - Database ID: {dataset_id}")
        print(f"   - Categories: {', '.join(categories)}")
        print(f"   - Annotation coverage: {annotations_count/len(training_samples)*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during import: {str(e)}")
        import traceback
        traceback.print_exc()
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
    print("🚁 VisDrone Dataset Import Utility")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage: python import_visdrone_dataset.py <source_path> [dataset_name]")
        print("\nExample:")
        print('python import_visdrone_dataset.py "C:\\Users\\HP\\Downloads\\VisDrone" "VisDrone Surveillance Dataset"')
        print("\nCurrent datasets:")
        list_datasets()
        return
    
    source_path = sys.argv[1]
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else "VisDrone Dataset"
    
    # Import the dataset
    success = import_visdrone_dataset(source_path, dataset_name)
    
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