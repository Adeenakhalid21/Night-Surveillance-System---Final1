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
    #!/usr/bin/env python3
    """Dataset Import Utility for Night Surveillance System.

    Functions:
      import_visdrone_dataset(source_path, target_name)
      import_lol_dataset(source_path, target_name)

    Usage examples:
      python import_lol_dataset.py "C:\\path\\to\\LOL" "LOL Enhancement Dataset"
    """

    import os
    import shutil
    import sqlite3
    import sys

    DATABASE = 'night_surveillance.db'

    def get_db_connection():
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        return conn

    def import_visdrone_dataset(source_path, target_name="VisDrone Dataset"):
        print(f"🔄 Importing VisDrone Dataset from: {source_path}")
        if not os.path.exists(source_path):
            print(f"❌ Source path does not exist: {source_path}")
            return False
        target_dir = "datasets/visdrone_dataset"
        os.makedirs(target_dir, exist_ok=True)
        images_dir = None
        annotations_dir = None
        for root, dirs, files in os.walk(source_path):
            if 'images' in dirs or any('image' in d.lower() for d in dirs):
                images_dir = next((os.path.join(root, d) for d in dirs if 'image' in d.lower()), None)
            if 'annotations' in dirs or any('annotation' in d.lower() for d in dirs):
                annotations_dir = next((os.path.join(root, d) for d in dirs if 'annotation' in d.lower()), None)
            if not images_dir and any(f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tiff')) for f in files):
                images_dir = root
        if not images_dir:
            images_dir = source_path
        print(f"📁 Images directory: {images_dir}")
        if annotations_dir:
            print(f"📝 Annotations directory: {annotations_dir}")
        total_files = 0
        for root, dirs, files in os.walk(images_dir):
            total_files += len([f for f in files if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tiff'))])
        print(f"📊 Found {total_files} image files to import")
        copied_files = 0
        training_samples = []
        try:
            for root, dirs, files in os.walk(images_dir):
                rel_path = os.path.relpath(root, images_dir)
                if rel_path == '.':
                    rel_path = ''
                target_subdir = os.path.join(target_dir,'images',rel_path)
                os.makedirs(target_subdir, exist_ok=True)
                for file in files:
                    if file.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tiff')):
                        source_file = os.path.join(root,file)
                        target_file = os.path.join(target_subdir,file)
                        shutil.copy2(source_file,target_file)
                        copied_files += 1
                        annotation_path = None
                        if annotations_dir:
                            base = os.path.splitext(file)[0]
                            for ext in ['.txt','.xml','.json']:
                                ann = os.path.join(annotations_dir,rel_path,base+ext)
                                if os.path.exists(ann):
                                    target_ann = os.path.join(target_dir,'annotations',rel_path,base+ext)
                                    os.makedirs(os.path.dirname(target_ann),exist_ok=True)
                                    shutil.copy2(ann,target_ann)
                                    annotation_path = target_ann
                                    break
                        category = 'drone_surveillance'
                        if 'train' in rel_path.lower(): category = 'training'
                        elif 'val' in rel_path.lower() or 'validation' in rel_path.lower(): category = 'validation'
                        elif 'test' in rel_path.lower(): category = 'testing'
                        training_samples.append({'image_path':target_file,'annotation_path':annotation_path,'label':category,'category':category})
                        if copied_files % 100 == 0:
                            print(f"📋 Copied {copied_files}/{total_files} files...")
            print(f"✅ Successfully copied {copied_files} files")
            conn = get_db_connection()
            existing = conn.execute('SELECT * FROM datasets WHERE dataset_name = ?', (target_name,)).fetchone()
            if existing:
                print(f"⚠️  Dataset '{target_name}' already exists. Updating...")
                dataset_id = existing['dataset_id']
                conn.execute('UPDATE datasets SET total_samples = ?, file_path = ?, description = ? WHERE dataset_id = ?', (len(training_samples), target_dir, 'VisDrone Dataset for drone-based object detection, tracking and surveillance', dataset_id))
                conn.execute('DELETE FROM training_data WHERE dataset_id = ?', (dataset_id,))
            else:
                cursor = conn.cursor()
                cursor.execute('INSERT INTO datasets (dataset_name, dataset_type, description, total_samples, file_path, is_active) VALUES (?, ?, ?, ?, ?, ?)', (target_name,'detection','VisDrone Dataset for drone-based object detection, tracking and surveillance',len(training_samples),target_dir,1))
                dataset_id = cursor.lastrowid
            print('📊 Adding training samples to database...')
            for i,sample in enumerate(training_samples):
                conn.execute('INSERT INTO training_data (dataset_id, image_path, annotation_path, label, category, is_validated) VALUES (?, ?, ?, ?, ?, ?)', (dataset_id,sample['image_path'],sample['annotation_path'],sample['label'],sample['category'],1))
                if (i+1) % 500 == 0:
                    print(f"💾 Added {i+1}/{len(training_samples)} training samples...")
            conn.commit(); conn.close()
            print('🎉 Successfully imported VisDrone Dataset!')
            return True
        except Exception as e:
            print(f"❌ Error during import: {e}")
            return False

    def import_lol_dataset(source_path, target_name="LOL Dataset"):
        print(f"🔄 Importing LOL Dataset from: {source_path}")
        if not os.path.exists(source_path):
            print(f"❌ Source path does not exist: {source_path}")
            return False
        target_dir = "datasets/lol_dataset"
        os.makedirs(target_dir, exist_ok=True)
        total_files = 0
        for root, dirs, files in os.walk(source_path):
            total_files += len([f for f in files if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tiff'))])
        print(f"📁 Found {total_files} image files to import")
        copied_files = 0
        training_samples = []
        try:
            for root, dirs, files in os.walk(source_path):
                rel_path = os.path.relpath(root, source_path)
                if rel_path == '.': rel_path = ''
                target_subdir = os.path.join(target_dir, rel_path)
                os.makedirs(target_subdir, exist_ok=True)
                for file in files:
                    if file.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tiff')):
                        source_file = os.path.join(root,file)
                        target_file = os.path.join(target_subdir,file)
                        shutil.copy2(source_file,target_file)
                        copied_files += 1
                        category = 'low_light'
                        if any(x in rel_path.lower() for x in ['high','normal']): category = 'normal_light'
                        elif any(x in rel_path.lower() for x in ['low','dark']): category = 'low_light'
                        training_samples.append({'image_path':target_file,'label':category,'category':category,'annotation_path':None})
                        if copied_files % 100 == 0:
                            print(f"📋 Copied {copied_files}/{total_files} files...")
            print(f"✅ Successfully copied {copied_files} files")
            conn = get_db_connection()
            existing = conn.execute('SELECT * FROM datasets WHERE dataset_name = ?', (target_name,)).fetchone()
            if existing:
                print(f"⚠️  Dataset '{target_name}' already exists. Updating...")
                dataset_id = existing['dataset_id']
                conn.execute('UPDATE datasets SET total_samples = ?, file_path = ?, description = ? WHERE dataset_id = ?', (len(training_samples),target_dir,'LOL Dataset for low-light image enhancement and object detection training',dataset_id))
                conn.execute('DELETE FROM training_data WHERE dataset_id = ?', (dataset_id,))
            else:
                cursor = conn.cursor()
                cursor.execute('INSERT INTO datasets (dataset_name, dataset_type, description, total_samples, file_path, is_active) VALUES (?, ?, ?, ?, ?, ?)', (target_name,'enhancement','LOL Dataset for low-light image enhancement and object detection training',len(training_samples),target_dir,1))
                dataset_id = cursor.lastrowid
            print('📊 Adding training samples to database...')
            for i,sample in enumerate(training_samples):
                conn.execute('INSERT INTO training_data (dataset_id, image_path, annotation_path, label, category, is_validated) VALUES (?, ?, ?, ?, ?, ?)', (dataset_id,sample['image_path'],sample['annotation_path'],sample['label'],sample['category'],1))
                if (i+1) % 500 == 0:
                    print(f"💾 Added {i+1}/{len(training_samples)} training samples...")
            conn.commit(); conn.close()
            print('🎉 Successfully imported LOL Dataset!')
            return True
        except Exception as e:
            print(f"❌ Error during import: {e}")
            return False

    def list_datasets():
        conn = get_db_connection()
        datasets = conn.execute('SELECT * FROM datasets ORDER BY created_date DESC').fetchall()
        print('\n📋 Current Datasets:')
        print('-'*80)
        for ds in datasets:
            print(f"ID: {ds['dataset_id']} | {ds['dataset_name']} ({ds['dataset_type']})")
            print(f"  📁 {ds['file_path']} | 📊 {ds['total_samples']} samples")
            print(f"  📝 {ds['description']}")
            print('-'*80)
        conn.close()

    def main():
        print('🔧 LOL Dataset Import Utility')
        print('='*50)
        if len(sys.argv) < 2:
            print('Usage: python import_lol_dataset.py <source_path> [dataset_name]')
            print('\nExample:')
            print('python import_lol_dataset.py "C:\\Users\\HP\\Downloads\\LOL Dataset\\lol_dataset" "LOL Enhancement Dataset"')
            print('\nCurrent datasets:')
            list_datasets(); return
        source_path = sys.argv[1]
        dataset_name = sys.argv[2] if len(sys.argv) > 2 else 'LOL Dataset'
        success = import_lol_dataset(source_path, dataset_name)
        if success:
            print('\n✅ Import completed successfully!')
            print('🌐 View datasets: http://127.0.0.1:5000/datasets')
            print('📊 Analytics: http://127.0.0.1:5000/dataset_analytics')
            print('\n'+'='*50)
            list_datasets()
        else:
            print('\n❌ Import failed!')

    if __name__ == '__main__':
        main()
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