#!/usr/bin/env python3
"""
Database initialization and management script for Night Surveillance System
This script creates and manages the SQLite database for the application.
"""

import sqlite3
import os

DATABASE = 'night_surveillance.db'

def create_database():
    """Create the SQLite database and tables"""
    # Remove existing database if it exists
    if os.path.exists(DATABASE):
        print(f"Removing existing database: {DATABASE}")
        os.remove(DATABASE)
    
    print(f"Creating new database: {DATABASE}")
    
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    
    # Create users table
    conn.execute('''
        CREATE TABLE user (
            sno INTEGER PRIMARY KEY AUTOINCREMENT,
            firstname TEXT NOT NULL,
            lastname TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    
    # Create cameras table
    conn.execute('''
        CREATE TABLE cam (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camname TEXT UNIQUE NOT NULL,
            camurl TEXT NOT NULL,
            camfps TEXT NOT NULL
        )
    ''')
    
    # Create datasets table for managing surveillance datasets
    conn.execute('''
        CREATE TABLE datasets (
            dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT UNIQUE NOT NULL,
            dataset_type TEXT NOT NULL,
            description TEXT,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_samples INTEGER DEFAULT 0,
            file_path TEXT,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # Create detection_results table for storing AI detection results
    conn.execute('''
        CREATE TABLE detection_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            object_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            bbox_x REAL,
            bbox_y REAL,
            bbox_width REAL,
            bbox_height REAL,
            image_path TEXT,
            dataset_id INTEGER,
            FOREIGN KEY (camera_id) REFERENCES cam (id),
            FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
        )
    ''')
    
    # Create training_data table for machine learning datasets
    conn.execute('''
        CREATE TABLE training_data (
            sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            annotation_path TEXT,
            label TEXT,
            category TEXT,
            is_validated BOOLEAN DEFAULT 0,
            added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
        )
    ''')
    
    # Create surveillance_events table for logging security events
    conn.execute('''
        CREATE TABLE surveillance_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER,
            event_type TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            severity TEXT DEFAULT 'medium',
            description TEXT,
            image_path TEXT,
            video_path TEXT,
            is_resolved BOOLEAN DEFAULT 0,
            FOREIGN KEY (camera_id) REFERENCES cam (id)
        )
    ''')
    
    # Insert some sample data for testing
    print("Adding sample data...")
    
    # Sample user
    conn.execute('''
        INSERT INTO user (firstname, lastname, email, password) 
        VALUES (?, ?, ?, ?)
    ''', ('Admin', 'User', 'admin@nightsurveillance.com', 'admin123'))
    
    # Sample camera
    conn.execute('''
        INSERT INTO cam (camname, camurl, camfps) 
        VALUES (?, ?, ?)
    ''', ('DefaultCam', '0', '30'))
    
    # Sample datasets
    conn.execute('''
        INSERT INTO datasets (dataset_name, dataset_type, description, total_samples, file_path) 
        VALUES (?, ?, ?, ?, ?)
    ''', ('Person Detection Dataset', 'detection', 'Dataset for training person detection models', 0, 'datasets/person_detection/'))
    
    conn.execute('''
        INSERT INTO datasets (dataset_name, dataset_type, description, total_samples, file_path) 
        VALUES (?, ?, ?, ?, ?)
    ''', ('Vehicle Classification', 'classification', 'Dataset for classifying different vehicle types', 0, 'datasets/vehicle_classification/'))
    
    conn.execute('''
        INSERT INTO datasets (dataset_name, dataset_type, description, total_samples, file_path) 
        VALUES (?, ?, ?, ?, ?)
    ''', ('Security Events Log', 'events', 'Historical surveillance events and alerts', 0, 'datasets/events/'))
    
    conn.commit()
    print("Database created successfully!")
    
    # Display created tables and sample data
    print("\n=== Database Tables ===")
    
    print("\nUsers table:")
    users = conn.execute('SELECT * FROM user').fetchall()
    for user in users:
        print(f"  ID: {user['sno']}, Name: {user['firstname']} {user['lastname']}, Email: {user['email']}")
    
    print("\nCameras table:")
    cams = conn.execute('SELECT * FROM cam').fetchall()
    for cam in cams:
        print(f"  ID: {cam['id']}, Name: {cam['camname']}, URL: {cam['camurl']}, FPS: {cam['camfps']}")
    
    print("\nDatasets table:")
    datasets = conn.execute('SELECT * FROM datasets').fetchall()
    for dataset in datasets:
        print(f"  ID: {dataset['dataset_id']}, Name: {dataset['dataset_name']}, Type: {dataset['dataset_type']}, Samples: {dataset['total_samples']}")
    
    print(f"\nAdditional tables created:")
    print("  - detection_results (for AI detection data)")
    print("  - training_data (for ML training samples)")
    print("  - surveillance_events (for security event logging)")
    
    conn.close()

def check_database():
    """Check if database exists and show its contents"""
    if not os.path.exists(DATABASE):
        print(f"Database {DATABASE} does not exist!")
        return False
    
    print(f"Database {DATABASE} exists. Contents:")
    
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    
    # Check users
    users = conn.execute('SELECT * FROM user').fetchall()
    print(f"\nUsers table ({len(users)} records):")
    for user in users:
        print(f"  ID: {user['sno']}, Name: {user['firstname']} {user['lastname']}, Email: {user['email']}")
    
    # Check cameras
    cams = conn.execute('SELECT * FROM cam').fetchall()
    print(f"\nCameras table ({len(cams)} records):")
    for cam in cams:
        print(f"  ID: {cam['id']}, Name: {cam['camname']}, URL: {cam['camurl']}, FPS: {cam['camfps']}")
    
    # Check datasets
    try:
        datasets = conn.execute('SELECT * FROM datasets').fetchall()
        print(f"\nDatasets table ({len(datasets)} records):")
        for dataset in datasets:
            print(f"  ID: {dataset['dataset_id']}, Name: {dataset['dataset_name']}, Type: {dataset['dataset_type']}, Samples: {dataset['total_samples']}")
        
        # Check detection results
        results = conn.execute('SELECT COUNT(*) as count FROM detection_results').fetchone()
        print(f"\nDetection Results: {results['count']} records")
        
        # Check training data
        training = conn.execute('SELECT COUNT(*) as count FROM training_data').fetchone()
        print(f"Training Data: {training['count']} records")
        
        # Check surveillance events
        events = conn.execute('SELECT COUNT(*) as count FROM surveillance_events').fetchone()
        print(f"Surveillance Events: {events['count']} records")
        
    except Exception as e:
        print(f"\nDataset tables not found - using legacy database schema: {e}")
    
    conn.close()
    return True

if __name__ == '__main__':
    print("Night Surveillance System - Database Manager")
    print("=" * 50)
    
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--check':
            check_database()
        elif sys.argv[1] == '--create':
            create_database()
        else:
            print("Usage: python db_manager.py [--create | --check]")
            print("  --create: Create a new database (removes existing)")
            print("  --check:  Check existing database contents")
    else:
        # Default action: create database
        create_database()