#!/usr/bin/env python3
"""
Test script to verify SQLite database operations for the Night Surveillance System
"""

import sqlite3
import sys

DATABASE = 'night_surveillance.db'

def test_database_operations():
    """Test all database operations used in the application"""
    
    print("Testing SQLite database operations...")
    print("=" * 50)
    
    # Check if database exists
    try:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        print("✓ Successfully connected to database")
    except Exception as e:
        print(f"✗ Failed to connect to database: {e}")
        return False
    
    # Test user operations (signup simulation)
    print("\n1. Testing user signup operation...")
    try:
        # Check if user exists
        existing_user = conn.execute('SELECT * FROM user WHERE email = ?', ('test@example.com',)).fetchone()
        if existing_user:
            print("  - User already exists, removing for test...")
            conn.execute('DELETE FROM user WHERE email = ?', ('test@example.com',))
            conn.commit()
        
        # Insert new user
        conn.execute('INSERT INTO user (firstname, lastname, email, password) VALUES (?, ?, ?, ?)', 
                    ('Test', 'User', 'test@example.com', 'testpass123'))
        conn.commit()
        print("  ✓ User signup operation successful")
        
        # Verify user was inserted
        user = conn.execute('SELECT * FROM user WHERE email = ?', ('test@example.com',)).fetchone()
        if user:
            print(f"  ✓ User verified: {user['firstname']} {user['lastname']} ({user['email']})")
        else:
            print("  ✗ User verification failed")
            
    except Exception as e:
        print(f"  ✗ User signup operation failed: {e}")
        return False
    
    # Test user login (authentication simulation)
    print("\n2. Testing user login operation...")
    try:
        user = conn.execute('SELECT * FROM user WHERE email = ? AND password = ?', 
                           ('test@example.com', 'testpass123')).fetchone()
        if user:
            print(f"  ✓ Login successful for user: {user['firstname']} {user['lastname']}")
        else:
            print("  ✗ Login failed - user not found")
            
    except Exception as e:
        print(f"  ✗ Login operation failed: {e}")
        return False
    
    # Test camera operations
    print("\n3. Testing camera operations...")
    try:
        # Check if camera exists
        existing_cam = conn.execute('SELECT * FROM cam WHERE camname = ?', ('TestCamera',)).fetchone()
        if existing_cam:
            print("  - Camera already exists, removing for test...")
            conn.execute('DELETE FROM cam WHERE camname = ?', ('TestCamera',))
            conn.commit()
        
        # Insert new camera
        conn.execute('INSERT INTO cam (camname, camurl, camfps) VALUES (?, ?, ?)', 
                    ('TestCamera', 'http://192.168.1.100:8080/video', '25'))
        conn.commit()
        print("  ✓ Camera addition operation successful")
        
        # Verify camera was inserted
        camera = conn.execute('SELECT * FROM cam WHERE camname = ?', ('TestCamera',)).fetchone()
        if camera:
            print(f"  ✓ Camera verified: {camera['camname']} ({camera['camurl']}) @ {camera['camfps']} fps")
        else:
            print("  ✗ Camera verification failed")
            
    except Exception as e:
        print(f"  ✗ Camera operation failed: {e}")
        return False
    
    # Show final database state
    print("\n4. Final database state:")
    try:
        users = conn.execute('SELECT * FROM user').fetchall()
        cameras = conn.execute('SELECT * FROM cam').fetchall()
        
        print(f"  Users ({len(users)}):")
        for user in users:
            print(f"    - {user['firstname']} {user['lastname']} ({user['email']})")
            
        print(f"  Cameras ({len(cameras)}):")
        for cam in cameras:
            print(f"    - {cam['camname']}: {cam['camurl']} @ {cam['camfps']} fps")
            
    except Exception as e:
        print(f"  ✗ Failed to retrieve final state: {e}")
        return False
    
    conn.close()
    print("\n" + "=" * 50)
    print("✓ All database operations completed successfully!")
    print("✓ Your Night Surveillance System is ready to use with SQLite!")
    
    return True

def main():
    """Main test function"""
    print("Night Surveillance System - Database Test")
    print("=" * 50)
    
    if test_database_operations():
        print("\n🎉 Database test PASSED! Your local SQLite setup is working correctly.")
        print("\nYou can now:")
        print("1. Start the application: python main.py")
        print("2. Open your browser to: http://127.0.0.1:5000")
        print("3. Create new accounts or use the test account:")
        print("   Email: admin@nightsurveillance.com")
        print("   Password: admin123")
    else:
        print("\n❌ Database test FAILED! Please check the errors above.")
        sys.exit(1)

if __name__ == '__main__':
    main()