#!/usr/bin/env python3
"""
Quick verification that SQLite backend persists data.
Inserts a test surveillance event and a test detection result, reads them back,
then cleans up the inserted rows.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = 'night_surveillance.db'


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def main():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}. Start the app once to initialize or run db_manager.py.")
        return 1

    conn = get_conn()
    cur = conn.cursor()

    # Record baseline counts
    base_events = cur.execute('SELECT COUNT(*) AS c FROM surveillance_events').fetchone()['c']
    base_results = cur.execute('SELECT COUNT(*) AS c FROM detection_results').fetchone()['c']

    # Insert test event
    ts = datetime.now().isoformat(timespec='seconds')
    cur.execute('''
        INSERT INTO surveillance_events (camera_id, event_type, severity, description, image_path, video_path)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (None, 'verification_test', 'low', f'Verification event at {ts}', None, None))
    event_id = cur.lastrowid

    # Find any dataset_id if exists to attach detection; else leave NULL
    ds = cur.execute('SELECT dataset_id FROM datasets ORDER BY dataset_id LIMIT 1').fetchone()
    dataset_id = ds['dataset_id'] if ds else None
    cur.execute('''
        INSERT INTO detection_results (camera_id, object_class, confidence, bbox_x, bbox_y, bbox_width, bbox_height, image_path, dataset_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (None, 'verification_object', 0.99, 0, 0, 10, 10, None, dataset_id))
    result_id = cur.lastrowid

    conn.commit()

    # Verify counts increased
    new_events = cur.execute('SELECT COUNT(*) AS c FROM surveillance_events').fetchone()['c']
    new_results = cur.execute('SELECT COUNT(*) AS c FROM detection_results').fetchone()['c']

    print('✅ Backend storage verification')
    print(f'- surveillance_events: {base_events} -> {new_events} (+{new_events - base_events})')
    print(f'- detection_results: {base_results} -> {new_results} (+{new_results - base_results})')
    print(f'- inserted event_id={event_id}, result_id={result_id}, dataset_id={dataset_id}')

    # Cleanup: remove inserted rows
    cur.execute('DELETE FROM surveillance_events WHERE event_id = ?', (event_id,))
    cur.execute('DELETE FROM detection_results WHERE result_id = ?', (result_id,))
    conn.commit(); conn.close()
    print('🧹 Cleanup complete (test rows removed).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
