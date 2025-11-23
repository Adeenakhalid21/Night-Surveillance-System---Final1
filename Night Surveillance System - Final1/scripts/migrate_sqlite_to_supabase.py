#!/usr/bin/env python3
"""
Migrate local SQLite data (night_surveillance.db) to a Supabase Postgres database.
- Creates tables in Postgres if they don't exist (quoted names to match current app schema).
- Copies rows from SQLite, preserving primary keys.
- Advances Postgres sequences to max(id) to avoid future conflicts.

Usage (PowerShell):
  $env:SUPABASE_DB_URL="postgresql://USER:PASS@HOST:5432/postgres?sslmode=require"
  python "Night Surveillance System - Final1/scripts/migrate_sqlite_to_supabase.py"

Tip: Get the connection string from Supabase > Project Settings > Database > Connection string > psql.
"""
import os
import sqlite3
import psycopg2
import psycopg2.extras
from urllib.parse import urlparse, parse_qs
BOOL_COLUMNS = {
    'datasets': {'is_active'},
    'training_data': {'is_validated'},
}

from textwrap import dedent

SQLITE_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'night_surveillance.db')
DROP_DETECTIONS = str(os.getenv('DROP_DETECTIONS', '1')).strip().lower() in ('1','true','yes','on')

SCHEMA_SQL = dedent(
    '''
    CREATE TABLE IF NOT EXISTS "user" (
        sno SERIAL PRIMARY KEY,
        firstname TEXT NOT NULL,
        lastname TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS cam (
        id SERIAL PRIMARY KEY,
        camname TEXT UNIQUE NOT NULL,
        camurl TEXT NOT NULL,
        camfps TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS datasets (
        dataset_id SERIAL PRIMARY KEY,
        dataset_name TEXT UNIQUE NOT NULL,
        dataset_type TEXT NOT NULL,
        description TEXT,
        created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        total_samples INTEGER DEFAULT 0,
        file_path TEXT,
        is_active BOOLEAN DEFAULT TRUE
    );

    CREATE TABLE IF NOT EXISTS training_data (
        sample_id SERIAL PRIMARY KEY,
        dataset_id INTEGER NOT NULL,
        image_path TEXT NOT NULL,
        annotation_path TEXT,
        label TEXT,
        category TEXT,
        is_validated BOOLEAN DEFAULT FALSE,
        added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
    );

    CREATE TABLE IF NOT EXISTS surveillance_events (
        event_id SERIAL PRIMARY KEY,
        camera_id INTEGER,
        event_type TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        severity TEXT DEFAULT 'medium',
        description TEXT,
        image_path TEXT,
        video_path TEXT,
        is_resolved BOOLEAN DEFAULT FALSE,
        FOREIGN KEY (camera_id) REFERENCES cam (id)
    );
    '''
)

SETVAL_SQL = [
    "SELECT setval(pg_get_serial_sequence('\"user\"','sno'), COALESCE((SELECT max(sno) FROM \"user\"), 1), true);",
    "SELECT setval(pg_get_serial_sequence('cam','id'), COALESCE((SELECT max(id) FROM cam), 1), true);",
    "SELECT setval(pg_get_serial_sequence('datasets','dataset_id'), COALESCE((SELECT max(dataset_id) FROM datasets), 1), true);",
    "SELECT setval(pg_get_serial_sequence('training_data','sample_id'), COALESCE((SELECT max(sample_id) FROM training_data), 1), true);",
    "SELECT setval(pg_get_serial_sequence('surveillance_events','event_id'), COALESCE((SELECT max(event_id) FROM surveillance_events), 1), true);",
]


def open_sqlite():
    if not os.path.exists(SQLITE_DB):
        raise SystemExit(f"SQLite file not found: {SQLITE_DB}")
    conn = sqlite3.connect(SQLITE_DB)
    conn.row_factory = sqlite3.Row
    return conn


def open_pg():
    dsn = os.getenv('SUPABASE_DB_URL') or os.getenv('DATABASE_URL')
    if not dsn:
        raise SystemExit('Set SUPABASE_DB_URL (or DATABASE_URL) to your Supabase Postgres connection string')
    # Ensure sslmode=require unless already present
    if 'sslmode=' not in dsn:
        joiner = '&' if '?' in dsn else '?'
        dsn = f"{dsn}{joiner}sslmode=require"
    try:
        return psycopg2.connect(dsn)
    except psycopg2.OperationalError as e:
        try:
            u = urlparse(dsn)
            host = u.hostname
            port = u.port
            print(f"Connection failed to host={host} port={port}.")
        except Exception:
            pass
        print("Hint: Verify internet/DNS, and try the pooled connection host from Supabase (Connection Pool).\n"
              "Example: postgresql://USER:PASSWORD@aws-0-<region>.pooler.supabase.com:6543/postgres?sslmode=require\n"
              "Also ensure your password is URL-encoded if it contains special characters.")
        raise


def upsert_table(cur_pg, table, cols, rows):
    # Coerce boolean-like integer fields (e.g., 0/1) to actual booleans for Postgres
    if rows:
        explicit = BOOL_COLUMNS.get(table, set())
        inferred = {c for c in cols if c.startswith('is_') or c.endswith('_flag')}
        bool_cols = explicit.union(inferred)
        if bool_cols:
            idx_map = {c: i for i, c in enumerate(cols) if c in bool_cols}
            coerced_rows = []
            for row in rows:
                row_list = list(row)
                for c, idx in idx_map.items():
                    v = row_list[idx]
                    if isinstance(v, bool) or v is None:
                        continue
                    s = str(v).strip().lower()
                    row_list[idx] = s in ('1', 'true', 't', 'yes', 'y')
                coerced_rows.append(tuple(row_list))
            rows = coerced_rows
    if not rows:
        return 0
    cols_list = ','.join(cols)
    placeholders = ','.join(['%s'] * len(cols))
    # Build ON CONFLICT on primary key column (assumed first col is the PK name ending with _id or id or sno/event_id, etc.)
    pk = cols[0]
    conflict_target = pk
    updates = ','.join([f"{c}=EXCLUDED.{c}" for c in cols[1:]])
    sql = f"INSERT INTO {table} ({cols_list}) VALUES ({placeholders}) ON CONFLICT ({conflict_target}) DO UPDATE SET {updates};"
    psycopg2.extras.execute_batch(cur_pg, sql, rows, page_size=1000)
    return len(rows)


def migrate():
    con_sqlite = open_sqlite()
    con_pg = open_pg()
    con_pg.autocommit = False

    with con_pg.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        con_pg.commit()
        if DROP_DETECTIONS:
            # Remove detection_results table to save space in Supabase
            try:
                cur.execute('DROP TABLE IF EXISTS detection_results;')
                con_pg.commit()
                print('Dropped table detection_results (DROP_DETECTIONS=1).')
            except Exception as e:
                print(f"Warning: failed to drop detection_results: {e}")

    with con_pg.cursor() as cur_pg:
        cur_pg.execute('SET CONSTRAINTS ALL DEFERRED;')

        cur_sql = con_sqlite.cursor()
        # user (reserved keyword, quoted)
        users = cur_sql.execute('SELECT sno, firstname, lastname, email, password FROM user').fetchall()
        upsert_table(cur_pg, '"user"', ['sno','firstname','lastname','email','password'], [tuple(u) for u in users])

        cams = cur_sql.execute('SELECT id, camname, camurl, camfps FROM cam').fetchall()
        upsert_table(cur_pg, 'cam', ['id','camname','camurl','camfps'], [tuple(c) for c in cams])

        datasets = cur_sql.execute('''SELECT dataset_id, dataset_name, dataset_type, description, created_date, total_samples, file_path, is_active FROM datasets''').fetchall()
        upsert_table(cur_pg, 'datasets', ['dataset_id','dataset_name','dataset_type','description','created_date','total_samples','file_path','is_active'], [tuple(d) for d in datasets])

        # detection_results intentionally skipped/removed to reduce storage

        train = cur_sql.execute('''SELECT sample_id, dataset_id, image_path, annotation_path, label, category, is_validated, added_date FROM training_data''').fetchall()
        upsert_table(cur_pg, 'training_data', ['sample_id','dataset_id','image_path','annotation_path','label','category','is_validated','added_date'], [tuple(t) for t in train])

        events = cur_sql.execute('''SELECT event_id, camera_id, event_type, timestamp, severity, description, image_path, video_path, is_resolved FROM surveillance_events''').fetchall()
        upsert_table(cur_pg, 'surveillance_events', ['event_id','camera_id','event_type','timestamp','severity','description','image_path','video_path','is_resolved'], [tuple(e) for e in events])

        # Advance sequences
        for q in SETVAL_SQL:
            cur_pg.execute(q)

    con_pg.commit()
    con_pg.close()
    con_sqlite.close()
    print('Migration completed successfully.')


if __name__ == '__main__':
    migrate()
