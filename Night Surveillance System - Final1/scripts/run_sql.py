#!/usr/bin/env python3
"""Run an ad‑hoc SQL query against Supabase Postgres (or local SQLite fallback).

Usage (PowerShell):
  # Ensure env var is set for Supabase pooled connection
  # $env:SUPABASE_DB_URL = "postgresql://..."
  python scripts/run_sql.py "SELECT sno, firstname, lastname, email FROM \"user\" ORDER BY sno DESC LIMIT 5;"

If SUPABASE_DB_URL is not set the script will read from local SQLite file night_surveillance.db.
"""
import os, sys, sqlite3

def main():
    if len(sys.argv) < 2:
        print("Provide a SQL query as a single argument.")
        print("Example: python run_sql.py \"SELECT COUNT(*) FROM \"user\";\"")
        sys.exit(1)
    sql = sys.argv[1]
    pg_url = os.getenv('SUPABASE_DB_URL') or os.getenv('DATABASE_URL')
    if pg_url:
        # Postgres path
        import psycopg2, psycopg2.extras
        if 'sslmode=' not in pg_url:
            pg_url += ('&' if '?' in pg_url else '?') + 'sslmode=require'
        conn = psycopg2.connect(pg_url, cursor_factory=psycopg2.extras.RealDictCursor)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall() if cur.description else []
        print(f"Rows: {len(rows)}")
        for r in rows:
            print(r)
        conn.close()
    else:
        # SQLite path
        db = 'night_surveillance.db'
        if not os.path.exists(db):
            print(f"SQLite file not found: {db}")
            sys.exit(2)
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall() if cur.description else []
        print(f"Rows: {len(rows)}")
        for r in rows:
            print(dict(r))
        conn.close()

if __name__ == '__main__':
    main()