#!/usr/bin/env python3
"""
run_etl.py -- orchestrates the full MySQL ETL pipeline end to end.

Runs, in order:
    sql/00_functions.sql
    sql/01_staging_schema.sql
    sql/02_load_staging.sql
    sql/03_warehouse_schema.sql
    sql/04_transform_load.sql
    sql/05_data_quality_checks.sql   (prints results, does not fail the run)

Each step's wall-clock time and any SELECT results are printed to stdout,
so a single `python3 scripts/run_etl.py` gives you a full run log.

Requires:  pip install mysql-connector-python

Connection settings come from environment variables (with defaults),
or override on the command line:

    MYSQL_HOST=localhost MYSQL_USER=root MYSQL_PASSWORD=secret \\
        python3 scripts/run_etl.py

    python3 scripts/run_etl.py --host localhost --user root --password secret

Regenerate the raw source data first if you haven't:
    python3 scripts/generate_raw_data.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
except ImportError:
    print(
        "ERROR: mysql-connector-python is not installed.\n"
        "        pip install mysql-connector-python\n",
        file=sys.stderr,
    )
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SQL_DIR = PROJECT_ROOT / "sql"

PIPELINE_STEPS = [
    "00_functions.sql",
    "01_staging_schema.sql",
    "02_load_staging.sql",
    "03_warehouse_schema.sql",
    "04_transform_load.sql",
    "05_data_quality_checks.sql",
]


def split_statements(sql_text: str):
    """
    Splits a .sql file into individual statements, honoring MySQL-CLI-style
    `DELIMITER xx` directives (used in 00_functions.sql to define stored
    functions whose bodies contain semicolons). This mirrors what the
    `mysql` command-line client does with DELIMITER, so the same .sql files
    work identically whether run via this script or `mysql ... < file.sql`.
    """
    statements = []
    delimiter = ";"
    buffer = []

    for line in sql_text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("DELIMITER "):
            delimiter = stripped.split(None, 1)[1].strip()
            continue
        buffer.append(line)
        joined = "\n".join(buffer)
        if joined.rstrip().endswith(delimiter):
            stmt = joined.rstrip()
            stmt = stmt[: -len(delimiter)].strip()
            if stmt:
                statements.append(stmt)
            buffer = []

    tail = "\n".join(buffer).strip()
    if tail:
        statements.append(tail)

    return [s for s in statements if s and not s.startswith("--")]


def print_table(columns, rows, max_rows=25):
    if not rows:
        print("   (no rows)")
        return
    widths = [len(str(c)) for c in columns]
    shown = rows[:max_rows]
    for r in shown:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], len(str(v)))
    header = "   " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(columns))
    print(header)
    print("   " + "-+-".join("-" * w for w in widths))
    for r in shown:
        print("   " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(r)))
    if len(rows) > max_rows:
        print(f"   ... ({len(rows) - max_rows} more rows)")


def run_sql_file(cursor, path: Path):
    sql_text = path.read_text()
    statements = split_statements(sql_text)
    for stmt in statements:
        cursor.execute(stmt)
        if cursor.description:  # SELECT-like statement with a result set
            cols = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
            if rows:
                print_table(cols, rows)
        else:
            # consume any pending result to allow the next statement to run
            try:
                cursor.fetchall()
            except MySQLError:
                pass


def main():
    parser = argparse.ArgumentParser(description="Run the finance ETL pipeline against MySQL.")
    parser.add_argument("--host", default=os.environ.get("MYSQL_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MYSQL_PORT", "3306")))
    parser.add_argument("--user", default=os.environ.get("MYSQL_USER", "root"))
    parser.add_argument("--password", default=os.environ.get("MYSQL_PASSWORD", ""))
    parser.add_argument("--database", default=os.environ.get("MYSQL_DATABASE", "finance_dw"))
    parser.add_argument(
        "--only", help="Run a single step file only, e.g. --only 04_transform_load.sql"
    )
    args = parser.parse_args()

    steps = [args.only] if args.only else PIPELINE_STEPS

    print(f"Connecting to MySQL at {args.host}:{args.port} as {args.user} ...")
    conn = mysql.connector.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        allow_local_infile=True,  # required for LOAD DATA LOCAL INFILE in 02_load_staging.sql
        autocommit=True,
    )
    cursor = conn.cursor()

    # Run everything relative to the project root so LOAD DATA LOCAL INFILE's
    # relative paths ('data/raw/....csv') resolve correctly.
    os.chdir(PROJECT_ROOT)

    overall_start = time.time()
    try:
        for step in steps:
            path = SQL_DIR / step
            if not path.exists():
                print(f"ERROR: {path} not found", file=sys.stderr)
                sys.exit(1)
            print(f"\n=== {step} " + "=" * max(1, 60 - len(step)))
            t0 = time.time()
            run_sql_file(cursor, path)
            print(f"--- {step} done in {time.time() - t0:.2f}s")
    except MySQLError as e:
        print(f"\nMySQL ERROR during pipeline run: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        cursor.close()
        conn.close()

    print(f"\nPipeline finished in {time.time() - overall_start:.2f}s")


if __name__ == "__main__":
    main()
