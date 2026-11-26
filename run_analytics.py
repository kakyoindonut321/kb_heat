
import csv
import os
import re
from pathlib import Path

import psycopg
import sqlparse


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SQL_FILE = BASE_DIR / "analytics.sql"
OUTPUT_DIR = BASE_DIR / "analytics_results"

DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": os.getenv("PGPORT", "5432"),
    "dbname": os.getenv("PGDATABASE", "kb_heat"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", ""),
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_filename(text):
    """Make a safe filename from SQL statement."""
    text = re.sub(r"[^a-zA-Z0-9_-]", "_", text)
    return text[:40].strip("_") or "query"


def get_query_name(statement):
    """Generate a readable query name."""
    formatted = sqlparse.format(
        statement,
        strip_comments=True,
        strip_whitespace=True
    )

    words = formatted.split()

    if not words:
        return "query"

    return clean_filename("_".join(words[:5]))


def export_csv(cursor, output_path):
    """Export current result set to UTF-8 CSV."""

    columns = [column.name for column in cursor.description]
    rows = cursor.fetchall()

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(columns)
        writer.writerows(rows)

    return len(rows)


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    if not SQL_FILE.exists():
        raise FileNotFoundError(
            f"SQL file not found: {SQL_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sql_content = SQL_FILE.read_text(encoding="utf-8")

    statements = sqlparse.split(sql_content)

    print("=" * 55)
    print("POSTGRESQL ANALYTICS EXECUTOR")
    print("=" * 55)

    print(f"SQL file : {SQL_FILE.name}")
    print(f"Database : {DB_CONFIG['dbname']}")
    print(f"Output   : {OUTPUT_DIR}")
    print(f"Statements found: {len(statements)}")
    print("=" * 55)

    exported = 0
    executed = 0

    # Autocommit allows each statement to be committed
    # independently, including CREATE INDEX and DROP INDEX.
    with psycopg.connect(
        **DB_CONFIG,
        autocommit=True
    ) as connection:

        for index, statement in enumerate(statements, start=1):

            # Ignore empty or comment-only statements
            cleaned = sqlparse.format(
                statement,
                strip_comments=True
            ).strip()

            if not cleaned:
                continue

            query_name = get_query_name(statement)

            print(f"\n[{index}/{len(statements)}] {query_name}")

            try:
                with connection.cursor() as cursor:

                    cursor.execute(statement)
                    executed += 1

                    # Export statements that return data
                    if cursor.description is not None:

                        filename = (
                            f"{index:03d}_{query_name}.csv"
                        )

                        output_path = OUTPUT_DIR / filename

                        row_count = export_csv(
                            cursor,
                            output_path
                        )

                        exported += 1

                        print(
                            f"  [CSV] {filename}"
                            f" ({row_count:,} rows)"
                        )

                    else:
                        print("  [OK] Executed successfully")

            except Exception as error:
                print(f"  [ERROR] {error}")
                print("Execution stopped.")
                raise

    print("\n" + "=" * 55)
    print("EXECUTION COMPLETED")
    print("=" * 55)

    print(f"Statements executed : {executed}")
    print(f"CSV files generated : {exported}")
    print(f"Output directory    : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
