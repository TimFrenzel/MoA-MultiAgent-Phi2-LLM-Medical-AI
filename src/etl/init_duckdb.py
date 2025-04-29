"""
Author: Tim Frenzel
Version: 1.20
Usage:  python src/etl/init_duckdb.py

Objective of the Code:
------------
This script initializes a DuckDB database instance for Synthea data.
It scans the specified Parquet directory ('data/parquet') for subdirectories,
each containing partitioned Parquet files for a specific data table (e.g., patients, conditions).
For each subdirectory found, it creates a DuckDB view that reads all Parquet files
within that directory, making the data easily queryable via the DuckDB database.
"""
import duckdb, pathlib, json, textwrap
import sys # For exit

ROOT     = pathlib.Path(__file__).resolve().parents[2]
PARQUET  = ROOT / "data" / "parquet"
DB_PATH  = ROOT / "data" / "duckdb" / "synthea.duckdb"

print(f"Source Parquet Directory: {PARQUET}")
print(f"Target DuckDB Database: {DB_PATH}")

# --- Pre-checks ---
if not PARQUET.exists() or not PARQUET.is_dir():
    print(f"\n[ERROR] Parquet directory not found: {PARQUET}")
    print(f"Please ensure the 'src/etl/csv_to_parquet.py' script ran successfully.")
    sys.exit(1)

# Ensure parent directory for DB exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Database Connection ---
try:
    # Connect (creates DB file if it doesn't exist)
    con = duckdb.connect(str(DB_PATH))
    print(f"\nSuccessfully connected to DuckDB database.")

    # --- View Creation ---
    print("Creating views for Parquet data...")
    
    view_count = 0
    # Register each folder as an external table view
    for folder in PARQUET.iterdir():
        if folder.is_dir(): # Process only directories
            view_name = folder.name
            parquet_path_pattern = str(folder / "*.parquet").replace('\\', '/') # Ensure forward slashes for DuckDB
            
            print(f"  Creating view: {view_name}" )
            print(f"    -> Reading from: {parquet_path_pattern}")
            
            try:
                # Use f-string carefully, ensuring paths are handled correctly
                sql = f"""
                    CREATE OR REPLACE VIEW \"{view_name}\" AS
                    SELECT * FROM read_parquet('{parquet_path_pattern}');
                """
                con.execute(sql)
                view_count += 1
            except Exception as e:
                print(f"    [ERROR] Failed to create view for {view_name}: {e}")
        else:
            print(f"  Skipping non-directory item: {folder.name}")

    print(f"\nSuccessfully created/updated {view_count} views.")

except Exception as e:
    print(f"\n[ERROR] Failed to connect to or process DuckDB database: {e}")
    sys.exit(1)

finally:
    # --- Close Connection ---
    if 'con' in locals() and con:
        con.close()
        print("Database connection closed.")

print(f"\nDuckDB initialization complete. Database ready at {DB_PATH}")
