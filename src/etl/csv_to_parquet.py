#!/usr/bin/env python
"""
Author: Tim Frenzel
Version: 1.10
Usage:  python src/etl/csv_to_parquet.py

Objective of the Code:
------------
This script converts selected Synthea CSV files located in the specified input
directory to the snappy-compressed Parquet format. It processes the CSVs in
chunks using pandas, retains only predefined columns for specific files,
and writes the resulting Arrow Tables as partitioned Parquet files to the
output directory.
"""

import duckdb, pathlib, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
import os, time, datetime, signal, sys
from tqdm import tqdm

# -------- config --------
ROOT   = pathlib.Path(__file__).resolve().parents[2]          # project root
RAW    = ROOT / "data" / "synthea_csv_merged"  # merged CSV files from extract_synthea_data.py
OUTDIR = ROOT / "data" / "parquet"
LOG_FILE = ROOT / "logs" / "parquet_conversion_log.txt"  # Log file

# Create directories
OUTDIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Global variables for tracking
current_file = ""
processed_files = set()
start_time = time.time()

keep_cols = {
    # Updated based on review of merged files and requirements
    "patients.csv":      ["ID", "BIRTHDATE", "DEATHDATE", "MARITAL", "RACE", "ETHNICITY", "GENDER"],
    "conditions.csv":    ["START", "STOP", "PATIENT", "CODE", "DESCRIPTION"], # Target variable source
    "observations.csv":  ["DATE", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "VALUE", "UNITS"],
    "encounters.csv":    ["ID", "DATE", "PATIENT", "CODE", "DESCRIPTION", "REASONCODE", "REASONDESCRIPTION"],
    "medications.csv":   ["START", "STOP", "PATIENT", "CODE", "DESCRIPTION", "ENCOUNTER", "REASONCODE", "REASONDESCRIPTION"],
    "procedures.csv":    ["DATE", "PATIENT", "CODE", "DESCRIPTION", "ENCOUNTER", "REASONCODE", "REASONDESCRIPTION"],
    "allergies.csv":     ["PATIENT", "CODE", "DESCRIPTION", "START", "STOP"], # Optional
    "immunizations.csv": ["PATIENT", "DATE", "CODE"], # Optional
    # devices.csv is confirmed unavailable in the source data
}

def format_time(seconds):
    """Format seconds into human-readable time."""
    return str(datetime.timedelta(seconds=int(seconds)))

def format_size(bytes_size):
    """Format bytes into human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"

def log_message(message, level="INFO", also_print=True):
    """Log a message to both the console and log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] [{level}] {message}"
    
    if also_print:
        print(formatted_msg)
    
    with open(LOG_FILE, "a") as f:
        f.write(formatted_msg + "\n")

def signal_handler(sig, frame):
    """Handle interrupt signals gracefully."""
    elapsed = time.time() - start_time
    log_message(f"\nProcess interrupted! Elapsed time: {format_time(elapsed)}", level="WARNING")
    log_message(f"Completed files: {len(processed_files)}/{len(list(RAW.glob('*.csv')))}", level="WARNING")
    log_message(f"Last file processed: {current_file}", level="WARNING")
    sys.exit(1)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def convert_to_parquet():
    """Convert CSV files to Parquet format."""
    global current_file, processed_files, start_time
    
    start_time = time.time()
    log_message("Starting CSV to Parquet conversion")
    log_message(f"Input directory: {RAW}")
    log_message(f"Output directory: {OUTDIR}")
    log_message(f"Log file: {LOG_FILE}")
    
    # Check if input directory exists
    if not RAW.exists():
        log_message(f"Error: Input directory {RAW} does not exist!", level="ERROR")
        log_message("Please run src/etl/extract_synthea_data.py first to create the merged CSV files.", level="ERROR")
        return False
    
    # Find all CSV files
    csv_files = list(RAW.glob("*.csv"))
    log_message(f"Found {len(csv_files)} CSV files")
    
    if not csv_files:
        log_message("No CSV files found in the input directory", level="ERROR")
        return False
    
    # Print sizes of input files
    total_size = 0
    log_message("Input file sizes:")
    for csv_path in csv_files:
        file_size = csv_path.stat().st_size
        total_size += file_size
        log_message(f"  {csv_path.name}: {format_size(file_size)}")
    
    log_message(f"Total input size: {format_size(total_size)}")
    log_message("-" * 80)
    
    # Track conversion statistics
    stats = {
        "total_files": len(csv_files),
        "processed_files": 0,
        "skipped_files": 0,
        "error_files": 0,
        "total_chunks": 0,
        "input_size": total_size,
        "output_size": 0
    }
    
    # Process each CSV file
    file_progress = tqdm(csv_files, desc="Converting files", unit="file")
    for csv_path in file_progress:
        file_start = time.time()
        name = csv_path.stem                     # e.g. patients
        filename = csv_path.name                 # e.g. patients.csv
        current_file = filename
        
        # Update progress bar description
        file_progress.set_description(f"Converting {filename}")
        
        log_message(f"Processing {filename}")
        
        if filename not in keep_cols:               # skip dropped files
            log_message(f"  Skipping {filename} (not in keep_cols)", level="WARNING")
            stats["skipped_files"] += 1
            continue
        
        try:
            # Get file size and stats
            file_size = csv_path.stat().st_size
            log_message(f"  Input file size: {format_size(file_size)}")
            
            # Determine columns to keep
            cols = None if keep_cols[filename] == "*" else keep_cols[filename]
            if cols:
                log_message(f"  Keeping {len(cols)} columns: {', '.join(cols)}")
            else:
                log_message(f"  Keeping all columns")
            
            # Create output directory for this file type
            out_dir = OUTDIR / name
            out_dir.mkdir(exist_ok=True)
            
            # Read and convert in chunks
            try:
                # First check if the file is readable
                with open(csv_path, 'r', encoding='utf-8') as f:
                    header = f.readline().strip()
                
                if not header:
                    log_message(f"  File appears to be empty - skipping", level="WARNING")
                    stats["skipped_files"] += 1
                    continue
                
                # Parse in chunks
                chunks = pd.read_csv(csv_path, usecols=cols, chunksize=250_000, low_memory=False)
                
                # Process each chunk
                i = 0
                chunk_progress = tqdm(desc="  Processing chunks", unit="chunk", leave=False)
                total_rows = 0
                output_size = 0
                
                for df in chunks:
                    chunk_progress.update(1)
                    # Track total rows
                    total_rows += len(df)
                    
                    # Convert to Arrow Table and write to Parquet
                    table = pa.Table.from_pandas(df, preserve_index=False)
                    out_file = out_dir / f"part-{i:05d}.parquet"
                    pq.write_table(table, out_file, compression="snappy")
                    
                    # Track output size
                    part_size = out_file.stat().st_size
                    output_size += part_size
                    
                    # Log detailed info
                    log_message(f"    Chunk {i}: {len(df)} rows, {format_size(part_size)}", also_print=False)
                    
                    i += 1
                    stats["total_chunks"] += 1
                
                chunk_progress.close()
                
                # Track total output size
                stats["output_size"] += output_size
                
                # Calculate compression ratio
                compression_ratio = file_size / output_size if output_size > 0 else 0
                
                # Log summary for this file
                file_elapsed = time.time() - file_start
                log_message(f"  Converted {filename} to {i} Parquet files")
                log_message(f"  Total rows: {total_rows}")
                log_message(f"  Input size: {format_size(file_size)}, Output size: {format_size(output_size)}")
                log_message(f"  Compression ratio: {compression_ratio:.2f}x")
                log_message(f"  Time taken: {format_time(file_elapsed)}")
                
                # Mark as processed
                processed_files.add(filename)
                stats["processed_files"] += 1
                
            except Exception as e:
                log_message(f"  Error reading/processing {filename}: {str(e)}", level="ERROR")
                stats["error_files"] += 1
        
        except Exception as e:
            log_message(f"  Error with {filename}: {str(e)}", level="ERROR")
            stats["error_files"] += 1
        
        log_message("-" * 80)
    
    # Print final summary
    total_elapsed = time.time() - start_time
    
    log_message("\nConversion Summary:")
    log_message(f"  Total files processed: {stats['processed_files']}/{stats['total_files']}")
    log_message(f"  Files skipped: {stats['skipped_files']}")
    log_message(f"  Files with errors: {stats['error_files']}")
    log_message(f"  Total chunks written: {stats['total_chunks']}")
    log_message(f"  Total input size: {format_size(stats['input_size'])}")
    log_message(f"  Total output size: {format_size(stats['output_size'])}")
    
    # Calculate overall compression ratio
    if stats['output_size'] > 0:
        overall_ratio = stats['input_size'] / stats['output_size']
        log_message(f"  Overall compression ratio: {overall_ratio:.2f}x")
    
    log_message(f"  Total execution time: {format_time(total_elapsed)}")
    
    if stats["processed_files"] == stats["total_files"] - stats["skipped_files"]:
        log_message("All files converted successfully!")
        return True
    else:
        log_message(f"Conversion completed with {stats['error_files']} errors", level="WARNING")
        return False

if __name__ == "__main__":
    print("\n" + "="*80)
    print(f"CSV TO PARQUET CONVERSION UTILITY".center(80))
    print(f"Starting at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
    print("="*80 + "\n")
    
    try:
        success = convert_to_parquet()
        exit_code = 0 if success else 1
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        exit_code = 1
    except Exception as e:
        log_message(f"Unhandled exception: {str(e)}", level="ERROR")
        exit_code = 1
    
    print("\n" + "="*80)
    print(f"CONVERSION PROCESS COMPLETE".center(80))
    print(f"Finished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
    print("="*80 + "\n")
    
    sys.exit(exit_code)
