#!/usr/bin/env python
"""
Author: Tim Frenzel
Version: 1.00
Usage:  python src/etl/extract_synthea_data.py

Objective of the Code:
------------
This script extracts all CSV files from the Synthea tar.gz archives and merges 
them into a common dataset directory. It preserves the original CSV names
(e.g., conditions.csv, patients.csv, etc.) but combines the data from all 12 archives
into a single set of files, ensuring no duplication occurs. The merged dataset is then
ready for conversion to Parquet format by the csv_to_parquet.py script.
"""

import os
import shutil
import tarfile
import pathlib
import pandas as pd
import time
import datetime
import signal
import sys
import argparse
from tqdm import tqdm

# Paths
ROOT = pathlib.Path(__file__).resolve().parents[2]  # Project root (2 levels up from etl directory)
TARGET_DIR = ROOT / "data" / "synthea_csv_merged"  # Where to put the merged CSV files
TEMP_DIR = ROOT / "temp_extraction"  # Temporary directory for extraction
LOG_FILE = ROOT / "logs" / "extraction_log.txt"  # Log file

# Create directories
TARGET_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Global variables for tracking
current_archive = ""
processed_archives = set()
start_time = time.time()

def format_time(seconds):
    """Format seconds into human-readable time."""
    return str(datetime.timedelta(seconds=int(seconds)))

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
    log_message(f"Completed archives: {len(processed_archives)}", level="WARNING")
    log_message(f"Last archive processed: {current_archive}", level="WARNING")
    
    # Cleanup temporary directory
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    
    sys.exit(1)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def extract_and_merge(source_dir_path: pathlib.Path):
    """Extract CSV files from all archives and merge them into the target directory."""
    global current_archive, processed_archives, start_time
    
    start_time = time.time()
    log_message(f"Starting extraction from {source_dir_path}")
    log_message(f"Target directory: {TARGET_DIR}")
    log_message(f"Temporary extraction directory: {TEMP_DIR}")
    log_message(f"Log file: {LOG_FILE}")
    
    # Get all tar.gz files
    tar_files = list(source_dir_path.glob("*.tar.gz"))
    
    if not tar_files:
        log_message(f"No tar.gz files found in {source_dir_path}", level="ERROR")
        return
    
    log_message(f"Found {len(tar_files)} archives to process")
    log_message(f"Estimated total size: ~21GB compressed")
    log_message("-" * 80)
    
    # Track dataframes for each CSV type
    dataframes = {}
    csv_file_types = set()
    csv_file_counts = {
        "total": 0,
        "empty": 0,
        "error": 0,
        "processed": 0
    }
    
    # Process each archive
    archive_progress = tqdm(tar_files, desc="Processing archives", unit="archive")
    for i, tar_path in enumerate(archive_progress, 1):
        archive_name = tar_path.name
        current_archive = archive_name
        
        # Update progress bar description
        archive_progress.set_description(f"Processing {archive_name}")
        
        archive_start = time.time()
        log_message(f"Archive {i}/{len(tar_files)}: {archive_name}")
        
        # Clear temp directory
        for item in TEMP_DIR.glob("*"):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        
        try:
            # Get archive size
            archive_size_mb = tar_path.stat().st_size / (1024 * 1024)
            log_message(f"  Archive size: {archive_size_mb:.2f} MB")
            
            # Extract only CSV files from the archive
            with tarfile.open(tar_path, 'r:gz') as tar:
                # Find all CSV files
                csv_members = [m for m in tar.getmembers() 
                              if m.name.endswith('.csv') and not m.isdir() and '/csv/' in m.name]
                
                if not csv_members:
                    log_message(f"  No CSV files found in {archive_name}", level="WARNING")
                    continue
                
                log_message(f"  Found {len(csv_members)} CSV files to extract")
                csv_file_counts["total"] += len(csv_members)
                
                # Process each CSV file
                file_progress = tqdm(csv_members, desc="  Extracting files", leave=False, unit="file")
                for member in file_progress:
                    filename = os.path.basename(member.name)
                    file_progress.set_description(f"  Extracting {filename}")
                    
                    try:
                        # Extract just this file
                        tar.extract(member, path=TEMP_DIR)
                        csv_file_types.add(filename)
                        
                        # Read the CSV file
                        csv_path = TEMP_DIR / member.name
                        file_size_mb = csv_path.stat().st_size / (1024 * 1024)
                        
                        # Try to read the file
                        try:
                            # Check if file is readable
                            with open(csv_path, 'r', encoding='utf-8') as f:
                                header = f.readline().strip()
                            
                            if not header:
                                log_message(f"    {filename}: Empty file (skipping)", level="WARNING")
                                csv_file_counts["empty"] += 1
                                continue
                            
                            # Read in chunks if the file is large, skip bad lines
                            chunk_size = 100000  # Adjust based on available memory
                            csv_reader = pd.read_csv(
                                csv_path, 
                                chunksize=chunk_size, 
                                low_memory=False, 
                                on_bad_lines='skip' # Skip rows with incorrect number of fields
                            )
                            
                            # Process chunks
                            chunks = []
                            total_rows = 0
                            
                            for chunk in csv_reader:
                                chunks.append(chunk)
                                total_rows += len(chunk)
                            
                            if chunks:
                                # Combine chunks
                                df = pd.concat(chunks, ignore_index=True)
                                
                                # Add to the dataframes dictionary for later concatenation
                                if filename in dataframes:
                                    dataframes[filename].append(df)
                                else:
                                    dataframes[filename] = [df]
                                
                                log_message(f"    {filename}: {len(df)} rows, {file_size_mb:.2f} MB", also_print=False)
                                csv_file_counts["processed"] += 1
                            else:
                                log_message(f"    {filename}: File appears to be empty or has no data rows", level="WARNING")
                                csv_file_counts["empty"] += 1
                                
                        except Exception as e:
                            log_message(f"    Error reading {filename}: {str(e)}", level="ERROR")
                            csv_file_counts["error"] += 1
                    
                    except Exception as e:
                        log_message(f"    Error extracting {filename}: {str(e)}", level="ERROR")
                        csv_file_counts["error"] += 1
                
            # Mark this archive as processed
            processed_archives.add(archive_name)
            
            # Archive completion stats
            archive_elapsed = time.time() - archive_start
            log_message(f"  Archive {archive_name} completed in {format_time(archive_elapsed)}")
            
            # Overall progress
            total_elapsed = time.time() - start_time
            avg_per_archive = total_elapsed / len(processed_archives)
            remaining = avg_per_archive * (len(tar_files) - len(processed_archives))
            
            log_message(f"  Overall progress: {len(processed_archives)}/{len(tar_files)} archives")
            log_message(f"  Elapsed time: {format_time(total_elapsed)}, Est. remaining: {format_time(remaining)}")
            log_message("-" * 80)
                
        except Exception as e:
            log_message(f"  Error processing archive {archive_name}: {str(e)}", level="ERROR")
    
    # Merge and save all dataframes
    log_message("\nMerging and saving combined CSV files...")
    log_message(f"CSV file processing summary:")
    log_message(f"  Total CSV files found: {csv_file_counts['total']}")
    log_message(f"  Empty files skipped: {csv_file_counts['empty']}")
    log_message(f"  Files with errors: {csv_file_counts['error']}")
    log_message(f"  Successfully processed: {csv_file_counts['processed']}")
    
    # Process each file type
    for filename, dfs in dataframes.items():
        if not dfs:
            log_message(f"  {filename}: No data to merge (skipping)", level="WARNING")
            continue
        
        try:
            merge_start = time.time()
            log_message(f"  Merging {filename} data from {len(dfs)} archives...")
            
            # Concatenate all dataframes for this file type
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Get memory usage
            memory_usage = combined_df.memory_usage(deep=True).sum() / (1024 * 1024)
            log_message(f"    Memory usage: {memory_usage:.2f} MB")
            
            # Check for duplicates based on Id column if it exists
            if 'Id' in combined_df.columns:
                before_count = len(combined_df)
                combined_df = combined_df.drop_duplicates(subset=['Id'])
                after_count = len(combined_df)
                
                if before_count > after_count:
                    log_message(f"    Removed {before_count - after_count} duplicate records ({(before_count - after_count) / before_count * 100:.2f}%)")
            
            # Save the combined dataframe
            output_path = TARGET_DIR / filename
            log_message(f"    Saving to {output_path}...")
            combined_df.to_csv(output_path, index=False)
            
            # Report completion
            merge_elapsed = time.time() - merge_start
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            log_message(f"    Saved {filename} with {len(combined_df)} rows, {file_size_mb:.2f} MB in {format_time(merge_elapsed)}")
            
        except Exception as e:
            log_message(f"    Error saving {filename}: {str(e)}", level="ERROR")
    
    # Cleanup temporary directory
    log_message("\nCleaning up temporary files...")
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    
    # Final summary
    total_elapsed = time.time() - start_time
    log_message("-" * 80)
    log_message("Extraction and merging complete!")
    log_message(f"Total execution time: {format_time(total_elapsed)}")
    log_message(f"Total archives processed: {len(processed_archives)}/{len(tar_files)}")
    log_message(f"Output files saved to: {TARGET_DIR}")
    log_message(f"CSV file types processed: {sorted(list(csv_file_types))}")
    log_message(f"Log file: {LOG_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and merge Synthea CSV data from tar.gz archives.")
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Path to the directory containing Synthea tar.gz archives.")
    
    args = parser.parse_args()
    
    source_path = pathlib.Path(args.source_dir)
    if not source_path.is_dir():
        log_message(f"Error: Source directory '{args.source_dir}' not found or is not a directory.", level="ERROR", also_print=True)
        sys.exit(1)
        
    extract_and_merge(source_path)
    
    print("\n" + "="*80)
    print(f"EXTRACTION PROCESS COMPLETE".center(80))
    print(f"Finished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
    print("="*80 + "\n") 