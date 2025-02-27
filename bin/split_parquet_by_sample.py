import os
import pandas as pd
import click
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import multiprocessing
import time
from datetime import timedelta
import sys
from packaging import version

# Suppress specific pandas PerformanceWarning
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Progress tracking class
class ProgressTracker:
    def __init__(self, total_rows, chunksize):
        self.total_rows = total_rows
        self.chunksize = chunksize
        self.processed_rows = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_processed_rows = 0
        
    def update(self, additional_rows):
        """Update progress with newly processed rows and print status"""
        current_time = time.time()
        self.processed_rows += additional_rows
        
        # Calculate overall progress
        percent_complete = (self.processed_rows / self.total_rows) * 100
        elapsed_time = current_time - self.start_time
        
        # Calculate recent processing rate (last update interval)
        interval = current_time - self.last_update_time
        rows_since_last = self.processed_rows - self.last_processed_rows
        
        if interval > 0:
            recent_rate = rows_since_last / interval
            overall_rate = self.processed_rows / elapsed_time if elapsed_time > 0 else 0
            
            # Estimate time remaining
            remaining_rows = self.total_rows - self.processed_rows
            eta_seconds = remaining_rows / recent_rate if recent_rate > 0 else 0
            
            # Create progress bar
            bar_length = 30
            filled_length = int(bar_length * self.processed_rows // self.total_rows)
            progress_bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Format elapsed and ETA times
            elapsed_str = str(timedelta(seconds=int(elapsed_time)))
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            
            # Print progress information
            sys.stdout.write(f"\r[{progress_bar}] {percent_complete:.1f}% | "
                            f"{self.processed_rows:,}/{self.total_rows:,} rows | "
                            f"Rate: {recent_rate:.2f} rows/s | "
                            f"Elapsed: {elapsed_str} | ETA: {eta_str}")
            sys.stdout.flush()
            
            # Store current values for next update
            self.last_update_time = current_time
            self.last_processed_rows = self.processed_rows
    
    def finish(self):
        """Print final statistics"""
        total_time = time.time() - self.start_time
        overall_rate = self.processed_rows / total_time if total_time > 0 else 0
        print(f"\nCompleted processing {self.processed_rows:,} rows in {str(timedelta(seconds=int(total_time)))}")
        print(f"Average processing rate: {overall_rate:.2f} rows/second")


@click.command()
@click.option('--parquet-file-path', type=click.Path(exists=True), help='Path to input parquet file')
@click.option('--output-dir', type=click.Path(), default='.', help='Directory to save output files')
@click.option('--chunksize', type=int, default=5000000, help='Number of rows to process at once')
@click.option('--ncpus', type=int, default=None, help='Number of CPUs for parallel processing')
def split_parquet_by_sampledown(parquet_file_path, output_dir, chunksize, ncpus):
    """
    Read a parquet file and split it by the 'sampledown' column.
    
    For each unique sampledown value:
    1. Extract all rows with that sampledown value
    2. Output to {output_dir}/{sampledown}.parquet
    
    Also creates a samplesheet.csv with columns: sampledown, label, parquet_file_path
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set number of CPUs if not specified
    if ncpus is None:
        ncpus = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
    
    print(f"Using {ncpus} CPU cores for parallel processing")
    
    # Read parquet metadata to get total row count
    print(f"Reading parquet file metadata: {parquet_file_path}")
    parquet_file = pd.read_parquet(parquet_file_path, columns=[])
    total_rows = len(parquet_file)
    print(f"Total rows in parquet file: {total_rows:,}")
    
    # Calculate number of chunks
    num_chunks = (total_rows + chunksize - 1) // chunksize
    print(f"Will process file in {num_chunks} chunks of {chunksize:,} rows")
    
    # Get column names to ensure all processes read the same columns
    # Read a small portion of the file without filtering
    print("Reading sample of data to verify columns...")
    first_chunk = pd.read_parquet(parquet_file_path).head(10)
    columns = list(first_chunk.columns)
    
    # Verify required columns exist
    required_columns = ['sampledown']
    missing_columns = [col for col in required_columns if col not in columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check if label column exists
    has_label_column = 'label' in columns
    print(f"Label column {'found' if has_label_column else 'not found'} in parquet file")
    
    # Create a list of chunk indices and their offsets
    chunk_offsets = [(i, i * chunksize) for i in range(num_chunks)]
    
    # Initialize progress tracker
    progress = ProgressTracker(total_rows, chunksize)
    print("Starting processing...")
    
    # Process chunks in parallel and collect filtered data directly
    with ProcessPoolExecutor(max_workers=ncpus) as executor:
        # Submit all chunk processing tasks
        futures = {
            executor.submit(
                process_chunk_parallel, 
                parquet_file_path, 
                chunk_idx, 
                offset, 
                chunksize, 
                total_rows,
                has_label_column
            ): (chunk_idx, min(chunksize, total_rows - offset)) for chunk_idx, offset in chunk_offsets
        }
        
        # Initialize storage for collected sample data and metadata
        sample_data = {}
        sample_labels = {}
        
        # Process results as they complete
        for future in as_completed(futures):
            chunk_idx, chunk_size = futures[future]
            try:
                # Each worker now returns the actual filtered data for each sampledown
                sampledown_data, sampledown_labels = future.result()
                
                # Merge sampledown data from this chunk
                for sampledown, data in sampledown_data.items():
                    if sampledown not in sample_data:
                        sample_data[sampledown] = []
                    sample_data[sampledown].append(data)
                
                # Merge sampledown labels
                for sampledown, label in sampledown_labels.items():
                    sample_labels[sampledown] = label
                
                # Update progress with the rows processed in this chunk
                progress.update(chunk_size)
                
            except Exception as e:
                print(f"\nError processing chunk {chunk_idx + 1}: {e}")
    
    # Finalize progress tracking
    progress.finish()
    
    # Create a list to store sample information for samplesheet
    samplesheet_data = []
    
    # For tracking output file row counts
    output_file_stats = []
    
    # Process each sampledown value and write to separate files
    print("\nWriting output files...")
    write_start_time = time.time()
    total_samples = len(sample_data)
    
    for idx, (sampledown, chunk_dfs) in enumerate(sample_data.items(), 1):
        print(f"Processing sampledown: {sampledown} ({idx}/{total_samples})")
        
        # Check if we have any data for this sampledown
        if not chunk_dfs:
            print(f"No data found for sampledown: {sampledown}")
            continue
        
        # Combine all chunks for this sampledown
        sample_combined = pd.concat(chunk_dfs, ignore_index=True)
        total_rows_for_sample = len(sample_combined)
        
        # Determine output filename
        output_file = os.path.join(output_dir, f"{sampledown}.parquet")
        
        # Get absolute path for samplesheet
        absolute_path = os.path.abspath(output_file)
        
        # Get label for this sampledown
        label = sample_labels.get(sampledown, "unknown")
        
        # Add to samplesheet data with label
        samplesheet_data.append({
            'sampledown': sampledown,
            'label': label,
            'parquet': absolute_path
        })
        
        # Store stats for the output file
        output_file_stats.append({
            'sampledown': sampledown,
            'label': label,
            'nrows': total_rows_for_sample,
            'file_path': absolute_path
        })
        
        # Write to parquet - use pyarrow engine for faster writing
        print(f"Writing to: {output_file} ({total_rows_for_sample:,} rows)")
        sample_combined.to_parquet(output_file, index=False, engine='pyarrow', compression='snappy')
    
    write_time = time.time() - write_start_time
    print(f"File writing completed in {str(timedelta(seconds=int(write_time)))}")
    
    # Create and write the samplesheet
    samplesheet_path = os.path.join(output_dir, 'samplesheet.csv')
    samplesheet_df = pd.DataFrame(samplesheet_data)
    samplesheet_df.to_csv(samplesheet_path, index=False)
    print(f"Samplesheet created at: {samplesheet_path}")
    
    # Create output file statistics report
    stats_df = pd.DataFrame(output_file_stats)
    stats_path = os.path.join(output_dir, 'output_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    
    # Display output file row counts
    print("\n" + "="*80)
    print(" OUTPUT FILE STATISTICS ".center(80, "="))
    print("="*80)
    print(f"{'SAMPLEDOWN':35} {'LABEL':10} {'ROWS':10} {'PATH'}")
    print("-"*80)
    
    # Sort by row count (descending)
    for stat in sorted(output_file_stats, key=lambda x: -x['nrows']):
        sampledown = stat['sampledown']
        label = stat['label']
        nrows = stat['nrows']
        path = os.path.basename(stat['file_path'])
        print(f"{sampledown:35} {label:10} {nrows:10,d} {path}")
    
    # Print summary statistics
    total_output_rows = sum(stat['nrows'] for stat in output_file_stats)
    print("-"*80)
    print(f"{'TOTAL':35} {' ':10} {total_output_rows:10,d}")
    print("="*80)
    print(f"Detailed statistics saved to: {stats_path}")
    
    # Calculate and display final statistics
    total_time = time.time() - progress.start_time
    print(f"\nTotal execution time: {str(timedelta(seconds=int(total_time)))}")
    print(f"Processing rate: {total_rows/total_time:.2f} rows/second")
    print("Processing complete")


def process_chunk_parallel(parquet_file_path, chunk_idx, offset, chunksize, total_rows, has_label_column):
    """
    Process a chunk in a parallel worker process and return filtered DataFrames for each sampledown value.
    This optimized version reduces I/O by reading each chunk only once and returning the actual filtered data.
    """
    import pandas as pd  # Re-import in the worker process
    import time  # For timing the chunk processing
    
    # Calculate actual chunk size (might be smaller for the last chunk)
    end_offset = min(offset + chunksize, total_rows)
    actual_size = end_offset - offset
    
    worker_start_time = time.time()
    worker_id = multiprocessing.current_process().name
    
    print(f"Worker {worker_id}: Starting chunk {chunk_idx+1} (rows {offset:,}-{end_offset:,})")
    
    # Read the chunk - be specific about columns to reduce memory usage
    try:
        # Determine which columns to read
        cols_to_read = ['sampledown']
        if has_label_column:
            cols_to_read.append('label')
        
        # Try reading with specific columns first
        chunk = pd.read_parquet(
            parquet_file_path, 
            engine='pyarrow',  # Use pyarrow engine for better performance
            filters=None,  # No pre-filtering at this stage
        ).iloc[offset:end_offset]
    except Exception as e:
        print(f"Worker {worker_id}: Error reading chunk: {e}. Trying an alternative approach.")
        try:
            # Alternative approach with basic options
            chunk = pd.read_parquet(parquet_file_path, use_pandas_metadata=False).iloc[offset:end_offset]
        except Exception as e2:
            print(f"Worker {worker_id}: Error with alternative approach: {e2}. Fatal error.")
            raise
    
    # Initialize dictionaries to store results
    sampledown_data = {}
    sampledown_labels = {}
    
    # Get unique sampledown values in this chunk
    unique_sampledowns = chunk['sampledown'].unique()
    
    # Process each sampledown group
    for sampledown in unique_sampledowns:
        # Filter for this sampledown
        sampledown_df = chunk[chunk['sampledown'] == sampledown]
        
        # Store the dataframe for this sampledown
        sampledown_data[sampledown] = sampledown_df
        
        # Store the label if available
        if has_label_column and not sampledown_df.empty:
            sampledown_labels[sampledown] = sampledown_df['label'].iloc[0]
    
    # Log worker completion time
    worker_time = time.time() - worker_start_time
    processing_rate = actual_size / worker_time if worker_time > 0 else 0
    print(f"Worker {worker_id}: Completed chunk {chunk_idx+1} in {worker_time:.2f}s ({processing_rate:.2f} rows/s)")
    
    return sampledown_data, sampledown_labels


if __name__ == "__main__":
    # Needed for Windows multiprocessing
    import concurrent.futures
    split_parquet_by_sampledown()
