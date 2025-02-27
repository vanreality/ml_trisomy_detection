import os
import pandas as pd
import click
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import warnings
import multiprocessing
import time
from datetime import timedelta
import sys

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
@click.option('--parquet-file-path', type=click.Path(exists=True))
@click.option('--output-dir', type=click.Path(), default='.')
@click.option('--chunksize', type=int, default=5000000, help='Number of rows to process at once')
@click.option('--ncpus', type=int, default=None, help='Number of CPUs for parallel processing')
def split_parquet_by_sample(parquet_file_path, output_dir, chunksize, ncpus):
    """
    Read a parquet file and split it by sample.
    
    For each sample:
    1. Group by seqname column
    2. Calculate the average value of prob_class_1 column for each group
    3. Replace prob_class_1 column value of each group by the average value
    4. Output to {output_dir}/{sample}_{label}.parquet
    
    Also creates a samplesheet.csv with columns: sample, label, parquet_file_path
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
    first_chunk = next(pd.read_parquet(parquet_file_path, chunksize=10))
    columns = list(first_chunk.columns)
    
    # Verify required columns exist
    required_columns = ['sample', 'seqname', 'prob_class_1', 'label']
    missing_columns = [col for col in required_columns if col not in columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Create a list of chunk indices and their offsets
    chunk_offsets = [(i, i * chunksize) for i in range(num_chunks)]
    
    # Initialize progress tracker
    progress = ProgressTracker(total_rows, chunksize)
    print("Starting processing...")
    
    # Process chunks in parallel
    results = []
    with ProcessPoolExecutor(max_workers=ncpus) as executor:
        # Submit all chunk processing tasks
        futures = {
            executor.submit(
                process_chunk_parallel, 
                parquet_file_path, 
                chunk_idx, 
                offset, 
                chunksize, 
                total_rows
            ): (chunk_idx, min(chunksize, total_rows - offset)) for chunk_idx, offset in chunk_offsets
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            chunk_idx, chunk_size = futures[future]
            try:
                chunk_result = future.result()
                results.append(chunk_result)
                
                # Update progress with the rows processed in this chunk
                progress.update(chunk_size)
                
            except Exception as e:
                print(f"\nError processing chunk {chunk_idx + 1}: {e}")
    
    # Finalize progress tracking
    progress.finish()
    
    # Merge results from all chunks
    print("\nMerging results from all chunks...")
    merge_start_time = time.time()
    
    sample_data = {}
    sample_labels = {}
    
    for chunk_sample_data, chunk_sample_labels in results:
        # Merge sample labels
        for sample, label in chunk_sample_labels.items():
            sample_labels[sample] = label
        
        # Merge sample data
        for sample, seqnames in chunk_sample_data.items():
            if sample not in sample_data:
                sample_data[sample] = {}
                
            for seqname, data in seqnames.items():
                if seqname in sample_data[sample]:
                    # Get existing values
                    existing_rows = sample_data[sample][seqname]['rows']
                    existing_sum = sample_data[sample][seqname]['sum']
                    
                    # Update with new data
                    new_rows = existing_rows + data['rows']
                    new_sum = existing_sum + data['sum']
                    new_avg = new_sum / new_rows
                    
                    # Store updated values
                    sample_data[sample][seqname] = {
                        'rows': new_rows,
                        'sum': new_sum,
                        'prob_class_1': new_avg
                    }
                    
                    # Copy other columns as needed
                    for col, val in data.items():
                        if col not in ['rows', 'sum', 'prob_class_1']:
                            sample_data[sample][seqname][col] = val
                else:
                    sample_data[sample][seqname] = data
    
    merge_time = time.time() - merge_start_time
    print(f"Merge completed in {str(timedelta(seconds=int(merge_time)))}")
    
    # Create a list to store sample information for samplesheet
    samplesheet_data = []
    
    # Write the processed data for each sample
    print("\nWriting output files...")
    write_start_time = time.time()
    total_samples = len(sample_data)
    
    for idx, (sample, data) in enumerate(sample_data.items(), 1):
        # Convert the dictionary to a dataframe
        sample_df = pd.DataFrame.from_dict(data, orient='index')
        sample_df = sample_df.reset_index().rename(columns={'index': 'seqname'})
        
        # Get the label
        label = sample_labels[sample]
        
        # Determine output filename
        output_file = os.path.join(output_dir, f"{sample}_{label}.parquet")
        
        # Get absolute path for samplesheet
        absolute_path = os.path.abspath(output_file)
        
        # Add to samplesheet data
        samplesheet_data.append({
            'sample': sample,
            'label': label,
            'parquet': absolute_path
        })
        
        # Write to parquet
        print(f"Writing to: {output_file} ({idx}/{total_samples})")
        sample_df.to_parquet(output_file, index=False)
    
    write_time = time.time() - write_start_time
    print(f"File writing completed in {str(timedelta(seconds=int(write_time)))}")
    
    # Create and write the samplesheet
    samplesheet_path = os.path.join(output_dir, 'samplesheet.csv')
    samplesheet_df = pd.DataFrame(samplesheet_data)
    samplesheet_df.to_csv(samplesheet_path, index=False)
    print(f"Samplesheet created at: {samplesheet_path}")
    
    # Calculate and display final statistics
    total_time = time.time() - progress.start_time
    print(f"\nTotal execution time: {str(timedelta(seconds=int(total_time)))}")
    print(f"Processing rate: {total_rows/total_time:.2f} rows/second")
    print("Processing complete")


def process_chunk_parallel(parquet_file_path, chunk_idx, offset, chunksize, total_rows):
    """Process a chunk in a parallel worker process."""
    import pandas as pd  # Re-import in the worker process
    import time  # For timing the chunk processing
    
    # Calculate actual chunk size (might be smaller for the last chunk)
    end_offset = min(offset + chunksize, total_rows)
    actual_size = end_offset - offset
    
    worker_start_time = time.time()
    worker_id = multiprocessing.current_process().name
    
    print(f"Worker {worker_id}: Starting chunk {chunk_idx+1} (rows {offset:,}-{end_offset:,})")
    
    # Read the specific chunk
    chunk = pd.read_parquet(
        parquet_file_path,
        engine='pyarrow',
        filters=[('__index_level_0__', '>=', offset), ('__index_level_0__', '<', end_offset)]
    )
    
    # Initialize dictionaries to store results
    chunk_sample_data = {}
    chunk_sample_labels = {}
    
    # Process each sample in the chunk
    samples_in_chunk = chunk['sample'].unique()
    for sample in samples_in_chunk:
        # Filter data for this sample without making a copy
        sample_chunk = chunk[chunk['sample'] == sample]
        
        # Store the label for this sample
        chunk_sample_labels[sample] = sample_chunk['label'].iloc[0]
        
        # Initialize sample data dictionary if not already done
        if sample not in chunk_sample_data:
            chunk_sample_data[sample] = {}
        
        # Process each seqname in this sample's chunk
        for _, group in sample_chunk.groupby('seqname'):
            seqname = group['seqname'].iloc[0]
            
            # Since each seqname has at most 2 rows, we can compute average directly
            avg_prob = group['prob_class_1'].mean()
            
            # Store new data
            row_data = {
                'rows': len(group),
                'sum': group['prob_class_1'].sum(),
                'prob_class_1': avg_prob
            }
            
            # Copy other columns as needed
            for col in group.columns:
                if col not in ['prob_class_1', 'seqname']:
                    row_data[col] = group[col].iloc[0]
            
            chunk_sample_data[sample][seqname] = row_data
    
    # Log worker completion time
    worker_time = time.time() - worker_start_time
    processing_rate = actual_size / worker_time if worker_time > 0 else 0
    print(f"Worker {worker_id}: Completed chunk {chunk_idx+1} in {worker_time:.2f}s ({processing_rate:.2f} rows/s)")
    
    return chunk_sample_data, chunk_sample_labels


# Keep the old function for reference
def process_chunk(chunk, sample_data, sample_labels):
    """Process a chunk of data efficiently."""
    # Get unique samples in this chunk
    samples_in_chunk = chunk['sample'].unique()
    
    for sample in samples_in_chunk:
        # Filter data for this sample without making a copy
        sample_chunk = chunk[chunk['sample'] == sample]
        
        # Store the label for this sample
        sample_labels[sample] = sample_chunk['label'].iloc[0]
        
        # Initialize sample data dictionary if not already done
        if sample not in sample_data:
            sample_data[sample] = {}
        
        # Process each seqname in this sample's chunk
        for _, group in sample_chunk.groupby('seqname'):
            seqname = group['seqname'].iloc[0]
            
            # Since each seqname has at most 2 rows, we can compute average directly
            avg_prob = group['prob_class_1'].mean()
            
            # Check if we already have data for this seqname
            if seqname in sample_data[sample]:
                # Get existing values
                existing_rows = sample_data[sample][seqname]['rows']
                existing_sum = sample_data[sample][seqname]['sum']
                
                # Update with new data
                new_rows = existing_rows + len(group)
                new_sum = existing_sum + group['prob_class_1'].sum()
                new_avg = new_sum / new_rows
                
                # Store updated values
                sample_data[sample][seqname] = {
                    'rows': new_rows,
                    'sum': new_sum,
                    'prob_class_1': new_avg
                }
                
                # Copy other columns as needed
                for col in group.columns:
                    if col not in ['prob_class_1', 'seqname']:
                        sample_data[sample][seqname][col] = group[col].iloc[0]
            else:
                # Store new data
                row_data = {
                    'rows': len(group),
                    'sum': group['prob_class_1'].sum(),
                    'prob_class_1': avg_prob
                }
                
                # Copy other columns as needed
                for col in group.columns:
                    if col not in ['prob_class_1', 'seqname']:
                        row_data[col] = group[col].iloc[0]
                
                sample_data[sample][seqname] = row_data


if __name__ == "__main__":
    # Needed for Windows multiprocessing
    import concurrent.futures
    split_parquet_by_sample()
