#!/usr/bin/env python
import os
import click
import pandas as pd
import pyarrow.parquet as pq
from multiprocessing import Pool
from functools import partial
import time
from tabulate import tabulate

def process_sample(sample_info, input_file, verbose=False):
    """
    Process a single sample from the parquet file and save it to a separate file.
    
    Args:
        sample_info: Tuple of (sample_name, label)
        input_file: Path to the input parquet file
        verbose: Whether to print progress messages
        
    Returns:
        Dict with sample information and row count
    """
    sample_name, label = sample_info
    output_file = f"{sample_name}_{label}.parquet"
    
    if verbose:
        print(f"Processing sample: {sample_name} with label: {label}")
    
    # Read only the subset of data matching this sample
    filters = [('sampledown', '=', sample_name)]
    sample_df = pd.read_parquet(input_file, filters=filters)
    
    # Write the subset to a new parquet file
    sample_df.to_parquet(output_file, index=False)
    
    return {
        'sample': sample_name,
        'label': label,
        'output_file': output_file,
        'nrow': len(sample_df)
    }

@click.command()
@click.option('--input', required=True, help='Path to the input parquet file')
@click.option('--ncpus', default=os.cpu_count(), type=int, help='Number of CPU cores to use')
@click.option('--verbose', is_flag=True, help='Print verbose output')
def main(input, ncpus, verbose):
    """
    Split a parquet file into multiple files based on the 'sampledown' column.
    Each output file will be named {sampledown}_{label}.parquet
    """
    start_time = time.time()
    
    # Check if the input file exists
    if not os.path.exists(input):
        click.echo(f"Error: Input file '{input}' does not exist", err=True)
        return
    
    # Read the unique sample names and their labels
    click.echo(f"Reading sample information from '{input}'...")
    parquet_file = pq.ParquetFile(input)
    table = parquet_file.read(['sampledown', 'label'])
    df_samples = table.to_pandas()
    unique_samples = df_samples[['sampledown', 'label']].drop_duplicates().values.tolist()
    
    # Convert to list of tuples
    sample_tuples = [(row[0], row[1]) for row in unique_samples]
    
    click.echo(f"Found {len(sample_tuples)} unique samples. Starting processing with {ncpus} workers...")
    
    # Process the samples in parallel
    process_fn = partial(process_sample, input_file=input, verbose=verbose)
    with Pool(processes=ncpus) as pool:
        results = pool.map(process_fn, sample_tuples)
    
    # Prepare summary data
    summary_data = []
    total_rows = 0
    
    for result in results:
        summary_data.append([
            result['sample'],
            result['label'],
            result['output_file'],
            result['nrow']
        ])
        total_rows += result['nrow']
    
    # Sort by sample name for consistent output
    summary_data.sort(key=lambda x: x[0])
    
    # Print summary table
    click.echo("\nSummary of output files:")
    headers = ["Sample", "Label", "Output File", "Row Count"]
    click.echo(tabulate(summary_data, headers=headers, tablefmt="grid"))
    
    click.echo(f"\nTotal rows processed: {total_rows}")
    click.echo(f"Processing completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
