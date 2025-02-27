import os
import click
import pandas as pd
import pyarrow.parquet as pq
from multiprocessing import Pool
from functools import partial
import time
from tabulate import tabulate

def process_sample(sample_info, input_file, output_dir=".", verbose=False):
    """
    Process a single sample from the parquet file and save it to a separate file.
    
    Args:
        sample_info: Tuple of (sample_name, label)
        input_file: Path to the input parquet file
        output_dir: Directory to save output files
        verbose: Whether to print progress messages
        
    Returns:
        Dict with sample information and file path
    """
    sample_name, label = sample_info
    output_filename = f"{sample_name}_{label}.parquet"
    output_file = os.path.join(output_dir, output_filename)
    
    if verbose:
        print(f"Processing sample: {sample_name} with label: {label}")
    
    # Read only the subset of data matching this sample
    filters = [('sampledown', '=', sample_name)]
    sample_df = pd.read_parquet(input_file, filters=filters)
    
    # Write the subset to a new parquet file
    sample_df.to_parquet(output_file, index=False)
    
    # Return sample information with absolute file path
    return {
        'sample': sample_name,
        'label': label,
        'parquet': os.path.abspath(output_file)
    }

@click.command()
@click.option('--input', required=True, help='Path to the input parquet file')
@click.option('--output-dir', default='.', help='Directory to save output files')
@click.option('--samplesheet', default='samplesheet.csv', help='Path to output samplesheet CSV')
@click.option('--ncpus', default=os.cpu_count(), type=int, help='Number of CPU cores to use')
@click.option('--verbose', is_flag=True, help='Print verbose output')
def main(input, output_dir, samplesheet, ncpus, verbose):
    """
    Split a parquet file into multiple files based on the 'sampledown' column.
    Each output file will be named {sampledown}_{label}.parquet.
    Generates a samplesheet.csv with sample, label, and parquet file paths.
    """
    start_time = time.time()
    
    # Check if the input file exists
    if not os.path.exists(input):
        click.echo(f"Error: Input file '{input}' does not exist", err=True)
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        click.echo(f"Created output directory: {output_dir}")
    
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
    process_fn = partial(process_sample, input_file=input, output_dir=output_dir, verbose=verbose)
    with Pool(processes=ncpus) as pool:
        results = pool.map(process_fn, sample_tuples)

    # Create a DataFrame from the results
    samplesheet_df = pd.DataFrame(results)
    
    # Write the samplesheet to a CSV file
    samplesheet_path = os.path.join(output_dir, samplesheet)
    samplesheet_df.to_csv(samplesheet_path, index=False)
    
    elapsed_time = time.time() - start_time
    click.echo(f"Processing completed in {elapsed_time:.2f} seconds")
    click.echo(f"Wrote {len(results)} samples to individual parquet files")
    click.echo(f"Generated samplesheet: {os.path.abspath(samplesheet_path)}")
    
    # Display sample of the samplesheet
    if verbose and not samplesheet_df.empty:
        sample_rows = min(5, len(samplesheet_df))
        click.echo("\nSample of samplesheet.csv:")
        click.echo(tabulate(samplesheet_df.head(sample_rows), headers='keys', tablefmt='pretty'))

if __name__ == "__main__":
    main()
