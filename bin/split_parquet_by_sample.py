import os
import click
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import time
from tqdm import tqdm
from tabulate import tabulate

def get_unique_samples(input_file):
    """
    Efficiently extract unique sample names and labels from a parquet file
    without loading the entire dataset into memory.
    
    Args:
        input_file: Path to the input parquet file
        
    Returns:
        List of tuples of (sample_name, label)
    """
    parquet_file = pq.ParquetFile(input_file)
    # Only read the required columns
    sample_table = parquet_file.read(['sampledown', 'label'])
    # Convert to pandas, get unique combinations and convert to list of tuples
    unique_samples = sample_table.to_pandas().drop_duplicates().values.tolist()
    return [(row[0], row[1]) for row in unique_samples]

def process_sample(sample_name, label, input_file, output_dir, verbose=False):
    """
    Process a single sample from the parquet file and save it to a separate file.
    
    Args:
        sample_name: Name of the sample
        label: Label of the sample
        input_file: Path to the input parquet file
        output_dir: Directory to save output files
        verbose: Whether to print additional information
        
    Returns:
        Dict with sample information and file path
    """
    output_filename = f"{sample_name}_{label}.parquet"
    output_file = os.path.join(output_dir, output_filename)
    
    # Use pyarrow.parquet.read_table with filters parameter for better memory efficiency
    filters = [('sampledown', '=', sample_name)]
    sample_table = pq.read_table(input_file, filters=filters)
    
    # Write the subset to a new parquet file
    pq.write_table(sample_table, output_file)
    
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
@click.option('--verbose', is_flag=True, help='Print verbose output')
def main(input, output_dir, samplesheet, verbose):
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
    
    # Get the unique sample names and their labels
    click.echo(f"Reading sample information from '{input}'...")
    sample_tuples = get_unique_samples(input)
    
    click.echo(f"Found {len(sample_tuples)} unique samples. Starting processing...")
    
    # Process the samples sequentially with a progress bar
    results = []
    for sample_name, label in tqdm(sample_tuples, desc="Processing samples"):
        result = process_sample(sample_name, label, input, output_dir, verbose)
        results.append(result)
        # Explicitly call garbage collection to free memory if needed
        if verbose:
            click.echo(f"Processed sample: {sample_name} with label: {label}")
            
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
