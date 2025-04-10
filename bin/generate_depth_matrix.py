#!/usr/bin/env python3
"""
Generate depth matrices from CSV files produced by stat_depth.py.

This script processes CSV files containing depth information for multiple samples,
and generates matrices for raw, probability-weighted, and normalized depth values.
The output is saved in parquet format for efficient storage and future processing.
"""

import os
import pandas as pd
import numpy as np
import click
from rich.console import Console
from rich.logging import RichHandler
from rich import progress
import logging
from tqdm import tqdm
import sys
import traceback
import gc
from scipy import sparse
import warnings
from pathlib import Path

# Increase display width for pandas
pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 100)

# Configure logging with rich - increase width for better readability
console = Console(width=120)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
)
logger = logging.getLogger(__name__)

# Suppress pandas PerformanceWarning
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

def read_meta_file(meta_file_path):
    """
    Read the metadata file containing sample information.
    
    Args:
        meta_file_path (str): Path to the CSV meta file.
        
    Returns:
        pd.DataFrame: DataFrame containing sample metadata.
        
    Raises:
        ValueError: If required columns are missing from the meta file.
    """
    logger.info(f"Reading meta file: {meta_file_path}")
    try:
        meta_df = pd.read_csv(meta_file_path)
        # Check required columns
        required_cols = ['sample', 'label', 'csv_file_path']
        missing_cols = [col for col in required_cols if col not in meta_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in meta file: {', '.join(missing_cols)}")
        return meta_df
    except Exception as e:
        logger.error(f"Error reading meta file: {str(e)}")
        raise

def read_bed_file(bed_file_path):
    """
    Read the BED file containing DMR regions.
    
    Args:
        bed_file_path (str): Path to the BED file.
        
    Returns:
        pd.DataFrame: DataFrame containing DMR regions.
        
    Raises:
        ValueError: If BED file format is invalid.
    """
    logger.info(f"Reading BED file: {bed_file_path}")
    try:
        # BED format typically has chromosome, start, end columns
        bed_df = pd.read_csv(bed_file_path, sep='\t', header=None)
        if bed_df.shape[1] < 3:
            raise ValueError("BED file must have at least 3 columns (chromosome, start, end)")
        
        # Rename columns to match expected format
        columns = ['chr_dmr', 'start_dmr', 'end_dmr']
        bed_df = bed_df.iloc[:, :3]
        bed_df.columns = columns
        
        return bed_df
    except Exception as e:
        logger.error(f"Error reading BED file: {str(e)}")
        raise

def read_depth_file_in_chunks(csv_file_path, bed_df, chunk_size=100000):
    """
    Read a depth CSV file in chunks to handle large files efficiently.
    
    Args:
        csv_file_path (str): Path to the depth CSV file.
        bed_df (pd.DataFrame): DataFrame containing DMR regions.
        chunk_size (int, optional): Number of rows to read at a time. Defaults to 100000.
        
    Returns:
        pd.DataFrame: DataFrame containing depth information for the regions in bed_df.
        
    Raises:
        ValueError: If required columns are missing from the depth file.
    """
    logger.info(f"Reading depth file: {csv_file_path}")
    try:
        # Create a reader that will read the file in chunks
        reader = pd.read_csv(csv_file_path, chunksize=chunk_size)
        
        # Initialize an empty list to store filtered chunks
        filtered_chunks = []
        
        # Required columns
        required_cols = ['index', 'chr_dmr', 'start_dmr', 'end_dmr', 'chr', 'start', 'end', 
                          'base', 'raw_depth', 'prob_weighted_depth', 'normalized_depth']
        
        # Process each chunk
        for chunk_idx, chunk in enumerate(tqdm(reader, desc="Reading chunks")):
            # Check for required columns in the first chunk
            if chunk_idx == 0:
                missing_cols = [col for col in required_cols if col not in chunk.columns]
                if missing_cols:
                    raise ValueError(f"Missing required columns in depth file: {', '.join(missing_cols)}")
            
            # Convert to float32 to reduce memory usage
            for col in ['raw_depth', 'prob_weighted_depth', 'normalized_depth']:
                if col in chunk.columns:
                    chunk[col] = chunk[col].astype('float32')
            
            # Filter the chunk to include only rows matching DMR regions
            filtered_chunk = pd.merge(
                chunk, 
                bed_df, 
                on=['chr_dmr', 'start_dmr', 'end_dmr'], 
                how='inner'
            )
            
            if not filtered_chunk.empty:
                filtered_chunks.append(filtered_chunk)
            
            # Free memory
            del chunk
            gc.collect()
        
        # Combine all filtered chunks
        if filtered_chunks:
            result_df = pd.concat(filtered_chunks, ignore_index=True)
            del filtered_chunks
            gc.collect()
            return result_df
        else:
            logger.warning(f"No matching DMR regions found in {csv_file_path}")
            # Return an empty DataFrame with the required columns
            return pd.DataFrame(columns=required_cols)
    
    except Exception as e:
        logger.error(f"Error reading depth file {csv_file_path}: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

def build_multi_index(df):
    """
    Build multi-index for depth data.
    
    Args:
        df (pd.DataFrame): DataFrame containing depth information.
        
    Returns:
        pd.DataFrame: DataFrame with multi-index.
    """
    logger.info("Building multi-index")
    try:
        # Create region-level indexer: {chr_dmr}_{start_dmr}_{end_dmr}
        df['region_index'] = df['chr_dmr'] + '_' + df['start_dmr'].astype(str) + '_' + df['end_dmr'].astype(str)
        
        # Create base-level indexer: {chr}_{start}_{end}_{base}
        df['base_index'] = df['chr'] + '_' + df['start'].astype(str) + '_' + df['end'].astype(str) + '_' + df['base']
        
        # Create the multi-index DataFrame
        multi_index_df = df.set_index(['region_index', 'base_index'])
        
        # Keep only the depth columns
        depth_cols = ['raw_depth', 'prob_weighted_depth', 'normalized_depth']
        multi_index_df = multi_index_df[depth_cols]
        
        return multi_index_df
    
    except Exception as e:
        logger.error(f"Error building multi-index: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

def process_sample(sample_row, bed_df, chunk_size):
    """
    Process a single sample to extract depth information.
    
    Args:
        sample_row (pd.Series): Row from the meta DataFrame for a single sample.
        bed_df (pd.DataFrame): DataFrame containing DMR regions.
        chunk_size (int): Number of rows to read at a time.
        
    Returns:
        tuple: A tuple containing (sample_id, label, multi_index_df).
    """
    sample_id = sample_row['sample']
    label = sample_row['label']
    csv_file_path = sample_row['csv_file_path']
    
    logger.info(f"Processing sample: {sample_id}")
    
    try:
        # Check if the CSV file exists
        if not os.path.isfile(csv_file_path):
            logger.warning(f"CSV file for sample {sample_id} not found: {csv_file_path}")
            return sample_id, label, None
        
        # Read the depth file in chunks and filter for DMR regions
        depth_df = read_depth_file_in_chunks(csv_file_path, bed_df, chunk_size)
        
        if depth_df.empty:
            logger.warning(f"No depth data found for sample {sample_id}")
            return sample_id, label, None
        
        # Build multi-index
        multi_index_df = build_multi_index(depth_df)
        
        # Free memory
        del depth_df
        gc.collect()
        
        return sample_id, label, multi_index_df
    
    except Exception as e:
        logger.error(f"Error processing sample {sample_id}: {str(e)}")
        logger.debug(traceback.format_exc())
        return sample_id, label, None

def process_matrices_in_batches(output_prefix, indices_map, sample_data, temp_dir):
    """
    Process matrices in batches to avoid memory issues.
    
    Args:
        output_prefix (str): Prefix for output files.
        indices_map (dict): Mapping from index to position in the matrix.
        sample_data (list): List of tuples (sample_id, label, raw_df, prob_df, norm_df).
        temp_dir (str): Directory for temporary files.
        
    Returns:
        tuple: Tuple of DataFrames for raw, prob, and norm matrices.
    """
    logger.info("Processing matrices in batches")
    
    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    # Count total number of regions
    n_regions = len(indices_map)
    n_samples = len(sample_data)
    
    logger.info(f"Total regions: {n_regions}, Total samples: {n_samples}")
    
    # Create sparse matrices for each depth type (csr_matrix for efficiency)
    raw_matrix = sparse.lil_matrix((n_regions, n_samples), dtype='float32')
    prob_matrix = sparse.lil_matrix((n_regions, n_samples), dtype='float32')
    norm_matrix = sparse.lil_matrix((n_regions, n_samples), dtype='float32')
    
    # Get list of sample IDs and labels
    sample_ids = [s[0] for s in sample_data]
    labels = [s[1] for s in sample_data]
    
    # Process each sample and add its data to the sparse matrices
    for i, (sample_id, label, raw_series, prob_series, norm_series) in enumerate(
            tqdm(sample_data, desc="Building sparse matrices")):
        
        # Add non-zero values to the sparse matrices
        for idx, val in raw_series.items():
            if idx in indices_map and not pd.isna(val):
                raw_matrix[indices_map[idx], i] = val
        
        for idx, val in prob_series.items():
            if idx in indices_map and not pd.isna(val):
                prob_matrix[indices_map[idx], i] = val
        
        for idx, val in norm_series.items():
            if idx in indices_map and not pd.isna(val):
                norm_matrix[indices_map[idx], i] = val
    
    # Convert to CSR format for efficient operations
    raw_matrix = raw_matrix.tocsr()
    prob_matrix = prob_matrix.tocsr()
    norm_matrix = norm_matrix.tocsr()
    
    # Create metadata DataFrame
    meta_df = pd.DataFrame({
        'sample': sample_ids,
        'label': labels
    })
    
    # Get the region indices as a list (preserve order)
    region_indices = list(indices_map.keys())
    
    # Save the matrices to parquet files
    meta_output = f"{output_prefix}_sample_meta.parquet"
    meta_df.to_parquet(meta_output)
    logger.info(f"Sample metadata saved to: {meta_output}")
    
    # Function to save a sparse matrix in a memory-efficient way
    def save_sparse_matrix(matrix, indices, columns, filepath):
        # Convert sparse matrix to COO format for efficient iteration
        coo = matrix.tocoo()
        
        # Create temporary CSV file for the matrix data
        temp_csv = f"{temp_dir}/temp_matrix.csv"
        
        # Write data to CSV in chunks
        with open(temp_csv, 'w') as f:
            # Write header
            f.write("region_base," + ",".join(columns) + "\n")
            
            # Process in batches
            batch_size = 10000
            batch_rows = []
            
            for i, j, v in zip(coo.row, coo.col, coo.data):
                # Skip NaN and zero values
                if pd.isna(v) or v == 0:
                    continue
                
                row_data = [indices[i]] + ["0"] * len(columns)
                row_data[j+1] = str(v)
                batch_rows.append(",".join(row_data))
                
                if len(batch_rows) >= batch_size:
                    f.write("\n".join(batch_rows) + "\n")
                    batch_rows = []
            
            # Write remaining rows
            if batch_rows:
                f.write("\n".join(batch_rows) + "\n")
        
        # Read the CSV and convert to parquet
        logger.info(f"Reading temporary CSV file for {filepath}")
        matrix_df = pd.read_csv(temp_csv)
        
        # Save as parquet
        logger.info(f"Saving to parquet: {filepath}")
        matrix_df.to_parquet(filepath)
        
        # Remove temporary CSV
        os.remove(temp_csv)
    
    # Save matrices
    raw_output = f"{output_prefix}_raw_depth.parquet"
    prob_output = f"{output_prefix}_prob_weighted_depth.parquet"
    norm_output = f"{output_prefix}_normalized_depth.parquet"
    
    logger.info("Saving raw depth matrix")
    save_sparse_matrix(raw_matrix, region_indices, sample_ids, raw_output)
    logger.info("Saving probability-weighted depth matrix")
    save_sparse_matrix(prob_matrix, region_indices, sample_ids, prob_output)
    logger.info("Saving normalized depth matrix")
    save_sparse_matrix(norm_matrix, region_indices, sample_ids, norm_output)
    
    logger.info(f"Raw depth matrix shape: {raw_matrix.shape}")
    logger.info(f"Probability-weighted depth matrix shape: {prob_matrix.shape}")
    logger.info(f"Normalized depth matrix shape: {norm_matrix.shape}")
    
    # Clean up temporary directory
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
    
    return meta_df, raw_matrix, prob_matrix, norm_matrix

def generate_depth_matrices(meta_df, bed_df, output_prefix, chunk_size, temp_dir=None):
    """
    Generate depth matrices from all samples.
    
    Args:
        meta_df (pd.DataFrame): DataFrame containing sample metadata.
        bed_df (pd.DataFrame): DataFrame containing DMR regions.
        output_prefix (str): Prefix for output files.
        chunk_size (int): Number of rows to read at a time.
        temp_dir (str, optional): Directory for temporary files. Defaults to a subdirectory of output_prefix.
        
    Returns:
        tuple: A tuple containing four sparse matrices (meta_df, raw_matrix, prob_matrix, norm_matrix).
    """
    logger.info("Generating depth matrices")
    
    # Create a temp directory if none provided
    if temp_dir is None:
        temp_dir = f"{os.path.dirname(output_prefix)}/temp_{os.path.basename(output_prefix)}"
    
    # Create the temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)
    
    # Keep track of all region-base pairs across all samples
    all_indices = set()
    
    # Process each sample to get index and depth information
    sample_data = []
    
    # Process each sample in smaller batches to reduce memory usage
    sample_batch_size = 5  # Process 5 samples at a time
    total_batches = (len(meta_df) + sample_batch_size - 1) // sample_batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * sample_batch_size
        end_idx = min((batch_idx + 1) * sample_batch_size, len(meta_df))
        
        logger.info(f"Processing sample batch {batch_idx+1}/{total_batches} (samples {start_idx+1}-{end_idx})")
        
        batch_samples = []
        
        # Process samples in this batch
        for i in range(start_idx, end_idx):
            sample_row = meta_df.iloc[i]
            sample_id, label, multi_index_df = process_sample(sample_row, bed_df, chunk_size)
            
            if multi_index_df is not None:
                # Extract individual depth series
                raw_series = multi_index_df['raw_depth']
                prob_series = multi_index_df['prob_weighted_depth']
                norm_series = multi_index_df['normalized_depth']
                
                # Update all indices
                all_indices.update(raw_series.index)
                
                # Store the data for this sample
                batch_samples.append((sample_id, label, raw_series, prob_series, norm_series))
                
                # Free memory
                del multi_index_df
                gc.collect()
        
        # Add batch samples to sample_data
        sample_data.extend(batch_samples)
        
        # Free memory
        del batch_samples
        gc.collect()
    
    # If no samples were processed successfully, exit
    if not sample_data:
        logger.error("No samples were processed successfully. Exiting.")
        return None, None, None, None
    
    # Create mapping from index to position in the matrix
    logger.info(f"Total unique region-base pairs: {len(all_indices)}")
    indices_map = {idx: i for i, idx in enumerate(sorted(all_indices))}
    
    # Process matrices in batches
    return process_matrices_in_batches(output_prefix, indices_map, sample_data, temp_dir)

@click.command()
@click.option('--meta', required=True, type=click.Path(exists=True), help='Path to the meta CSV file with sample information')
@click.option('--bed', required=True, type=click.Path(exists=True), help='Path to the BED file with DMR regions')
@click.option('--output', required=True, type=str, help='Prefix for output files')
@click.option('--chunk-size', default=100000, type=int, help='Number of rows to read at a time from CSV files')
@click.option('--temp-dir', type=str, help='Directory for temporary files (default: alongside output)')
def main(meta, bed, output, chunk_size, temp_dir):
    """
    Generate depth matrices from CSV files produced by stat_depth.py.
    
    This tool follows these steps:
    1. Reads a meta file with sample information and paths to CSV files
    2. Reads a BED file with DMR regions
    3. For each sample, extracts depth information for the specified regions
    4. Builds multi-index DataFrames for each sample
    5. Merges all samples into matrices for raw, probability-weighted, and normalized depth
    6. Outputs the matrices in parquet format
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        console.print(f"[bold green]Starting depth matrix generation[/bold green]")
        
        # Read input files
        meta_df = read_meta_file(meta)
        bed_df = read_bed_file(bed)
        
        console.print(f"[green]Found {len(meta_df)} samples in meta file[/green]")
        console.print(f"[green]Found {len(bed_df)} DMR regions in BED file[/green]")
        
        # Generate depth matrices
        generate_depth_matrices(
            meta_df, bed_df, output, chunk_size, temp_dir
        )
        
        # Output file paths
        raw_output = f"{output}_raw_depth.parquet"
        prob_output = f"{output}_prob_weighted_depth.parquet"
        norm_output = f"{output}_normalized_depth.parquet"
        meta_output = f"{output}_sample_meta.parquet"
        
        # Check if output files were created
        missing_files = []
        for file_path in [raw_output, prob_output, norm_output, meta_output]:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            console.print(f"[bold yellow]Warning: The following output files were not created: {', '.join(missing_files)}[/bold yellow]")
        else:
            console.print("[bold green]Depth matrix generation completed successfully![/bold green]")
            console.print(f"[bold green]Output files:[/bold green]")
            console.print(f"[green]Sample metadata: {meta_output}[/green]")
            console.print(f"[green]Raw depth matrix: {raw_output}[/green]")
            console.print(f"[green]Probability-weighted depth matrix: {prob_output}[/green]")
            console.print(f"[green]Normalized depth matrix: {norm_output}[/green]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        # Print full traceback for better debugging
        console.print_exception(show_locals=True, width=120)
        sys.exit(1)

if __name__ == '__main__':
    main()
