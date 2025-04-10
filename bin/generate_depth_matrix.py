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

# Configure logging with rich - with wider console width for better traceback readability
console = Console(width=120)  # Increase console width for better traceback formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True, tracebacks_width=120)]  # Explicitly set tracebacks width
)
logger = logging.getLogger(__name__)

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

def generate_depth_matrices(meta_df, bed_df, chunk_size):
    """
    Generate depth matrices from all samples.
    
    Args:
        meta_df (pd.DataFrame): DataFrame containing sample metadata.
        bed_df (pd.DataFrame): DataFrame containing DMR regions.
        chunk_size (int): Number of rows to read at a time.
        
    Returns:
        tuple: A tuple containing three DataFrames (raw_matrix, prob_matrix, norm_matrix).
    """
    logger.info("Generating depth matrices")
    
    # Initialize dictionaries to store matrices for each depth type
    raw_data = {'sample': [], 'label': []}
    prob_data = {'sample': [], 'label': []}
    norm_data = {'sample': [], 'label': []}
    
    # Keep track of all multi-index keys
    all_indices = set()
    
    # Track sample data for each depth type
    sample_dfs = []
    
    # Process each sample
    for _, sample_row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Processing samples"):
        sample_id, label, multi_index_df = process_sample(sample_row, bed_df, chunk_size)
        
        if multi_index_df is not None:
            # Store sample information
            raw_data['sample'].append(sample_id)
            raw_data['label'].append(label)
            prob_data['sample'].append(sample_id)
            prob_data['label'].append(label)
            norm_data['sample'].append(sample_id)
            norm_data['label'].append(label)
            
            # Extract individual depth DataFrames
            raw_df = multi_index_df['raw_depth'].unstack()
            prob_df = multi_index_df['prob_weighted_depth'].unstack()
            norm_df = multi_index_df['normalized_depth'].unstack()
            
            # Save the indices
            all_indices.update(raw_df.index)
            
            # Store the DataFrames for this sample
            sample_dfs.append((sample_id, raw_df, prob_df, norm_df))
            
            # Free memory
            del multi_index_df
            gc.collect()
    
    # Create empty matrices with all indices
    all_indices = sorted(list(all_indices))
    logger.info(f"Total unique region-base pairs: {len(all_indices)}")
    
    # Create the matrices by merging all samples
    raw_matrix = pd.DataFrame(index=all_indices)
    prob_matrix = pd.DataFrame(index=all_indices)
    norm_matrix = pd.DataFrame(index=all_indices)
    
    # Add sample columns to the matrices
    for sample_id, raw_df, prob_df, norm_df in tqdm(sample_dfs, desc="Building matrices"):
        raw_matrix[sample_id] = raw_df
        prob_matrix[sample_id] = prob_df
        norm_matrix[sample_id] = norm_df
    
    # Reset index to convert to regular columns
    raw_matrix = raw_matrix.reset_index()
    prob_matrix = prob_matrix.reset_index()
    norm_matrix = norm_matrix.reset_index()
    
    # Rename the index column
    raw_matrix = raw_matrix.rename(columns={'index': 'region_base'})
    prob_matrix = prob_matrix.rename(columns={'index': 'region_base'})
    norm_matrix = norm_matrix.rename(columns={'index': 'region_base'})
    
    # Add sample and label metadata
    raw_df_meta = pd.DataFrame({
        'sample': raw_data['sample'],
        'label': raw_data['label']
    })
    
    prob_df_meta = pd.DataFrame({
        'sample': prob_data['sample'],
        'label': prob_data['label']
    })
    
    norm_df_meta = pd.DataFrame({
        'sample': norm_data['sample'],
        'label': norm_data['label']
    })
    
    # Drop columns with NA values
    raw_matrix = raw_matrix.dropna(axis=1)
    prob_matrix = prob_matrix.dropna(axis=1)
    norm_matrix = norm_matrix.dropna(axis=1)
    
    logger.info(f"Raw depth matrix shape: {raw_matrix.shape}")
    logger.info(f"Probability-weighted depth matrix shape: {prob_matrix.shape}")
    logger.info(f"Normalized depth matrix shape: {norm_matrix.shape}")
    
    return raw_df_meta, raw_matrix, prob_df_meta, prob_matrix, norm_df_meta, norm_matrix

@click.command()
@click.option('--meta', required=True, type=click.Path(exists=True), help='Path to the meta CSV file with sample information')
@click.option('--bed', required=True, type=click.Path(exists=True), help='Path to the BED file with DMR regions')
@click.option('--output', required=True, type=str, help='Prefix for output files')
@click.option('--chunk-size', default=100000, type=int, help='Number of rows to read at a time from CSV files')
def main(meta, bed, output, chunk_size):
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
        console.print(f"[bold green]Starting depth matrix generation[/bold green]")
        
        # Read input files
        meta_df = read_meta_file(meta)
        bed_df = read_bed_file(bed)
        
        console.print(f"[green]Found {len(meta_df)} samples in meta file[/green]")
        console.print(f"[green]Found {len(bed_df)} DMR regions in BED file[/green]")
        
        # Generate depth matrices
        raw_meta, raw_matrix, prob_meta, prob_matrix, norm_meta, norm_matrix = generate_depth_matrices(
            meta_df, bed_df, chunk_size
        )
        
        # Prepare output file paths
        raw_output = f"{output}_raw_depth.parquet"
        prob_output = f"{output}_prob_weighted_depth.parquet"
        norm_output = f"{output}_normalized_depth.parquet"
        meta_output = f"{output}_sample_meta.parquet"
        
        # Save matrices to parquet files
        console.print("[bold blue]Saving matrices to parquet files...[/bold blue]")
        
        # Save sample metadata
        raw_meta.to_parquet(meta_output)
        console.print(f"[blue]Sample metadata saved to: {meta_output}[/blue]")
        
        # Save depth matrices
        raw_matrix.to_parquet(raw_output)
        prob_matrix.to_parquet(prob_output)
        norm_matrix.to_parquet(norm_output)
        
        console.print(f"[bold green]Raw depth matrix saved to: {raw_output}[/bold green]")
        console.print(f"[bold green]Probability-weighted depth matrix saved to: {prob_output}[/bold green]")
        console.print(f"[bold green]Normalized depth matrix saved to: {norm_output}[/bold green]")
        
        console.print("[bold green]Depth matrix generation completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        logger.exception("An error occurred during processing")
        sys.exit(1)

if __name__ == '__main__':
    main()
