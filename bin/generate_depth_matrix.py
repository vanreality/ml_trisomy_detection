import pandas as pd
import numpy as np
import click
import pyarrow.parquet as pq
import pyarrow as pa
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm
import logging
import os
import sys
from typing import List, Dict, Tuple, Optional

# Configure logging with rich
console = Console(width=280)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

def read_metadata_csv(meta_file: str) -> pd.DataFrame:
    """
    Read and validate the metadata CSV file.
    
    Args:
        meta_file: Path to the metadata CSV file
        
    Returns:
        A DataFrame containing the metadata
        
    Raises:
        ValueError: If the CSV file is missing required columns or is empty
    """
    try:
        logger.info(f"Reading metadata from: {meta_file}")
        meta_df = pd.read_csv(meta_file)
        
        # Validate required columns
        required_columns = ['sample', 'label', 'parquet_file_path']
        missing_columns = [col for col in required_columns if col not in meta_df.columns]
        
        if missing_columns:
            raise ValueError(f"Metadata CSV missing required columns: {', '.join(missing_columns)}")
        
        if meta_df.empty:
            raise ValueError("Metadata CSV is empty")
        
        # Validate that parquet files exist
        for idx, row in meta_df.iterrows():
            if not os.path.exists(row['parquet_file_path']):
                raise ValueError(f"Parquet file not found: {row['parquet_file_path']}")
        
        return meta_df
    
    except Exception as e:
        logger.error(f"Error reading metadata CSV: {str(e)}")
        raise

def read_first_sample(parquet_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Read the first sample parquet file to initialize the matrices.
    
    Args:
        parquet_path: Path to the parquet file for the first sample
        
    Returns:
        A tuple containing the initialized DataFrame and a list of metadata column names
        
    Raises:
        ValueError: If the parquet file cannot be read or has unexpected structure
    """
    try:
        logger.info(f"Reading first sample from: {parquet_path}")
        
        # Read the parquet file
        df = pq.read_table(parquet_path, columns=['chr_dmr', 'start_dmr', 'end_dmr', 'chr', 'start', 'end', 'base']).to_pandas()
        
        return df
    
    except Exception as e:
        logger.error(f"Error reading first sample: {str(e)}")
        raise

def append_sample_columns(
    depth_dict: dict, 
    parquet_path: str, 
    sample_name: str, 
    label: str, 
    depth_column: str
) -> dict:
    """
    Append columns from a sample to existing matrices.
    
    Args:
        depth_dict: The dictionary containing the depth matrices
        parquet_path: Path to the parquet file for the current sample
        sample_name: The name of the current sample
        label: The label of the current sample
        depth_column: The column name of the depth values
        
    Returns:
        Dictionary containing the updated depth matrices
        
    Raises:
        ValueError: If the parquet file has unexpected structure
    """
    try:
        logger.info(f"Processing sample: {sample_name} ({label})")
        
        # Read only necessary columns from parquet
        depth_list = pq.read_table(parquet_path, columns=[depth_column])[depth_column].to_pylist()
        
        # Create column name for this sample
        col_name = f"{sample_name}_{label}"
        
        # Create depth matrices for each type
        depth_dict[col_name] = depth_list
            
        return depth_dict
    
    except Exception as e:
        logger.error(f"Error processing sample {sample_name}: {str(e)}")
        raise

def generate_depth_matrices(meta_df: pd.DataFrame, depth_column: str) -> pd.DataFrame:
    """
    Generate depth matrices for all samples.
    
    Args:
        meta_df: DataFrame containing metadata information
        
    Returns:
        Dictionary containing matrices for each depth type
        
    Raises:
        ValueError: If any sample cannot be processed
    """
    try:
        # Initialize with first sample
        first_sample = meta_df.iloc[0]
        first_df = read_first_sample(first_sample['parquet_file_path'])
        
        # Dictionary to store depth matrices
        depth_dict = {}
        
        # Process all samples
        for idx, row in tqdm(meta_df.iterrows(), desc="Processing samples"):
            # Append columns for the current sample
            depth_dict = append_sample_columns(
                depth_dict,
                row['parquet_file_path'],
                row['sample'],
                row['label'],
                depth_column
            )
            
        depth_df = pd.DataFrame(depth_dict)
        concated_df = pd.concat([first_df, depth_df], axis=1)
        return concated_df
    
    except Exception as e:
        logger.error(f"Error generating depth matrices: {str(e)}")
        raise

@click.command()
@click.option('--meta', required=True, type=click.Path(exists=True), help='Path to the metadata CSV file')
@click.option('--output', required=True, type=str, help='Prefix for output files')
def main(meta: str, output: str):
    """
    Generate depth matrices from depth statistics parquet files.
    
    This script reads depth statistics from parquet files specified in a metadata CSV,
    merges them into matrices, and outputs three parquet files for different depth metrics.
    """
    try:
        # Read metadata CSV
        meta_df = read_metadata_csv(meta)
        
        # Generate matrices
        logger.info("Generating depth matrices...")
        for depth_column in ['raw_depth', 'prob_weighted_depth', 'normalized_depth']:
            logger.info(f"Generating {depth_column} matrix...")
            matrix_df = generate_depth_matrices(meta_df, depth_column)
            output_file = f"{output}_{depth_column}_matrix.parquet"
            logger.info(f"Writing {depth_column} matrix to: {output_file}")
            table = pa.Table.from_pandas(matrix_df)
            pq.write_table(table, output_file)
        
        logger.info("Depth matrix generation completed successfully")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        logger.exception("An error occurred during processing")
        sys.exit(1)

if __name__ == '__main__':
    main()
