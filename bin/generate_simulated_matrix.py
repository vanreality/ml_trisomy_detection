import pandas as pd
import numpy as np
import sys
import click
from multiprocessing import Pool
from functools import partial
from rich.console import Console
from rich.progress import Progress
from tqdm import tqdm
import random
import os

# Set up Rich console
console = Console()

def read_prob_file(prob_file_path):
    """Reads a probability file into a DataFrame.
    
    Args:
        prob_file_path (str): Path to the probability CSV file.
        
    Returns:
        pd.DataFrame: DataFrame containing the probability data.
        
    Raises:
        ValueError: If there's an error reading the probability file.
    """
    try:
        df = pd.read_csv(prob_file_path)
        df['prob_class_1'] = df['prob_class_1'].astype(float)
        # Ensure chr column is string type
        df['chr'] = df['chr'].astype(str)
        return df
    except ValueError as e:
        raise ValueError(f"Error reading probability file {prob_file_path}") from e
    except FileNotFoundError:
        raise FileNotFoundError(f"Probability file not found: {prob_file_path}")

def read_bed_file(bed_file):
    """Reads a BED file containing selected regions.
    
    Args:
        bed_file (str): Path to the BED file.
        
    Returns:
        pd.DataFrame: DataFrame containing the regions.
        
    Raises:
        ValueError: If there's an error reading the BED file.
    """
    try:
        return pd.read_csv(bed_file, sep='\t', header=None, usecols=[0, 1, 2], names=['chr', 'start', 'end'])
    except ValueError as e:
        raise ValueError(f"Error reading BED file {bed_file}") from e
    except FileNotFoundError:
        raise FileNotFoundError(f"BED file not found: {bed_file}")

def filter_sites_by_regions(prob_df, regions_df):
    """Filters probability sites that fall within specified regions.
    
    Args:
        prob_df (pd.DataFrame): DataFrame containing probability data.
        regions_df (pd.DataFrame): DataFrame containing regions to filter by.
        
    Returns:
        pd.DataFrame: Filtered probability DataFrame.
    """
    prob_df['chr'] = prob_df['chr'].astype(str)
    regions_df['chr'] = regions_df['chr'].astype(str)
    
    results = []
    common_chroms = set(prob_df['chr'].unique()) & set(regions_df['chr'].unique())
    
    for chrom in common_chroms:
        prob_chr = prob_df[prob_df['chr'] == chrom]
        regions_chr = regions_df[regions_df['chr'] == chrom]
        
        if len(prob_chr) == 0 or len(regions_chr) == 0:
            continue
            
        chr_intervals = pd.IntervalIndex.from_arrays(
            regions_chr['start'],
            regions_chr['end'] + 1,
            closed='both'
        )
        
        positions = prob_chr['start'].values
        interval_matches = chr_intervals.get_indexer(positions)
        
        valid_matches = interval_matches != -1
        if not valid_matches.any():
            continue
            
        matched_probs = prob_chr[valid_matches].copy().reset_index(drop=True)
        results.append(matched_probs)
    
    if not results:
        return pd.DataFrame(columns=prob_df.columns)
    
    return pd.concat(results, axis=0, ignore_index=True)

def calculate_chr_methylation(prob_df):
    """Calculates chromosome-level methylation rates.
    
    Args:
        prob_df (pd.DataFrame): DataFrame containing filtered probability data.
        
    Returns:
        pd.DataFrame: DataFrame with chromosome-level methylation rates.
    """
    # Group by chromosome and calculate weighted methylation rate
    # Assuming 'status' column exists or can be derived from prob_class_1
    if 'status' not in prob_df.columns:
        # If status is not in columns, derive it (this is a placeholder - adjust as needed)
        prob_df['status'] = (prob_df['prob_class_1'] >= 0.5).astype(int)
    
    chr_stats = prob_df.groupby('chr').apply(
        lambda x: np.sum(x['prob_class_1'] * x['status']) / np.sum(x['prob_class_1']) * 100 
        if np.sum(x['prob_class_1']) > 0 else np.nan
    ).reset_index()
    chr_stats.columns = ['chr', 'meth_rate']
    
    return chr_stats

def calculate_sample_average_methylation(prob_df):
    """Calculate average methylation rate for the entire sample.
    
    Args:
        prob_df (pd.DataFrame): DataFrame containing probability data.
        
    Returns:
        float: Average methylation rate across all CpGs in the sample.
    """
    if prob_df.empty:
        return 0.0
    
    if 'status' not in prob_df.columns:
        # If status is not in columns, derive it (adjust as needed)
        prob_df['status'] = (prob_df['prob_class_1'] >= 0.5).astype(int)
    
    total_weighted_status = np.sum(prob_df['status'] * prob_df['prob_class_1'])
    total_weight = np.sum(prob_df['prob_class_1'])
    
    if total_weight > 0:
        return (total_weighted_status / total_weight) * 100
    return 0.0

def create_background_pool(meta, background_label, regions_df):
    """Creates a pool of background reads by merging all background samples.
    
    Args:
        meta (pd.DataFrame): DataFrame containing sample metadata.
        background_label (str): Label identifying background samples.
        regions_df (pd.DataFrame): DataFrame containing regions to filter by.
        
    Returns:
        dict: Dictionary mapping read names to their data rows.
        
    Raises:
        ValueError: If no background samples are found or if pooling fails.
    """
    background_samples = meta[meta['label'] == background_label]
    
    if len(background_samples) == 0:
        raise ValueError(f"No samples found with background label: {background_label}")
    
    console.print(f"[bold blue]Creating background pool from {len(background_samples)} samples...[/bold blue]")
    
    # Dictionary to store read data
    background_pool = {}
    
    for _, sample in tqdm(background_samples.iterrows(), total=len(background_samples), desc="Reading background samples"):
        try:
            prob_df = read_prob_file(sample['prob_file_path'])
            
            # Apply regions filter if provided
            if regions_df is not None:
                prob_df = filter_sites_by_regions(prob_df, regions_df)
            
            if 'name' not in prob_df.columns:
                console.print(f"[yellow]Warning: 'name' column not found in {sample['sample']}. Skipping.[/yellow]")
                continue
                
            # Group by read name and add to pool
            read_groups = prob_df.groupby('name')
            
            for name, group in read_groups:
                if name not in background_pool:
                    background_pool[name] = group.reset_index(drop=True)
        
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to process background sample {sample['sample']}: {str(e)}[/yellow]")
    
    if not background_pool:
        raise ValueError("Failed to create background pool. No valid background reads found.")
    
    console.print(f"[green]Background pool created with {len(background_pool)} unique reads.[/green]")
    return background_pool

def dilute_sample(target_sample_info, background_pool, dilute_percentage, regions_df):
    """Dilutes a target sample with reads from the background pool.
    
    Args:
        target_sample_info (dict): Dictionary containing target sample information.
        background_pool (dict): Dictionary of background reads.
        dilute_percentage (float): Percentage of target sample reads to add from background.
        regions_df (pd.DataFrame): DataFrame containing regions to filter by.
        
    Returns:
        pd.DataFrame: Diluted probability DataFrame.
        
    Raises:
        ValueError: If dilution fails or if there are not enough reads.
    """
    sample_name = target_sample_info['sample']
    
    try:
        # Read target sample
        prob_df = read_prob_file(target_sample_info['prob_file_path'])
        
        # Apply regions filter if provided
        if regions_df is not None:
            prob_df = filter_sites_by_regions(prob_df, regions_df)
            
        if prob_df.empty:
            raise ValueError(f"No sites found in regions for sample: {sample_name}")
        
        # Get target read count
        if 'name' not in prob_df.columns:
            raise ValueError(f"'name' column not found in {sample_name}. Cannot dilute.")
            
        target_reads = prob_df['name'].unique()
        target_read_count = len(target_reads)
        
        # Calculate number of background reads to add
        bg_read_count = int(target_read_count * dilute_percentage)
        if bg_read_count <= 0:
            console.print(f"[yellow]Warning: Dilution percentage too low for sample {sample_name}. No reads added.[/yellow]")
            return prob_df
            
        # Select random reads from background pool
        bg_pool_keys = list(background_pool.keys())
        if len(bg_pool_keys) < bg_read_count:
            console.print(f"[yellow]Warning: Not enough reads in background pool. Using all {len(bg_pool_keys)} reads.[/yellow]")
            bg_read_count = len(bg_pool_keys)
            
        selected_reads = random.sample(bg_pool_keys, bg_read_count)
        
        # Create a DataFrame from selected background reads
        bg_dfs = [background_pool[read] for read in selected_reads]
        if not bg_dfs:
            raise ValueError(f"Failed to select background reads for sample: {sample_name}")
            
        bg_df = pd.concat(bg_dfs, ignore_index=True)
        
        # Combine target and background data
        diluted_df = pd.concat([prob_df, bg_df], ignore_index=True)
        
        console.print(f"[green]Sample {sample_name} diluted with {bg_read_count} background reads ({dilute_percentage*100:.1f}%).[/green]")
        
        return diluted_df
        
    except Exception as e:
        console.print(f"[red]Error diluting sample {sample_name}: {str(e)}[/red]")
        raise
        
def process_diluted_sample(sample_info, background_pool, dilute_percentage, regions_df, centralize):
    """Process a single diluted sample and return chromosome-level methylation data.
    
    Args:
        sample_info (dict): Dictionary containing sample information.
        background_pool (dict): Dictionary of background reads.
        dilute_percentage (float): Percentage of target sample reads to add from background.
        regions_df (pd.DataFrame): DataFrame containing regions to filter by.
        centralize (bool): If True, centralize methylation rates by subtracting the sample average.
        
    Returns:
        tuple: Sample name, dilute percentage, and chromosome-level methylation data.
        
    Raises:
        Exception: If there's an error processing the sample.
    """
    sample_name = sample_info['sample']
    
    try:
        # Dilute the sample with background reads
        diluted_df = dilute_sample(sample_info, background_pool, dilute_percentage, regions_df)
        
        # Calculate sample-level average methylation if centralization is requested
        sample_avg_meth = 0.0
        if centralize:
            sample_avg_meth = calculate_sample_average_methylation(diluted_df)
        
        # Calculate chromosome-level methylation
        chr_meth = calculate_chr_methylation(diluted_df)
        
        # Check if any chromosomes have valid methylation rates
        if chr_meth['meth_rate'].isna().all():
            raise ValueError(f"No valid methylation rates calculated for any chromosome in sample: {sample_name}")
        
        # Apply centralization if requested
        if centralize:
            chr_meth['meth_rate'] = chr_meth['meth_rate'] - sample_avg_meth
        
        return sample_name, dilute_percentage, chr_meth
        
    except Exception as e:
        console.print(f"[red]Error processing diluted sample {sample_name}: {str(e)}[/red]")
        raise

def process_samples_parallel(target_samples, background_pool, dilute_percentage, regions_df, centralize, ncpus):
    """Processes samples in parallel and merges results.
    
    Args:
        target_samples (pd.DataFrame): DataFrame containing target sample metadata.
        background_pool (dict): Dictionary of background reads.
        dilute_percentage (float): Percentage of target sample reads to add from background.
        regions_df (pd.DataFrame): DataFrame containing regions to filter by.
        centralize (bool): If True, centralize methylation rates by subtracting the sample average.
        ncpus (int): Number of CPU cores to use.
        
    Returns:
        pd.DataFrame: DataFrame with chromosome-level methylation data for all diluted samples.
    """
    # Initialize progress bar
    console.print("[bold green]Processing diluted samples...[/bold green]")
    
    with Pool(ncpus) as pool:
        process_func = partial(process_diluted_sample, 
                               background_pool=background_pool,
                               dilute_percentage=dilute_percentage,
                               regions_df=regions_df,
                               centralize=centralize)
        
        # Use tqdm for progress monitoring
        results = []
        failed_samples = []
        
        for result in tqdm(pool.imap(process_func, target_samples.to_dict('records')),
                         total=len(target_samples),
                         desc="Processing target samples"):
            try:
                results.append(result)
            except Exception as e:
                failed_samples.append(str(e))
        
        if failed_samples:
            console.print("\n[red]Failed samples:[/red]")
            for error in failed_samples:
                console.print(f"[red]{error}[/red]")
            if len(failed_samples) == len(target_samples):
                raise ValueError("All samples failed to process. Please check the error messages above.")
            console.print(f"\n[yellow]Warning: {len(failed_samples)} samples failed to process.[/yellow]")
    
    # Process results
    chr_data = {}
    dilute_percentages = {}
    for sample_name, dilute_pct, chr_meth in results:
        # Convert to dictionary for easy pivoting
        sample_dict = {row['chr']: row['meth_rate'] for _, row in chr_meth.iterrows()}
        chr_data[sample_name] = sample_dict
        dilute_percentages[sample_name] = dilute_pct
    
    # Create DataFrame from chromosome data
    chr_df = pd.DataFrame(chr_data).T
    
    # Ensure all chromosomes are present (fill with NaN if missing)
    all_chromosomes = sorted(list(set().union(*[set(data.keys()) for data in chr_data.values()])))
    for chr_name in all_chromosomes:
        if chr_name not in chr_df.columns:
            chr_df[chr_name] = np.nan
    
    # Sort chromosomes naturally (chr1, chr2, ..., chr22, chrX, chrY)
    def chr_sort_key(c):
        c = c.replace('chr', '')
        return (0, int(c)) if c.isdigit() else (1, c)
    
    chr_df = chr_df[sorted(chr_df.columns, key=chr_sort_key)]
    
    # Reset index to get sample names as a column
    chr_df = chr_df.reset_index().rename(columns={'index': 'sample'})
    
    # Add dilute_percentage column
    chr_df['dilute_percentage'] = chr_df['sample'].map(dilute_percentages)
    
    # Reorder columns to have sample, dilute_percentage first, then chromosomes
    cols = ['sample', 'dilute_percentage'] + [col for col in chr_df.columns if col not in ['sample', 'dilute_percentage']]
    chr_df = chr_df[cols]
    
    return chr_df

def process_diluted_samples(meta, target_label, background_label, dilute_percentage, regions_df, centralize, prefix, ncpus):
    """Process all target samples with dilution and generate the output file.
    
    Args:
        meta (pd.DataFrame): DataFrame containing sample metadata.
        target_label (str): Label identifying target samples.
        background_label (str): Label identifying background samples.
        dilute_percentage (float): Percentage of target sample reads to add from background.
        regions_df (pd.DataFrame): DataFrame containing regions to filter by (optional).
        centralize (bool): If True, centralize methylation rates by subtracting the sample average.
        prefix (str): Prefix for output files.
        ncpus (int): Number of CPU cores to use.
    """
    # Filter to get target samples
    target_samples = meta[meta['label'] == target_label]
    if len(target_samples) == 0:
        raise ValueError(f"No samples found with target label: {target_label}")
    
    console.print(f"[bold blue]Found {len(target_samples)} target samples.[/bold blue]")
    
    # Create background pool
    background_pool = create_background_pool(meta, background_label, regions_df)
    
    # Process samples in parallel
    chr_df = process_samples_parallel(target_samples, background_pool, dilute_percentage, regions_df, centralize, ncpus)
    
    # Merge with sample metadata (preserving label)
    result_df = pd.merge(meta[['sample', 'label']], chr_df, on='sample', how='right', suffixes=('', '_chr'))
    
    # Drop any duplicate columns that might have been created
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]
    
    # Reorder columns to have sample, label, dilute_percentage first, then chromosomes
    chr_cols = [col for col in result_df.columns if col.startswith('chr')]
    other_cols = [col for col in result_df.columns if not col.startswith('chr') and col not in ['sample', 'label', 'dilute_percentage']]
    result_df = result_df[['sample', 'label', 'dilute_percentage'] + chr_cols + other_cols]
    
    # Save results
    output_file = f'{prefix}_diluted_{dilute_percentage:.2f}.csv'
    result_df.to_csv(output_file, index=False)
    console.print(f"[bold green]Diluted chromosome-level methylation data saved to {output_file}[/bold green]")

@click.command()
@click.option('--meta-file', required=True, help='Path to metadata CSV file')
@click.option('--dmr', required=False, help='Path to DMR regions BED file (optional)')
@click.option('--target-label', required=True, help='Label identifying target samples')
@click.option('--background-label', default='Delivered', help='Label identifying background samples')
@click.option('--dilute-percentage', required=True, type=float, help='Percentage of target sample reads to add from background (0.2 means 20%)')
@click.option('--centralize', is_flag=True, default=False, help='Centralize methylation rates by subtracting the sample average')
@click.option('--output-prefix', required=True, help='Prefix for output files')
@click.option('--ncpus', default=1, type=int, help='Number of CPU cores to use')
def main(meta_file, dmr, target_label, background_label, dilute_percentage, centralize, output_prefix, ncpus):
    """Generate in-silico diluted methylation matrix from probability files.
    
    This script simulates dilution of target samples with background samples at
    a specified percentage ratio. It calculates methylation rates at the chromosome
    level for the diluted samples, with optional filtering by DMR regions and
    centralization of methylation rates.
    """
    try:
        # Print banner
        console.print("[bold blue]In-silico Diluted Methylation Matrix Simulator[/bold blue]")
        
        # Validate dilute percentage
        if dilute_percentage <= 0:
            console.print("[red]Error: Dilute percentage must be greater than 0[/red]")
            sys.exit(1)
        elif dilute_percentage > 4:
            console.print("[red]Error: Dilute percentage must be less than or equal to 4 (400%)[/red]")
            sys.exit(1)
        
        # Read metadata
        console.print(f"Reading metadata from {meta_file}...")
        meta = pd.read_csv(meta_file)
        
        if 'sample' not in meta.columns or 'label' not in meta.columns or 'prob_file_path' not in meta.columns:
            console.print("[red]Error: Metadata file must contain 'sample', 'label', and 'prob_file_path' columns[/red]")
            sys.exit(1)
        
        # Read DMR file if provided
        regions_df = None
        if dmr:
            console.print(f"Reading DMR regions from {dmr}...")
            regions_df = read_bed_file(dmr)
        
        # Print simulation parameters
        console.print(f"[green]Target label: {target_label}[/green]")
        console.print(f"[green]Background label: {background_label}[/green]")
        console.print(f"[green]Dilute percentage: {dilute_percentage:.2f} ({dilute_percentage*100:.1f}%)[/green]")
        
        if centralize:
            console.print("[green]Centralizing methylation rates by subtracting sample averages.[/green]")
        
        process_diluted_samples(meta, target_label, background_label, dilute_percentage, regions_df, centralize, output_prefix, ncpus)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)

if __name__ == '__main__':
    main()
