import pandas as pd
import numpy as np
import sys
import click
from multiprocessing import Pool
from functools import partial
from rich.console import Console
from rich.progress import Progress
from tqdm import tqdm

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
        df['status'] = df['status'].astype(int)
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

def filter_by_insert_size(prob_df, insert_size_cutoff):
    """Filters probability data based on insert size.
    
    Args:
        prob_df (pd.DataFrame): DataFrame containing probability data.
        insert_size_cutoff (int, optional): Maximum insert size to include. If None, no filtering is applied.
        
    Returns:
        pd.DataFrame: Filtered probability DataFrame.
    """
    if insert_size_cutoff is None:
        return prob_df
        
    if 'insert_size' not in prob_df.columns:
        console.print("[yellow]Warning: 'insert_size' column not found in probability file. No filtering applied.[/yellow]")
        return prob_df
    
    return prob_df[prob_df['insert_size'] < insert_size_cutoff].reset_index(drop=True)

def calculate_chr_methylation(prob_df):
    """Calculates chromosome-level methylation rates.
    
    Args:
        prob_df (pd.DataFrame): DataFrame containing filtered probability data.
        
    Returns:
        pd.DataFrame: DataFrame with chromosome-level methylation rates.
    """
    # Group by chromosome and calculate weighted methylation rate
    chr_stats = prob_df.groupby('chr').apply(
        lambda x: np.sum(x['prob_class_1'] * x['status']) / np.sum(x['prob_class_1']) * 100 
        if np.sum(x['prob_class_1']) > 0 else np.nan
    ).reset_index()
    chr_stats.columns = ['chr', 'meth_rate']
    
    return chr_stats

def process_single_sample(sample_info, regions_df, insert_size_cutoff):
    """Process a single sample and return chromosome-level methylation data.
    
    Args:
        sample_info (dict): Dictionary containing sample information.
        regions_df (pd.DataFrame): DataFrame containing regions to filter by (optional).
        insert_size_cutoff (int, optional): Maximum insert size to include. If None, no filtering is applied.
        
    Returns:
        tuple: Sample name and chromosome-level methylation data.
        
    Raises:
        Exception: If there's an error processing the sample.
    """
    sample_name = sample_info['sample']
    prob_file_path = sample_info['prob_file_path']
    
    try:
        # Read probability file
        prob_df = read_prob_file(prob_file_path)
        
        # Apply regions filter if regions_df is provided
        if regions_df is not None:
            prob_df = filter_sites_by_regions(prob_df, regions_df)
        
        # Filter by insert size if cutoff is provided
        prob_df = filter_by_insert_size(prob_df, insert_size_cutoff)
        
        # Calculate chromosome-level methylation
        chr_meth = calculate_chr_methylation(prob_df)
        
        return sample_name, chr_meth
        
    except Exception as e:
        console.print(f"[red]Error processing sample {sample_name}: {str(e)}[/red]")
        raise

def process_samples_parallel(meta, regions_df, insert_size_cutoff, ncpus):
    """Processes samples in parallel and merges results.
    
    Args:
        meta (pd.DataFrame): DataFrame containing sample metadata.
        regions_df (pd.DataFrame): DataFrame containing regions to filter by (optional).
        insert_size_cutoff (int, optional): Maximum insert size to include. If None, no filtering is applied.
        ncpus (int): Number of CPU cores to use.
        
    Returns:
        pd.DataFrame: DataFrame with chromosome-level methylation data for all samples.
    """
    # Initialize progress bar
    console.print("[bold green]Processing samples...[/bold green]")
    
    with Pool(ncpus) as pool:
        process_func = partial(process_single_sample, 
                               regions_df=regions_df, 
                               insert_size_cutoff=insert_size_cutoff)
        
        # Use tqdm for progress monitoring
        results = list(tqdm(
            pool.imap(process_func, meta.to_dict('records')),
            total=len(meta),
            desc="Processing samples"
        ))
    
    # Process results
    chr_data = {}
    for sample_name, chr_meth in results:
        # Convert to dictionary for easy pivoting
        sample_dict = {row['chr']: row['meth_rate'] for _, row in chr_meth.iterrows()}
        chr_data[sample_name] = sample_dict
    
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
    
    return chr_df

def process_samples(meta, regions_df, insert_size_cutoff, prefix, ncpus):
    """Process all samples and generate the output file.
    
    Args:
        meta (pd.DataFrame): DataFrame containing sample metadata.
        regions_df (pd.DataFrame): DataFrame containing regions to filter by (optional).
        insert_size_cutoff (int, optional): Maximum insert size to include. If None, no filtering is applied.
        prefix (str): Prefix for output files.
        ncpus (int): Number of CPU cores to use.
    """
    # Process samples in parallel
    chr_df = process_samples_parallel(meta, regions_df, insert_size_cutoff, ncpus)
    
    # Merge with sample metadata (preserving label)
    result_df = pd.merge(meta[['sample', 'label']], chr_df, on='sample', how='right')
    
    # Save results
    output_file = f'{prefix}_chr_level.tsv'
    result_df.to_csv(output_file, sep='\t', index=False)
    console.print(f"[bold green]Chromosome-level methylation data saved to {output_file}[/bold green]")

@click.command()
@click.option('--meta-file', required=True, help='Path to metadata CSV file')
@click.option('--bed-file', required=False, help='Path to BED file with selected regions (optional)')
@click.option('--insert-size-cutoff', type=int, help='Maximum insert size to include (optional)')
@click.option('--output-prefix', required=True, help='Prefix for output files')
@click.option('--ncpus', default=1, type=int, help='Number of CPU cores to use')
def main(meta_file, bed_file, insert_size_cutoff, output_prefix, ncpus):
    """Generate chromosome-level methylation matrix from probability files.
    
    This script calculates methylation rates at the chromosome level based on
    probability files, with optional filtering by regions and insert size.
    """
    try:
        # Print banner
        console.print("[bold blue]Chromosome-level Methylation Calculator[/bold blue]")
        
        # Read metadata
        console.print(f"Reading metadata from {meta_file}...")
        meta = pd.read_csv(meta_file)
        
        if 'sample' not in meta.columns or 'label' not in meta.columns or 'prob_file_path' not in meta.columns:
            console.print("[red]Error: Metadata file must contain 'sample', 'label', and 'prob_file_path' columns[/red]")
            sys.exit(1)
        
        # Read BED file if provided
        regions_df = None
        if bed_file:
            console.print(f"Reading regions from {bed_file}...")
            regions_df = read_bed_file(bed_file)
        
        # Process samples
        if insert_size_cutoff is not None:
            console.print(f"Using insert size cutoff: {insert_size_cutoff}")
        else:
            console.print("[yellow]No insert size cutoff provided. Using all sites.[/yellow]")
            
        process_samples(meta, regions_df, insert_size_cutoff, output_prefix, ncpus)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)

if __name__ == '__main__':
    main()
