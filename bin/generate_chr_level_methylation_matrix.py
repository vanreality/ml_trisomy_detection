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

def filter_by_cpg_sites(prob_df, cpg_sites_df):
    """Filters probability data to include only specific CpG sites.
    
    Args:
        prob_df (pd.DataFrame): DataFrame containing probability data.
        cpg_sites_df (pd.DataFrame): DataFrame containing specific CpG sites to include.
        
    Returns:
        pd.DataFrame: Filtered probability DataFrame.
    """
    prob_df['chr'] = prob_df['chr'].astype(str)
    cpg_sites_df['chr'] = cpg_sites_df['chr'].astype(str)
    
    results = []
    common_chroms = set(prob_df['chr'].unique()) & set(cpg_sites_df['chr'].unique())
    
    for chrom in common_chroms:
        prob_chr = prob_df[prob_df['chr'] == chrom]
        cpg_chr = cpg_sites_df[cpg_sites_df['chr'] == chrom]
        
        if len(prob_chr) == 0 or len(cpg_chr) == 0:
            continue
        
        # Create a set of positions for faster lookup
        cpg_positions = set(zip(cpg_chr['start'].values, cpg_chr['end'].values))
        
        # Filter prob_df to only include rows with positions in cpg_positions
        matched_positions = [
            (start, end) in cpg_positions 
            for start, end in zip(prob_chr['start'].values, prob_chr['end'].values)
        ]
        
        if not any(matched_positions):
            continue
            
        matched_probs = prob_chr[matched_positions].copy().reset_index(drop=True)
        results.append(matched_probs)
    
    if not results:
        return pd.DataFrame(columns=prob_df.columns)
    
    return pd.concat(results, axis=0, ignore_index=True)

def filter_by_depth(prob_df, min_depth):
    """Filters probability data based on minimum depth at each CpG position.
    
    Args:
        prob_df (pd.DataFrame): DataFrame containing probability data.
        min_depth (int): Minimum number of reads required at each CpG position.
        
    Returns:
        pd.DataFrame: Filtered probability DataFrame.
    """
    # Group by chromosome and position to count depth
    depth_df = prob_df.groupby(['chr', 'start', 'end']).size().reset_index(name='depth')
    
    # Filter positions with sufficient depth
    valid_positions = depth_df[depth_df['depth'] >= min_depth]
    
    # Merge back with original data to get all rows for valid positions
    # Use suffixes to handle duplicate column names
    filtered_df = pd.merge(prob_df, valid_positions[['chr', 'start', 'end']], 
                          on=['chr', 'start', 'end'], how='inner', suffixes=('', '_valid'))
    
    # Drop any duplicate columns that might have been created
    filtered_df = filtered_df.loc[:, ~filtered_df.columns.duplicated()]
    
    return filtered_df.reset_index(drop=True)

def apply_prob_cutoff(prob_df, prob_cutoff):
    """Apply probability cutoff to the prob_class_1 values.
    
    Args:
        prob_df (pd.DataFrame): DataFrame containing probability data.
        prob_cutoff (float): Cutoff value for probabilities. Values below this will be set to 0, those above or equal to it will be set to 1.
        
    Returns:
        pd.DataFrame: DataFrame with modified probability values.
    """
    if prob_cutoff is None:
        return prob_df
    
    prob_df = prob_df.copy()
    # Set values below cutoff to 0, and above/equal to cutoff to 1
    prob_df['prob_class_1'] = (prob_df['prob_class_1'] >= prob_cutoff).astype(float)
    
    return prob_df

def calculate_sample_average_methylation(prob_df):
    """Calculate average methylation rate for the entire sample.
    
    Args:
        prob_df (pd.DataFrame): DataFrame containing probability data.
        
    Returns:
        float: Average methylation rate across all CpGs in the sample.
    """
    if prob_df.empty:
        return 0.0
    
    total_weighted_status = np.sum(prob_df['status'] * prob_df['prob_class_1'])
    total_weight = np.sum(prob_df['prob_class_1'])
    
    if total_weight > 0:
        return (total_weighted_status / total_weight) * 100
    return 0.0

def process_single_sample(sample_info, regions_df, cpg_sites_df, insert_size_cutoff, min_depth, prob_cutoff, centralize):
    """Process a single sample and return chromosome-level methylation data.
    
    Args:
        sample_info (dict): Dictionary containing sample information.
        regions_df (pd.DataFrame): DataFrame containing regions to filter by (optional).
        cpg_sites_df (pd.DataFrame): DataFrame containing specific CpG sites to include (optional).
        insert_size_cutoff (int, optional): Maximum insert size to include. If None, no filtering is applied.
        min_depth (int, optional): Minimum number of reads required at each CpG position.
        prob_cutoff (float, optional): Cutoff value for probabilities. Values below this will be set to 0, those above will be set to 1.
        centralize (bool): If True, centralize methylation rates by subtracting the sample average.
        
    Returns:
        tuple: Sample name and chromosome-level methylation data.
        
    Raises:
        Exception: If there's an error processing the sample or if filtering results in empty DataFrame.
    """
    sample_name = sample_info['sample']
    prob_file_path = sample_info['prob_file_path']
    
    try:
        # Read probability file
        prob_df = read_prob_file(prob_file_path)
        if prob_df.empty:
            raise ValueError(f"Empty probability file: {prob_file_path}")
        
        # Apply probability cutoff if provided
        prob_df = apply_prob_cutoff(prob_df, prob_cutoff)
        
        # Apply regions filter if regions_df is provided
        if regions_df is not None:
            prob_df = filter_sites_by_regions(prob_df, regions_df)
            if prob_df.empty:
                raise ValueError(f"No sites found in DMR regions for sample: {sample_name}")
        
        # Apply CpG sites filter if cpg_sites_df is provided
        if cpg_sites_df is not None:
            prob_df = filter_by_cpg_sites(prob_df, cpg_sites_df)
            if prob_df.empty:
                raise ValueError(f"No matching CpG sites found for sample: {sample_name}")
        
        # Filter by insert size if cutoff is provided
        if insert_size_cutoff is not None:
            prob_df = filter_by_insert_size(prob_df, insert_size_cutoff)
            if prob_df.empty:
                raise ValueError(f"No sites pass insert size filter ({insert_size_cutoff}) for sample: {sample_name}")
        
        # Filter by depth if min_depth is provided
        if min_depth is not None:
            prob_df = filter_by_depth(prob_df, min_depth)
            if prob_df.empty:
                raise ValueError(f"No sites pass depth filter ({min_depth}) for sample: {sample_name}")
        
        # Calculate sample-level average methylation if centralization is requested
        sample_avg_meth = 0.0
        if centralize:
            sample_avg_meth = calculate_sample_average_methylation(prob_df)
        
        # Calculate chromosome-level methylation
        chr_meth = calculate_chr_methylation(prob_df)
        
        # Check if any chromosomes have valid methylation rates
        if chr_meth['meth_rate'].isna().all():
            raise ValueError(f"No valid methylation rates calculated for any chromosome in sample: {sample_name}")
        
        # Apply centralization if requested
        if centralize:
            chr_meth['meth_rate'] = chr_meth['meth_rate'] - sample_avg_meth
        
        return sample_name, chr_meth
        
    except Exception as e:
        console.print(f"[red]Error processing sample {sample_name}: {str(e)}[/red]")
        raise

def process_samples_parallel(meta, regions_df, cpg_sites_df, insert_size_cutoff, min_depth, prob_cutoff, centralize, ncpus):
    """Processes samples in parallel and merges results.
    
    Args:
        meta (pd.DataFrame): DataFrame containing sample metadata.
        regions_df (pd.DataFrame): DataFrame containing regions to filter by (optional).
        cpg_sites_df (pd.DataFrame): DataFrame containing specific CpG sites to include (optional).
        insert_size_cutoff (int, optional): Maximum insert size to include. If None, no filtering is applied.
        min_depth (int, optional): Minimum number of reads required at each CpG position.
        prob_cutoff (float, optional): Cutoff value for probabilities. Values below this will be set to 0, those above will be set to 1.
        centralize (bool): If True, centralize methylation rates by subtracting the sample average.
        ncpus (int): Number of CPU cores to use.
        
    Returns:
        pd.DataFrame: DataFrame with chromosome-level methylation data for all samples.
    """
    # Initialize progress bar
    console.print("[bold green]Processing samples...[/bold green]")
    
    with Pool(ncpus) as pool:
        process_func = partial(process_single_sample, 
                               regions_df=regions_df,
                               cpg_sites_df=cpg_sites_df,
                               insert_size_cutoff=insert_size_cutoff,
                               min_depth=min_depth,
                               prob_cutoff=prob_cutoff,
                               centralize=centralize)
        
        # Use tqdm for progress monitoring
        results = []
        failed_samples = []
        
        for result in tqdm(pool.imap(process_func, meta.to_dict('records')),
                         total=len(meta),
                         desc="Processing samples"):
            try:
                results.append(result)
            except Exception as e:
                failed_samples.append(str(e))
        
        if failed_samples:
            console.print("\n[red]Failed samples:[/red]")
            for error in failed_samples:
                console.print(f"[red]{error}[/red]")
            if len(failed_samples) == len(meta):
                raise ValueError("All samples failed to process. Please check the error messages above.")
            console.print(f"\n[yellow]Warning: {len(failed_samples)} samples failed to process.[/yellow]")
    
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

def process_samples(meta, regions_df, cpg_sites_df, insert_size_cutoff, min_depth, prob_cutoff, centralize, prefix, ncpus):
    """Process all samples and generate the output file.
    
    Args:
        meta (pd.DataFrame): DataFrame containing sample metadata.
        regions_df (pd.DataFrame): DataFrame containing regions to filter by (optional).
        cpg_sites_df (pd.DataFrame): DataFrame containing specific CpG sites to include (optional).
        insert_size_cutoff (int, optional): Maximum insert size to include. If None, no filtering is applied.
        min_depth (int, optional): Minimum number of reads required at each CpG position.
        prob_cutoff (float, optional): Cutoff value for probabilities. Values below this will be set to 0, those above will be set to 1.
        centralize (bool): If True, centralize methylation rates by subtracting the sample average.
        prefix (str): Prefix for output files.
        ncpus (int): Number of CPU cores to use.
    """
    # Process samples in parallel
    chr_df = process_samples_parallel(meta, regions_df, cpg_sites_df, insert_size_cutoff, min_depth, prob_cutoff, centralize, ncpus)
    
    # Merge with sample metadata (preserving label)
    # Use suffixes to handle duplicate column names
    result_df = pd.merge(meta[['sample', 'label']], chr_df, on='sample', how='right', suffixes=('', '_chr'))
    
    # Drop any duplicate columns that might have been created
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]
    
    # Save results
    output_file = f'{prefix}_chr_level.tsv'
    result_df.to_csv(output_file, sep='\t', index=False)
    console.print(f"[bold green]Chromosome-level methylation data saved to {output_file}[/bold green]")

@click.command()
@click.option('--meta-file', required=True, help='Path to metadata CSV file')
@click.option('--hypo-dmr-bed', required=False, help='Path to hypo DMR regions BED file (optional)')
@click.option('--hyper-dmr-bed', required=False, help='Path to hyper DMR regions BED file (optional)')
@click.option('--cpg', required=False, help='Path to CpG sites BED file (optional)')
@click.option('--insert-size-cutoff', type=int, help='Maximum insert size to include (optional)')
@click.option('--min-depth', type=int, help='Minimum number of reads required at each CpG position (optional)')
@click.option('--prob-cutoff', type=float, help='Probability cutoff. Values below this will be set to 0, those above or equal to it will be set to 1.')
@click.option('--centralize', is_flag=True, default=False, help='Centralize methylation rates by subtracting the sample average')
@click.option('--output-prefix', required=True, help='Prefix for output files')
@click.option('--ncpus', default=1, type=int, help='Number of CPU cores to use')
def main(meta_file, hypo_dmr_bed, hyper_dmr_bed, cpg, insert_size_cutoff, min_depth, prob_cutoff, centralize, output_prefix, ncpus):
    """Generate chromosome-level methylation matrix from probability files.
    
    This script calculates methylation rates at the chromosome level based on
    probability files, with optional filtering by DMR regions, specific CpG sites,
    insert size, and minimum read depth. It can also apply probability cutoffs
    and centralize methylation rates.
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
        
        # Read DMR file if provided
        hypo_regions_df = None
        hyper_regions_df = None
        if hypo_dmr_bed:
            console.print(f"Reading hypo DMR regions from {hypo_dmr_bed}...")
            hypo_regions_df = read_bed_file(hypo_dmr_bed)
        if hyper_dmr_bed:
            console.print(f"Reading hyper DMR regions from {hyper_dmr_bed}...")
            hyper_regions_df = read_bed_file(hyper_dmr_bed)
        
        # Read CpG sites file if provided
        cpg_sites_df = None
        if cpg:
            console.print(f"Reading CpG sites from {cpg}...")
            cpg_sites_df = read_bed_file(cpg)
        
        # Print filtering information
        if insert_size_cutoff is not None:
            console.print(f"Using insert size cutoff: {insert_size_cutoff}")
        else:
            console.print("[yellow]No insert size cutoff provided. Using all sites.[/yellow]")
            
        if min_depth is not None:
            console.print(f"Using minimum depth filter: {min_depth}")
        else:
            console.print("[yellow]No minimum depth filter provided. Using all sites.[/yellow]")
        
        if prob_cutoff is not None:
            console.print(f"Using probability cutoff: {prob_cutoff}")
        else:
            console.print("[yellow]No probability cutoff provided. Using original probability values.[/yellow]")
        
        if centralize:
            console.print("[green]Centralizing methylation rates by subtracting sample averages.[/green]")
        
        process_samples(meta, hypo_regions_df, cpg_sites_df, insert_size_cutoff, min_depth, prob_cutoff, centralize, f'{output_prefix}_hypo', ncpus)
        process_samples(meta, hyper_regions_df, cpg_sites_df, insert_size_cutoff, min_depth, prob_cutoff, centralize, f'{output_prefix}_hyper', ncpus)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)

if __name__ == '__main__':
    main()
