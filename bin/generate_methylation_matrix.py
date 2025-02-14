import pandas as pd
import numpy as np
import os
import click
from multiprocessing import Pool
from functools import partial

def read_bedgraph(bedgraph_file_path):
    """Reads a bedgraph file into a DataFrame, skipping the first line start with track."""
    try:
        return pd.read_csv(bedgraph_file_path, sep="\t", header=None, comment='t', usecols=[0, 1, 2, 3],
                          names=['chr', 'start', 'end', 'meth_rate'])
    except ValueError as e:
        raise ValueError(f"Error reading bedGraph file {bedgraph_file_path}") from e

def read_dmr_list(dmr_file):
    """Reads a DMR bed file into a DataFrame"""
    try:
        return pd.read_csv(dmr_file, sep='\t', header=None, usecols=[0, 1, 2], names=['chr', 'start', 'end'])
    except ValueError as e:
        raise ValueError(f"Error reading DMR file {dmr_file}") from e

def extract_cpgs_in_dmr(bedgraph_df, dmr_df):
    """Filters CpG sites in bedgraph_df that fall within the regions defined in dmr_df using vectorized operations."""
    # Create IntervalIndex for the DMR regions
    dmr_intervals = pd.IntervalIndex.from_arrays(
        dmr_df['start'],
        dmr_df['end'],
        closed='both'
    )
    
    results = []
    # Process each chromosome separately to reduce memory usage
    for chrom in bedgraph_df['chr'].unique():
        # Filter bedgraph and dmr data for current chromosome
        bed_chr = bedgraph_df[bedgraph_df['chr'] == chrom]
        dmr_chr = dmr_df[dmr_df['chr'] == chrom]
        
        if len(bed_chr) == 0 or len(dmr_chr) == 0:
            continue
            
        # Create intervals for current chromosome
        chr_intervals = pd.IntervalIndex.from_arrays(
            dmr_chr['start'],
            dmr_chr['end'],
            closed='both'
        )
        
        # Find which interval each CpG belongs to
        cpg_positions = bed_chr['start'].values
        interval_matches = chr_intervals.get_indexer(cpg_positions)
        
        # Filter valid matches
        valid_matches = interval_matches != -1
        if not valid_matches.any():
            continue
            
        # Create result DataFrame for matched CpGs
        matched_cpgs = bed_chr[valid_matches].copy()
        matched_dmrs = dmr_chr.iloc[interval_matches[valid_matches]].reset_index(drop=True)
        
        # Add DMR information
        matched_cpgs['chr_dmr'] = matched_dmrs['chr']
        matched_cpgs['start_dmr'] = matched_dmrs['start']
        matched_cpgs['end_dmr'] = matched_dmrs['end']
        
        results.append(matched_cpgs)
    
    if not results:
        return pd.DataFrame(columns=['chr', 'start', 'end', 'meth_rate', 'chr_dmr', 'start_dmr', 'end_dmr'])
    
    return pd.concat(results, axis=0, ignore_index=True)

def process_single_bedgraph(bedgraph_info, dmr_list):
    """Process a single bedgraph file and return CpG and DMR level data for one sample."""
    sample_name = bedgraph_info['sample']
    bedgraph_file_path = bedgraph_info['bedgraph_file_path']
    
    # Read the bedgraph file and extract CpG sites within DMRs
    cpgs_in_dmr = extract_cpgs_in_dmr(read_bedgraph(bedgraph_file_path), dmr_list)
    
    # Process CpG level data
    meth_cpgs = cpgs_in_dmr[['chr', 'start', 'end', 'meth_rate']]
    meth_cpgs.columns = ['chr', 'start', 'end', sample_name]
    
    # Process DMR level data
    meth_dmrs = pd.DataFrame(cpgs_in_dmr.groupby(['chr_dmr', 'start_dmr', 'end_dmr'])['meth_rate'].mean().reset_index())
    meth_dmrs.columns = ['chr', 'start', 'end', sample_name]
    
    return meth_cpgs, meth_dmrs

def process_bedgraph_files(meta, dmr_list, ncpus):
    """Processes bedgraph files in parallel and merges results."""
    
    # Create a pool of workers
    with Pool(ncpus) as pool:
        # Process each bedgraph file in parallel
        process_func = partial(process_single_bedgraph, dmr_list=dmr_list)
        results = pool.map(process_func, meta.to_dict('records'))
    
    # Unzip results
    cpg_dfs, dmr_dfs = zip(*results)
    
    # Merge CpG results
    all_cpgs_df = cpg_dfs[0]
    for df in cpg_dfs[1:]:
        all_cpgs_df = pd.merge(all_cpgs_df, df, how='outer', on=['chr', 'start', 'end'])
    
    # Merge DMR results
    all_dmrs_df = dmr_dfs[0]
    for df in dmr_dfs[1:]:
        all_dmrs_df = pd.merge(all_dmrs_df, df, how='outer', on=['chr', 'start', 'end'])
    
    # Fill NA values with np.nan
    all_cpgs_df = all_cpgs_df.fillna(np.nan)
    all_dmrs_df = all_dmrs_df.fillna(np.nan)
    
    return all_cpgs_df, all_dmrs_df

def process_samples(meta, dmr_list, prefix, ncpus):
    all_cpgs_df, all_dmrs_df = process_bedgraph_files(meta, dmr_list, ncpus)
    
    # Create index for both dataframes
    all_cpgs_df['index'] = all_cpgs_df['chr'].astype(str) + ':' + all_cpgs_df['start'].astype(str) + '-' + all_cpgs_df['end'].astype(str)
    all_cpgs_df = all_cpgs_df.set_index('index')
    all_cpgs_df = all_cpgs_df.drop(columns=['chr', 'start', 'end'])

    all_dmrs_df['index'] = all_dmrs_df['chr'].astype(str) + ':' + all_dmrs_df['start'].astype(str) + '-' + all_dmrs_df['end'].astype(str)
    all_dmrs_df = all_dmrs_df.set_index('index')
    all_dmrs_df = all_dmrs_df.drop(columns=['chr', 'start', 'end'])

    # Create final sample matrices
    sample_cpgs = pd.concat([meta[['sample', 'label']], all_cpgs_df[meta['sample']].T.reset_index()], axis=1).drop(columns=['index'])
    sample_dmrs = pd.concat([meta[['sample', 'label']], all_dmrs_df[meta['sample']].T.reset_index()], axis=1).drop(columns=['index'])

    # Save results
    sample_cpgs.to_csv(f'{prefix}_cpgs.tsv', sep='\t', index=False)
    sample_dmrs.to_csv(f'{prefix}_dmrs.tsv', sep='\t', index=False)

@click.command()
@click.option('--meta-file', required=True, help='Path to metadata CSV file')
@click.option('--dmr-file', required=True, help='Path to DMR bed file')
@click.option('--output-prefix', required=True, help='Prefix for output files')
@click.option('--ncpus', default=1, type=int, help='Number of CPU cores to use')
def main(meta_file, dmr_file, output_prefix, ncpus):
    """Generate methylation matrix from bedGraph files."""
    # Read metadata
    meta = pd.read_csv(meta_file)
    
    # Read DMR list
    dmr_list = read_dmr_list(dmr_file)
    
    # Process samples
    process_samples(meta, dmr_list, output_prefix, ncpus)

if __name__ == '__main__':
    main()
    