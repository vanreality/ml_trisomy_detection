import pandas as pd
import numpy as np
import sys
import click
from multiprocessing import Pool
from functools import partial

def read_bedgraph(bedgraph_file_path):
    """Reads a bedgraph file into a DataFrame, skipping the first line start with track."""
    try:
        # First read just the header to check number of columns
        header = pd.read_csv(bedgraph_file_path, sep="\t", header=None, nrows=1, skiprows=1)
        num_cols = len(header.columns)
        
        if num_cols == 4:
            return pd.read_csv(bedgraph_file_path, sep="\t", header=None, skiprows=1, usecols=[0, 1, 2, 3],
                             names=['chr', 'start', 'end', 'meth_rate'])
        elif num_cols == 6:
            return pd.read_csv(bedgraph_file_path, sep="\t", header=None, skiprows=1, usecols=[0, 1, 2, 3, 4, 5],
                             names=['chr', 'start', 'end', 'meth_rate', 'meth_count', 'unmeth_count'])
        else:
            raise ValueError(f"Unexpected number of columns ({num_cols}) in bedGraph file")
    except ValueError as e:
        raise ValueError(f"Error reading bedGraph file {bedgraph_file_path}") from e

def read_dmr_list(dmr_file):
    """Reads a DMR bed file into a DataFrame"""
    try:
        return pd.read_csv(dmr_file, sep='\t', header=None, usecols=[0, 1, 2], names=['chr', 'start', 'end'])
    except ValueError as e:
        raise ValueError(f"Error reading DMR file {dmr_file}") from e

def read_prob_file(prob_file_path):
    """Reads a probability file into a DataFrame"""
    try:
        df = pd.read_csv(prob_file_path)
        df['prob_class_1'] = df['prob_class_1'].astype(float)
        df['status'] = df['status'].astype(int)
        return df
    except ValueError as e:
        raise ValueError(f"Error reading probability file {prob_file_path}") from e

def extract_cpgs_in_dmr(bedgraph_df, dmr_df):
    """Filters CpG sites in bedgraph_df that fall within the regions defined in dmr_df using vectorized operations."""
    # Ensure chromosome names are in the same format
    bedgraph_df['chr'] = bedgraph_df['chr'].astype(str)
    dmr_df['chr'] = dmr_df['chr'].astype(str)
    
    results = []

    # Process each chromosome that appears in both bedgraph and dmr files
    common_chroms = set(bedgraph_df['chr'].unique()) & set(dmr_df['chr'].unique())
    
    for chrom in common_chroms:
        # Filter bedgraph and dmr data for current chromosome
        bed_chr = bedgraph_df[bedgraph_df['chr'] == chrom]
        dmr_chr = dmr_df[dmr_df['chr'] == chrom]
        
        if len(bed_chr) == 0 or len(dmr_chr) == 0:
            continue
            
        # Create intervals for current chromosome
        chr_intervals = pd.IntervalIndex.from_arrays(
            dmr_chr['start'],
            dmr_chr['end'] + 1,
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
        matched_cpgs = bed_chr[valid_matches].copy().reset_index(drop=True)
        matched_dmrs = dmr_chr.iloc[interval_matches[valid_matches]].reset_index(drop=True)
        
        # Add DMR information
        matched_cpgs['chr_dmr'] = matched_dmrs['chr']
        matched_cpgs['start_dmr'] = matched_dmrs['start']
        matched_cpgs['end_dmr'] = matched_dmrs['end']
        
        results.append(matched_cpgs)
    
    if not results:
        columns = ['chr', 'start', 'end', 'meth_rate', 'chr_dmr', 'start_dmr', 'end_dmr']
        if 'meth_count' in bedgraph_df.columns:
            columns.extend(['meth_count', 'unmeth_count'])
        return pd.DataFrame(columns=columns)
    
    return pd.concat(results, axis=0, ignore_index=True)

def extract_probs_in_dmr(prob_df, dmr_df):
    """Filters probability sites that fall within DMR regions using vectorized operations."""
    prob_df['chr'] = prob_df['chr'].astype(str)
    dmr_df['chr'] = dmr_df['chr'].astype(str)
    
    results = []
    common_chroms = set(prob_df['chr'].unique()) & set(dmr_df['chr'].unique())
    
    for chrom in common_chroms:
        prob_chr = prob_df[prob_df['chr'] == chrom]
        dmr_chr = dmr_df[dmr_df['chr'] == chrom]
        
        if len(prob_chr) == 0 or len(dmr_chr) == 0:
            continue
            
        chr_intervals = pd.IntervalIndex.from_arrays(
            dmr_chr['start'],
            dmr_chr['end'] + 1,
            closed='both'
        )
        
        positions = prob_chr['start'].values
        interval_matches = chr_intervals.get_indexer(positions)
        
        valid_matches = interval_matches != -1
        if not valid_matches.any():
            continue
            
        matched_probs = prob_chr[valid_matches].copy().reset_index(drop=True)
        matched_dmrs = dmr_chr.iloc[interval_matches[valid_matches]].reset_index(drop=True)
        
        matched_probs['chr_dmr'] = matched_dmrs['chr']
        matched_probs['start_dmr'] = matched_dmrs['start'] 
        matched_probs['end_dmr'] = matched_dmrs['end']
        
        results.append(matched_probs)
    
    if not results:
        return pd.DataFrame(columns=['chr', 'start', 'end', 'name', 'status', 'prob_class_1', 'chr_dmr', 'start_dmr', 'end_dmr'])
    
    return pd.concat(results, axis=0, ignore_index=True)

def process_single_sample(sample_info, dmr_list):
    """Process a single sample and return CpG and DMR level data."""
    sample_name = sample_info['sample']
    bedgraph_file_path = sample_info['bedgraph_file_path']
    
    try:
        # Read and process bedgraph file for CpG level data
        bedgraph_df = read_bedgraph(bedgraph_file_path)
        cpgs_in_dmr = extract_cpgs_in_dmr(bedgraph_df, dmr_list)
        meth_cpgs = cpgs_in_dmr[['chr', 'start', 'end', 'meth_rate']]
        meth_cpgs.columns = ['chr', 'start', 'end', sample_name]
        
        # Process DMR level data based on whether prob_file_path exists
        if 'prob_file_path' in sample_info and pd.notna(sample_info['prob_file_path']):
            # New way using probability files
            prob_df = read_prob_file(sample_info['prob_file_path'])
            probs_in_dmr = extract_probs_in_dmr(prob_df, dmr_list)
            
            # Calculate weighted methylation rate for each DMR
            dmr_stats = probs_in_dmr.groupby(['chr_dmr', 'start_dmr', 'end_dmr']).agg({
                'prob_class_1': lambda x: np.sum(x * probs_in_dmr.loc[x.index, 'status']) / np.sum(x) * 100
            }).reset_index()
            
            meth_dmrs = dmr_stats[['chr_dmr', 'start_dmr', 'end_dmr', 'prob_class_1']]
            meth_dmrs.columns = ['chr', 'start', 'end', sample_name]
            
        else:
            # Process DMR level data based on bedgraph format
            if 'meth_count' in cpgs_in_dmr.columns:
                # New way using meth_count and unmeth_count
                dmr_stats = cpgs_in_dmr.groupby(['chr_dmr', 'start_dmr', 'end_dmr']).agg({
                    'meth_count': 'sum',
                    'unmeth_count': 'sum'
                }).reset_index()
                dmr_stats[sample_name] = dmr_stats['meth_count'] / (dmr_stats['meth_count'] + dmr_stats['unmeth_count']) * 100
                meth_dmrs = dmr_stats[['chr_dmr', 'start_dmr', 'end_dmr', sample_name]]
            else:
                # Old way using average meth_rate
                meth_dmrs = cpgs_in_dmr.groupby(['chr_dmr', 'start_dmr', 'end_dmr'])['meth_rate'].agg(['mean', 'count']).reset_index()
                meth_dmrs = meth_dmrs[meth_dmrs['count'] > 0]
                meth_dmrs = meth_dmrs[['chr_dmr', 'start_dmr', 'end_dmr', 'mean']]
            
            meth_dmrs.columns = ['chr', 'start', 'end', sample_name]
        
        return meth_cpgs, meth_dmrs
        
    except Exception as e:
        print(f"Error processing sample {sample_name}: {str(e)}")
        sys.exit(1)

def process_samples_parallel(meta, dmr_list, ncpus):
    """Processes samples in parallel and merges results."""
    
    with Pool(ncpus) as pool:
        process_func = partial(process_single_sample, dmr_list=dmr_list)
        results = pool.map(process_func, meta.to_dict('records'))
    
    cpg_dfs, dmr_dfs = zip(*results)
    
    # Merge CpG results
    all_cpgs_df = cpg_dfs[0]
    for df in cpg_dfs[1:]:
        all_cpgs_df = pd.merge(all_cpgs_df, df, how='outer', on=['chr', 'start', 'end'])
    
    # Merge DMR results
    all_dmrs_df = dmr_dfs[0]
    for df in dmr_dfs[1:]:
        all_dmrs_df = pd.merge(all_dmrs_df, df, how='outer', on=['chr', 'start', 'end'])
    
    all_cpgs_df = all_cpgs_df.fillna(np.nan)
    all_dmrs_df = all_dmrs_df.fillna(np.nan)
    
    return all_cpgs_df, all_dmrs_df

def process_samples(meta, dmr_list, prefix, ncpus):
    all_cpgs_df, all_dmrs_df = process_samples_parallel(meta, dmr_list, ncpus)
    
    # Create index for both dataframes
    all_cpgs_df['index'] = all_cpgs_df['chr'].astype(str) + ':' + all_cpgs_df['start'].astype(int).astype(str) + '-' + all_cpgs_df['end'].astype(int).astype(str)
    all_cpgs_df = all_cpgs_df.set_index('index')
    all_cpgs_df = all_cpgs_df.drop(columns=['chr', 'start', 'end'])

    all_dmrs_df['index'] = all_dmrs_df['chr'].astype(str) + ':' + all_dmrs_df['start'].astype(int).astype(str) + '-' + all_dmrs_df['end'].astype(int).astype(str)
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