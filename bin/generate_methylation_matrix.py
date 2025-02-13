import pandas as pd
import numpy as np
import os
import click

def read_bedgraph(bedgraph_file_path):
    """Reads a bedgraph file into a DataFrame, skipping the first line start with track."""
    try:
        return pd.read_csv(bedgraph_file_path, sep="\t", header=None, comment='t', usecols=[0, 1, 2, 3],
                          names=['chr', 'start', 'end', 'meth_rate'])
    except ValueError as e:
        raise ValueError(f"Error reading bedGraph file {bedgraph_file_path}. "
                        f"Make sure it has at least 4 columns: chromosome, start, end, and methylation rate.") from e

def read_dmr_list(dmr_file):
    """Reads a DMR bed file into a DataFrame"""
    try:
        return pd.read_csv(dmr_file, sep='\t', header=None, usecols=[0, 1, 2], names=['chr', 'start', 'end'])
    except ValueError as e:
        raise ValueError(f"Error reading DMR file {dmr_file}. "
                        f"Make sure it has at least 3 columns: chromosome, start, and end.") from e

def extract_cpgs_in_dmr(bedgraph_df, dmr_df):
    """Filters CpG sites in bedgraph_df that fall within the regions defined in dmr_df."""
    results = None
    for _, dmr in dmr_df.iterrows():
        chr_filter = bedgraph_df['chr'] == dmr['chr']
        start_filter = bedgraph_df['start'] >= dmr['start']
        end_filter = bedgraph_df['end'] <= dmr['end']
        filtered_cpgs = bedgraph_df[chr_filter & start_filter & end_filter].copy()
        filtered_cpgs['chr_dmr'] = dmr['chr']
        filtered_cpgs['start_dmr'] = dmr['start']
        filtered_cpgs['end_dmr'] = dmr['end']
        if results is None:
            results = filtered_cpgs
        else:
            results = pd.concat([results, filtered_cpgs], axis=0)
    return results

def process_bedgraph_files(meta, dmr_list):
    """Processes each bedgraph file according to the meta DataFrame and outputs two DataFrames."""
    
    all_cpgs_df = None
    all_dmrs_df = None

    for _, row in meta.iterrows():
        sample_name = row['sample']
        bedgraph_file_path = row['bedgraph_file_path']
        
        # Read the bedgraph file and extract CpG sites within DMRs
        cpgs_in_dmr = extract_cpgs_in_dmr(read_bedgraph(bedgraph_file_path), dmr_list)

        meth_cpgs = cpgs_in_dmr[['chr', 'start', 'end', 'meth_rate']]
        meth_cpgs.columns = ['chr', 'start', 'end', sample_name]
        
        # Merge CpGs data
        if all_cpgs_df is None:
            all_cpgs_df = meth_cpgs.copy()
        else:
            all_cpgs_df = pd.merge(all_cpgs_df, meth_cpgs, how='outer', on=['chr', 'start', 'end'])

        # Merge DMRs data
        meth_dmrs = pd.DataFrame(cpgs_in_dmr.groupby(['chr_dmr', 'start_dmr', 'end_dmr'])['meth_rate'].mean().reset_index())
        meth_dmrs.columns = ['chr', 'start', 'end', sample_name]
        if all_dmrs_df is None:
            all_dmrs_df = meth_dmrs
        else:
            all_dmrs_df = pd.merge(all_dmrs_df, meth_dmrs, how='outer', on=['chr', 'start', 'end'])

    return all_cpgs_df, all_dmrs_df

def process_samples(meta, dmr_list, prefix):
    all_cpgs_df, all_dmrs_df = process_bedgraph_files(meta, dmr_list)

    all_cpgs_df['index'] = all_cpgs_df['chr'].astype(str) + ':' + all_cpgs_df['start'].astype(str) + '-' + all_cpgs_df['end'].astype(str)
    all_cpgs_df = all_cpgs_df.set_index('index')
    all_cpgs_df = all_cpgs_df.drop(columns=['chr', 'start', 'end'])

    all_dmrs_df['index'] = all_dmrs_df['chr'].astype(str) + ':' + all_dmrs_df['start'].astype(str) + '-' + all_dmrs_df['end'].astype(str)
    all_dmrs_df = all_dmrs_df.set_index('index')
    all_dmrs_df = all_dmrs_df.drop(columns=['chr', 'start', 'end'])

    sample_cpgs = pd.concat([meta[['sample', 'label']], all_cpgs_df[meta['sample']].T.reset_index()], axis=1).drop(columns=['index'])
    sample_dmrs = pd.concat([meta[['sample', 'label']], all_dmrs_df[meta['sample']].T.reset_index()], axis=1).drop(columns=['index'])

    sample_cpgs.to_csv(f'{prefix}_cpgs.tsv', sep='\t', index=False)
    sample_dmrs.to_csv(f'{prefix}_dmrs.tsv', sep='\t', index=False)

@click.command()
@click.option('--meta-file', required=True, help='Path to metadata CSV file')
@click.option('--dmr-file', required=True, help='Path to DMR bed file')
@click.option('--output-prefix', required=True, help='Prefix for output files')
def main(meta_file, dmr_file, output_prefix):
    """Generate methylation matrix from bedGraph files."""
    # Read metadata
    meta = pd.read_csv(meta_file)
    
    # Read DMR list
    dmr_list = read_dmr_list(dmr_file)
    
    # Process samples
    process_samples(meta, dmr_list, output_prefix)

if __name__ == '__main__':
    main()
    