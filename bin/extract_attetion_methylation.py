import pandas as pd
import numpy as np
import pysam
import click
from Bio import SeqIO
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial

def get_cpg_sites(bed, fasta):
    """Extracts CpG sites from the DMR regions in the reference genome."""
    cpg_sites = []
    fasta_sequences = SeqIO.to_dict(SeqIO.parse(fasta, "fasta"))
    
    bed_df = pd.read_csv(bed, sep='\t', header=None, usecols=[0, 1, 2], names=['chr', 'start', 'end'])
    for _, row in bed_df.iterrows():
        chr_seq = str(fasta_sequences[row['chr']].seq)
        for i in range(row['start'], row['end']):
            if chr_seq[i:i+2] == "CG" or chr_seq[i:i+2] == "cg":
                cpg_sites.append([row['chr'], i, i + 2, row['chr'], row['start'], row['end']])
    
    return pd.DataFrame(cpg_sites, columns=['chr', 'start', 'end', 'chr_dmr', 'start_dmr', 'end_dmr'])

def process_chunk(chunk_info, txt_df_indexed, cpg_site_df):
    """Process a chunk of the BAM file."""
    bam_file, region = chunk_info
    cpg_prob_list = []
    
    # Open BAM file for this chunk
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        # Fetch reads only from the specified region
        for read in bam.fetch(region[0], region[1], region[2]):
            read_name = read.query_name.split('/')[0]
            
            # Create lookup key
            lookup_key = (read.reference_name, read.reference_start)
            if lookup_key not in txt_df_indexed:
                continue
            
            read_data = txt_df_indexed[lookup_key]
            if read_name != read_data['name']:
                continue
                
            prob1 = read_data['prob_class_1']
            sequence = read_data['text']
            chrom = read_data['chr']
            start = read_data['start']
            end = read_data['end']
            
            # Filter relevant CpG sites using binary search
            cpg_mask = (
                (cpg_site_df['chr'] == chrom) & 
                (cpg_site_df['start'] >= start) & 
                (cpg_site_df['end'] <= end + 1)
            )
            cpg_sites = cpg_site_df[cpg_mask]
            
            if cpg_sites.empty:
                continue
                
            ref_pos = start
            seq_pos = 0
            # TODO: Parse BaseQuality for filtering
            for op, length in read.cigartuples:
                if op in [0, 7, 8]:  # Match/mismatch or sequence match
                    for cpg in cpg_sites.itertuples():
                        if cpg.start >= ref_pos and cpg.start < ref_pos + length:
                            seq_offset = cpg.start - ref_pos
                            seq_index = seq_pos + seq_offset
                            
                            if sequence[seq_index] == 'M':
                                status = 1
                            elif sequence[seq_index] == 'C':
                                status = 0
                            else:
                                continue
                                
                            cpg_prob_list.append([
                                cpg.chr, cpg.start, cpg.end,
                                read_name, status, prob1
                            ])
                    
                    ref_pos += length
                    seq_pos += length
                elif op == 2:  # Deletion
                    ref_pos += length
                elif op in [1, 4]:  # Insertion or soft clip
                    seq_pos += length
    
    return cpg_prob_list

def get_bam_chunks(bam_file, n_chunks=None):
    """Split BAM file into roughly equal chunks."""
    if n_chunks is None:
        n_chunks = multiprocessing.cpu_count()
        
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        chunks = []
        for ref in bam.references:
            ref_len = bam.get_reference_length(ref)
            chunk_size = ref_len // n_chunks + 1
            
            for i in range(0, ref_len, chunk_size):
                end = min(i + chunk_size, ref_len)
                chunks.append((ref, i, end))
    
    return chunks

def create_txt_index(txt_df):
    """Create an indexed dictionary from txt_df for faster lookups."""
    txt_df_dict = {}
    for _, row in txt_df.iterrows():
        txt_df_dict[(row['chr'], row['start'])] = row.to_dict()
    return txt_df_dict

def get_methylation_status(bam, txt, cpg_site_df, ncpus=None):
    """Extracts read methylation status based on CIGAR and txt file using parallel processing."""
    # Read and preprocess txt file
    txt_df = pd.read_csv(txt, sep='\t')
    if not txt_df['chr'].astype(str).str.contains('chr').all():
        txt_df['chr'] = 'chr' + txt_df['chr'].astype(str)
    
    # Create indexed version of txt_df
    txt_df_indexed = create_txt_index(txt_df)
    
    # Sort cpg_site_df for faster filtering
    cpg_site_df = cpg_site_df.sort_values(['chr', 'start', 'end'])
    
    # Split BAM file into chunks based on ncpus
    chunks = get_bam_chunks(bam, n_chunks=ncpus)
    chunk_infos = [(bam, chunk) for chunk in chunks]
    
    # Process chunks in parallel
    process_func = partial(process_chunk, txt_df_indexed=txt_df_indexed, cpg_site_df=cpg_site_df)
    
    all_results = []
    with ProcessPoolExecutor(max_workers=ncpus) as executor:
        for chunk_results in executor.map(process_func, chunk_infos):
            all_results.extend(chunk_results)
    
    return pd.DataFrame(
        all_results, 
        columns=['chr', 'start', 'end', 'name', 'status', 'prob_class_1']
    ).dropna()

def compute_methylation_rate(cpg_site_prob_df):
    """Calculates methylation rate for each CpG site."""
    rates = cpg_site_prob_df.groupby(['chr', 'start', 'end']).apply(
        lambda x: (x['prob_class_1'] * x['status']).sum() / x['prob_class_1'].sum() * 100 if x['prob_class_1'].sum() > 0 else np.nan
    ).dropna().reset_index(name='methylation_rate')
    
    return rates

@click.command()
@click.option('--bam', required=True, type=click.Path(exists=True), help='Path to the bam file')
@click.option('--txt', required=True, type=click.Path(exists=True), help='Path to the txt file')
@click.option('--bed', required=True, type=click.Path(exists=True), help='Path to the dmr bed file')
@click.option('--fasta', required=True, type=click.Path(exists=True), help='Path to the reference fasta file')
@click.option('--output', required=True, type=click.STRING, help='Output file prefix')
@click.option('--ncpus', type=int, default=None, help='Number of CPUs to use for parallel processing')
def main(bam, txt, bed, fasta, output, ncpus):
    cpg_site_df = get_cpg_sites(bed, fasta)
    cpg_site_prob_df = get_methylation_status(bam, txt, cpg_site_df, ncpus)
    cpg_site_df = compute_methylation_rate(cpg_site_prob_df)
    
    cpg_site_df.to_csv(f"{output}_CpG.bedGraph", index=False, sep='\t')
    cpg_site_prob_df.to_csv(f"{output}_cpg_prob.csv", index=False)

if __name__ == '__main__':
    main()
    