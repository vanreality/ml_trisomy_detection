import pandas as pd
import numpy as np
import click
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial
import sys
import os
import logging
import pyarrow.parquet as pq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def read_reference_genome(fasta_file):
    """
    Read the reference genome from FASTA file.
    Returns a dictionary of chromosome sequences.
    """
    logger.info(f"Reading reference genome from: {fasta_file}")
    reference = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        # Convert sequence to uppercase
        reference[record.id] = str(record.seq).upper()
    return reference

def extract_region_from_reference(reference, chrom, start, end):
    """
    Extract a region from the reference genome.
    """
    if chrom not in reference:
        # Try adding/removing 'chr' prefix
        alt_chrom = chrom.replace('chr', '') if chrom.startswith('chr') else f'chr{chrom}'
        if alt_chrom in reference:
            chrom = alt_chrom
        else:
            logger.error(f"Chromosome {chrom} not found in reference genome")
            return None
    
    try:
        return reference[chrom][start:end]
    except IndexError:
        logger.error(f"Invalid coordinates for chromosome {chrom}: {start}-{end}")
        return None

def local_realign(seq, ref_seq):
    """
    Perform local re-alignment between a read sequence and reference sequence
    using the Bio.Align.PairwiseAligner.
    Returns the alignment.
    """
    # Replace 'M' with 'C' for alignment purposes
    seq_for_alignment = seq.replace('M', 'C')
    
    # Configure aligner for local alignment
    aligner = PairwiseAligner(scoring='blastn')
    aligner.mode = 'local'
    # aligner.match_score = 2
    # aligner.mismatch_score = -1
    # aligner.open_gap_score = -2
    # aligner.extend_gap_score = -0.5
    
    # Perform alignment
    alignments = aligner.align(ref_seq, seq_for_alignment)
    
    # Get the best alignment
    if not alignments:
        return None
    
    return alignments[0]

def identify_cpg_sites(reference_seq, alignment, ref_start_pos, original_query):
    """
    Identify CpG sites in the alignment and determine methylation status.
    
    Args:
        reference_seq: Original reference sequence
        alignment: Alignment object from PairwiseAligner
        ref_start_pos: Start position in the reference
        original_seq: Original query sequence with 'M' representing methylated cytosines
        
    Returns:
        List of CpG sites with their methylation status
    """
    cpg_sites = []
    
    # Extract aligned sequences with gaps
    aligned_ref = str(alignment[0])
    aligned_query = str(alignment[1])
    aligned_ref_pos = alignment.coordinates[0][0]
    aligned_query_pos = alignment.coordinates[1][0]

    # Recover the M bases in the query sequence
    recovered_query = ''
    for q_base in aligned_query:
        if q_base != '-':
            recovered_query += original_query[aligned_query_pos]
            aligned_query_pos+=1
        else:
            recovered_query += '-'
    aligned_query = recovered_query

    # Find all CpG sites in the reference
    idx_to_skip = 0
    for i in range(len(aligned_ref) - 1):
        if aligned_ref[i] == '-':
            idx_to_skip += 1
        if aligned_ref[i:i+2] == "CG":
            # CpG site in reference at position ref_pos
            ref_pos = ref_start_pos + aligned_ref_pos + i - idx_to_skip

            # Check if the cytosine is methylated (represented as 'M')
            methylation_status = 0
            if aligned_query[i] == 'M':
                methylation_status = 1
                
            cpg_sites.append({
                'cpg_start': ref_pos,
                'cpg_end': ref_pos + 2,
                'status': methylation_status
            })
    
    return cpg_sites

def process_batch(batch, reference_genome, n_bp_downstream=20, n_bp_upstream=20):
    """
    Process a batch of sequences from the parquet file.
    
    Args:
        batch: Pandas DataFrame containing a batch of sequence records
        reference_genome: Dictionary of chromosome sequences
        n_bp_downstream: Number of base pairs downstream to include in the reference extract
        n_bp_upstream: Number of base pairs upstream to include in the reference extract
        
    Returns:
        List of dictionaries with CpG methylation data
    """
    results = []
    
    for _, row in batch.iterrows():
        chr_name = row['chr']
        start_pos = row['start']
        end_pos = row['end']
        sequence = row['seq']
        prob = row['prob_class_1']
        name = row['name']
        insert_size = row['insert_size']
        chr_dmr = row['chr_dmr']
        start_dmr = row['start_dmr']
        end_dmr = row['end_dmr']
        
        # Extract reference region with extra bases upstream and downstream
        ref_region = extract_region_from_reference(
            reference_genome, chr_name, start_pos - n_bp_upstream, end_pos + n_bp_downstream
        )
        
        if ref_region is None or len(ref_region) == 0:
            continue
        
        # Perform local re-alignment
        alignment = local_realign(sequence, ref_region)
        
        if alignment:
            # Identify CpG sites and methylation status
            cpg_sites = identify_cpg_sites(
                ref_region, alignment, start_pos - n_bp_upstream, sequence
            )
            
            # Add data to results
            for cpg in cpg_sites:
                results.append({
                    'chr': chr_name,
                    'start': cpg['cpg_start'],
                    'end': cpg['cpg_end'],
                    'status': cpg['status'],
                    'prob_class_1': prob,
                    'name': name,
                    'insert_size': insert_size,
                    'chr_dmr': chr_dmr,
                    'start_dmr': start_dmr,
                    'end_dmr': end_dmr
                })
    
    return results

def process_parquet_file(parquet_file, reference_genome, batch_size=1000, num_workers=None, n_bp_downstream=20, n_bp_upstream=20):
    """
    Process sequences from a parquet file to determine CpG methylation status.
    
    Args:
        parquet_file: Path to the parquet file
        reference_genome: Dictionary of chromosome sequences
        batch_size: Number of records to process in each batch
        num_workers: Number of worker processes for parallel processing
        n_bp_downstream: Number of base pairs downstream to include in the reference extract
        n_bp_upstream: Number of base pairs upstream to include in the reference extract
        
    Returns:
        Pandas DataFrame with CpG methylation data
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    logger.info(f"Processing parquet file: {parquet_file} with {num_workers} workers")
    
    # Read parquet file
    table = pq.read_table(parquet_file)
    df = table.to_pandas()
    
    # Verify required columns
    required_columns = ['chr', 'start', 'end', 'seq', 'prob_class_1', 'chr_dmr', 'start_dmr', 'end_dmr']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {', '.join(missing_columns)}")
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Process data in batches
    all_results = []
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        batch_processor = partial(process_batch, reference_genome=reference_genome, n_bp_downstream=n_bp_downstream, n_bp_upstream=n_bp_upstream)
        
        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            batch = df.iloc[start_idx:end_idx]
            
            logger.info(f"Processing batch {i+1}/{total_batches} with {len(batch)} records")
            batch_results = executor.submit(batch_processor, batch)
            all_results.extend(batch_results.result())
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    return results_df

def calculate_weighted_methylation(results_df):
    """
    Calculate the weighted methylation rate for each CpG site using the formula:
    sum(status * prob_class_1) / sum(prob_class_1) * 100
    
    Args:
        results_df: DataFrame containing CpG methylation data
        
    Returns:
        DataFrame with weighted methylation rates
    """
    logger.info("Calculating weighted methylation rates")
    
    # Group by CpG site location
    grouped = results_df.groupby(['chr', 'start', 'end'])
    
    # Calculate weighted methylation rate
    weighted_meth = grouped.apply(
        lambda x: (np.sum(x['status'] * x['prob_class_1']) / np.sum(x['prob_class_1'])) * 100
    ).reset_index(name='rate')
    
    return weighted_meth

@click.command()
@click.option('--parquet', required=True, type=click.Path(exists=True), help='Path to the parquet file')
@click.option('--fasta', required=True, type=click.Path(exists=True), help='Path to the reference genome FASTA file')
@click.option('--output', required=True, type=str, help='Prefix for output files')
@click.option('--batch-size', default=100000, type=int, help='Number of records to process in each batch')
@click.option('--num-workers', default=None, type=int, help='Number of worker processes for parallel processing')
@click.option('--n-bp-downstream', default=20, type=int, help='Number of base pairs downstream to include in the reference extract')
@click.option('--n-bp-upstream', default=0, type=int, help='Number of base pairs upstream to include in the reference extract')
def main(parquet, fasta, output, batch_size, num_workers, n_bp_downstream, n_bp_upstream):
    """
    Process sequencing data to determine CpG methylation status.
    """
    # Read reference genome
    reference_genome = read_reference_genome(fasta)
    
    # Process parquet file
    results_df = process_parquet_file(
        parquet, reference_genome, 
        batch_size=batch_size, 
        num_workers=num_workers,
        n_bp_downstream=n_bp_downstream,
        n_bp_upstream=n_bp_upstream
    )
    
    # Write detailed results to CSV
    csv_output = f"{output}_cpg_prob.csv"
    results_df.to_csv(csv_output, index=False)
    logger.info(f"Detailed results written to: {csv_output}")
    
    # Calculate weighted methylation rates
    weighted_meth_df = calculate_weighted_methylation(results_df)
    
    # Write weighted methylation rates to bedGraph file
    bedgraph_output = f"{output}_CpG.bedGraph"
    weighted_meth_df.to_csv(
        bedgraph_output, 
        columns=['chr', 'start', 'end', 'rate'],
        sep='\t',
        header=False,
        index=False
    )
    logger.info(f"Weighted methylation rates written to: {bedgraph_output}")

if __name__ == '__main__':
    main()
