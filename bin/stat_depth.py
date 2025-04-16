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
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm
import traceback
import gc
import pyarrow as pa

# Configure logging with rich
console = Console(width=180)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

def read_reference_genome(fasta_file):
    """
    Read the reference genome from FASTA file.
    
    Args:
        fasta_file: Path to the FASTA file containing the reference genome.
        
    Returns:
        A dictionary of chromosome sequences.
    """
    logger.info(f"Reading reference genome from: {fasta_file}")
    reference = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        # Convert sequence to uppercase
        reference[record.id] = str(record.seq).upper()
    return reference

def read_dmr_file(dmr_file):
    """
    Read the DMR BED file and return a dataframe.
    """
    return pd.read_csv(dmr_file, sep='\t', usecols=[0,1,2], names=['chr', 'start', 'end'])

def read_vcf_file(vcf_file):
    """
    Read the VCF file and return a dataframe.
    """ 
    compression_type = 'gzip' if vcf_file.endswith('.gz') else None

    vcf = pd.read_csv(
        vcf_file,
        sep='\t',
        comment='#',
        compression=compression_type,
        usecols=[0, 1, 3, 4],
        names=['chr', 'start', 'ref', 'alt']
    )
    vcf['end'] = vcf['start'].astype(int) + 1
    vcf = vcf[vcf['ref'].isin(['A', 'G']) & vcf['alt'].isin(['A', 'G'])]
    return vcf

def extract_vcf_inside_dmr(vcf, dmr):
    """
    Extract variants that are inside the DMR regions.
    """
    variants = []
    for _, region in dmr.iterrows():
        vcf_region = vcf[(vcf['chr'] == region['chr']) & 
                         (vcf['start'] >= region['start']) & 
                         (vcf['end'] <= region['end'])]
        variants.append(vcf_region)

    return pd.concat(variants, axis=0)

def extract_sequence_from_reference(reference, chrom, start, end):
    """
    Extract a region from the reference genome.
    
    Args:
        reference: Dictionary containing reference genome sequences.
        chrom: Chromosome name.
        start: Start position (0-based).
        end: End position (exclusive).
        
    Returns:
        The extracted sequence or None if the region is invalid.
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
    Perform local re-alignment between a read sequence and reference sequence.
    
    Args:
        seq: The query sequence, potentially with 'M' representing methylated cytosines.
        ref_seq: The reference sequence.
        
    Returns:
        The best alignment or None if no alignment was found.
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

def initialize_position_dict(reference_genome, dmr_df, variants_df):
    """
    Initialize a dictionary that records information for each position inside DMR regions.
    
    Args:
        reference_genome: Dictionary of chromosome sequences.
        dmr_df: DataFrame containing DMR regions.
        variants_df: DataFrame containing variant positions.
        
    Returns:
        Dictionary with position coordinates as keys, and dictionaries of base info as values.
    """
    position_dict = {}
    
    # Convert variants_df to a set of tuples for faster lookup
    variant_positions = set(zip(variants_df['chr'], variants_df['start']))
    
    # Iterate through each DMR region
    for _, region in dmr_df.iterrows():
        chr_name = region['chr']
        start_pos = region['start']
        end_pos = region['end']
        
        # Extract reference sequence for this region
        ref_seq = extract_sequence_from_reference(reference_genome, chr_name, start_pos, end_pos)
        if ref_seq is None:
            logger.warning(f"Skipping DMR region {chr_name}:{start_pos}-{end_pos} - could not extract reference sequence")
            continue
        
        # Iterate through each position in the DMR region
        for pos_offset in range(len(ref_seq)):
            pos = start_pos + pos_offset
            ref_base = ref_seq[pos_offset]
            
            # Position key
            pos_key = (chr_name, pos)
            
            # Initialize dictionary for this position
            if pos_key not in position_dict:
                position_dict[pos_key] = {
                    'chr_dmr': chr_name,
                    'start_dmr': start_pos,
                    'end_dmr': end_pos,
                    'bases': {}
                }
            
            # Check if this is a variant position
            if pos_key in variant_positions:
                # Find the variant entry
                variant = variants_df[(variants_df['chr'] == chr_name) & (variants_df['start'] == pos)]
                if not variant.empty:
                    variant_row = variant.iloc[0]
                    ref_allele = variant_row['ref']
                    alt_allele = variant_row['alt']
                    
                    # Add both possible bases to the dictionary
                    position_dict[pos_key]['bases'][ref_allele] = {'raw_depth': 0, 'prob_weighted_depth': 0, 'normalized_depth': 0}
                    position_dict[pos_key]['bases'][alt_allele] = {'raw_depth': 0, 'prob_weighted_depth': 0, 'normalized_depth': 0}
            
            # Check if this is a CpG site (C followed by G)
            elif pos_offset < len(ref_seq) - 1 and ref_base == 'C' and ref_seq[pos_offset + 1] == 'G':
                # CpG site - C can be either C or M (methylated C)
                position_dict[pos_key]['bases']['C'] = {'raw_depth': 0, 'prob_weighted_depth': 0, 'normalized_depth': 0}
                position_dict[pos_key]['bases']['M'] = {'raw_depth': 0, 'prob_weighted_depth': 0, 'normalized_depth': 0}
            else:
                # Normal position - just record the reference base
                position_dict[pos_key]['bases'][ref_base] = {'raw_depth': 0, 'prob_weighted_depth': 0, 'normalized_depth': 0}
    
    return position_dict

def process_dmr_batch(args):
    batch_df, batch_dmrs, reference_genome, dmr_df, variants_df, n_bp_downstream, n_bp_upstream = args
    
    # Filter DMR entries for the current batch
    batch_dmrs_set = set(batch_dmrs)
    batch_dmr_df = dmr_df[dmr_df.apply(lambda row: (row['chr'], row['start'], row['end']) in batch_dmrs_set, axis=1)]
    
    # Filter variants for the current batch
    batch_variants_list = []
    for dmr in batch_dmrs:
        dchr, dstart, dend = dmr
        filtered = variants_df[(variants_df['chr'] == dchr) & (variants_df['start'] >= dstart) & (variants_df['start'] < dend)]
        if not filtered.empty:
            batch_variants_list.append(filtered)
    if batch_variants_list:
        batch_variants_df = pd.concat(batch_variants_list, axis=0)
    else:
        batch_variants_df = pd.DataFrame(columns=variants_df.columns)
    
    # Initialize a dedicated position dictionary for this batch based on its DMRs and variants
    position_dict = initialize_position_dict(reference_genome, batch_dmr_df, batch_variants_df)
    
    # Process each row in the batch
    for _, row in batch_df.iterrows():
        chr_name = row['chr']
        start_pos = row['start']
        end_pos = row['end']
        sequence = row['seq']
        prob_class_1 = row['prob_class_1']
        
        ref_seq = extract_sequence_from_reference(reference_genome, chr_name, start_pos - n_bp_upstream, end_pos + n_bp_downstream)
        if ref_seq is None or len(ref_seq) == 0:
            continue
        alignment = local_realign(sequence, ref_seq)
        if alignment is None:
            continue
        try:
            aligned_target = str(alignment[0])
            aligned_query = str(alignment[1])
            ref_pos = alignment.coordinates[0][0]
            query_pos = alignment.coordinates[1][0]

            # Recover the M bases in the query sequence
            recovered_query = ''
            for q_base in aligned_query:
                if q_base != '-':
                    recovered_query += sequence[query_pos]
                    query_pos+=1
                else:
                    recovered_query += '-'
            aligned_query = recovered_query

            for t_base, q_base in zip(aligned_target, aligned_query):
                if t_base == '-' and q_base == '-':
                    continue
                if t_base != '-':
                    genomic_coord = (chr_name, start_pos - n_bp_upstream + ref_pos)
                    if genomic_coord in position_dict:
                        if q_base != '-' and q_base in position_dict[genomic_coord]['bases']:
                            position_dict[genomic_coord]['bases'][q_base]['raw_depth'] += 1
                            position_dict[genomic_coord]['bases'][q_base]['prob_weighted_depth'] += prob_class_1
                ref_pos += 1
            
        except Exception as e:
            logger.warning(f"Error processing alignment: {str(e)}")
            continue
    return position_dict

def process_parquet_file(parquet_file, reference_genome, dmr_df, variants_df, batch_size=10000, 
                        num_workers=None, n_bp_downstream=20, n_bp_upstream=20):
    """
    Process sequences from a parquet file to calculate depth statistics.
    
    Args:
        parquet_file: Path to the parquet file containing read data.
        reference_genome: Dictionary of chromosome sequences.
        dmr_df: DataFrame containing DMR regions.
        variants_df: DataFrame containing variant positions.
        batch_size: Number of records to process in each batch.
        num_workers: Number of worker processes for parallel processing.
        n_bp_downstream: Number of base pairs downstream to include in the reference extract.
        n_bp_upstream: Number of base pairs upstream to include in the reference extract.
        
    Returns:
        Two DataFrames: one for regular positions and one for variant positions.
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    logger.info(f"Processing parquet file: {parquet_file} with {num_workers} workers")
    
    # Initialize position dictionary
    position_dict = initialize_position_dict(reference_genome, dmr_df, variants_df)
    logger.info(f"Initialized position dictionary with {len(position_dict)} positions")
    
    try:
        # Read parquet file
        table = pq.read_table(parquet_file)
        df = table.to_pandas()
        
        # Verify required columns
        required_columns = ['chr_dmr', 'start_dmr', 'end_dmr', 'chr', 'start', 'end', 'seq', 'prob_class_1']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {', '.join(missing_columns)}")
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Track all DMRs present in the parquet file
        processed_dmrs = set()
        
        # Group the dataframe by complete DMR boundaries using the 'chr_dmr', 'start_dmr', 'end_dmr' columns
        dmr_groups = list(df.groupby(['chr_dmr', 'start_dmr', 'end_dmr']))
        batches = []
        for i in range(0, len(dmr_groups), batch_size):
            batch_groups = dmr_groups[i:i+batch_size]
            batch_df = pd.concat([group for key, group in batch_groups], axis=0)
            batch_dmrs = [key for key, group in batch_groups]  # Each key is a tuple (chr_dmr, start_dmr, end_dmr)
            # Add DMRs to the set of processed DMRs
            processed_dmrs.update(batch_dmrs)
            batches.append((batch_df, batch_dmrs))
        total_batches = len(batches)
        
        merged_position_dict = {}
        with tqdm(total=total_batches, desc="Processing batches") as pbar:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for batch_df, batch_dmrs in batches:
                    args = (batch_df, batch_dmrs, reference_genome, dmr_df, variants_df, n_bp_downstream, n_bp_upstream)
                    futures.append(executor.submit(process_dmr_batch, args))
                for future in futures:
                    batch_result = future.result()
                    # Since batches are DMR-isolated, merge by updating the dictionary directly
                    merged_position_dict.update(batch_result)
                    pbar.update(1)
                    gc.collect()
        
        # Check for missing DMRs and initialize them
        logger.info("Checking for missing DMR regions...")
        all_dmrs = set(zip(dmr_df['chr'], dmr_df['start'], dmr_df['end']))
        missing_dmrs = all_dmrs - processed_dmrs
        
        if missing_dmrs:
            logger.info(f"Found {len(missing_dmrs)} missing DMR regions, initializing with depth 0")
            # Create position entries for missing DMRs
            missing_dmr_df = dmr_df[dmr_df.apply(lambda row: (row['chr'], row['start'], row['end']) in missing_dmrs, axis=1)]
            missing_variants_list = []
            
            for _, dmr_row in tqdm(missing_dmr_df.iterrows(), desc="Processing missing DMRs", total=len(missing_dmr_df)):
                dchr, dstart, dend = dmr_row['chr'], dmr_row['start'], dmr_row['end']
                # Check for variants in this DMR
                filtered = variants_df[(variants_df['chr'] == dchr) & 
                                      (variants_df['start'] >= dstart) & 
                                      (variants_df['start'] < dend)]
                if not filtered.empty:
                    missing_variants_list.append(filtered)
            
            if missing_variants_list:
                missing_variants_df = pd.concat(missing_variants_list, axis=0)
            else:
                missing_variants_df = pd.DataFrame(columns=variants_df.columns)
            
            # Initialize positions for missing DMRs
            missing_positions = initialize_position_dict(reference_genome, missing_dmr_df, missing_variants_df)
            
            # Merge with existing positions
            merged_position_dict.update(missing_positions)
            
        # Use merged results from all batches
        position_dict = merged_position_dict
        
        # Calculate the total average prob_weighted_depth for normalization
        total_prob_weighted_depth = 0
        total_positions = 0
        
        for pos_data in position_dict.values():
            for base_data in pos_data['bases'].values():
                total_prob_weighted_depth += base_data['prob_weighted_depth']
                if base_data['raw_depth'] > 0:
                    total_positions += 1
        
        avg_prob_weighted_depth = total_prob_weighted_depth / max(1, total_positions)
        logger.info(f"Average prob-weighted depth: {avg_prob_weighted_depth:.2f}")
        
        # Normalize the prob_weighted_depth values
        for pos_key, pos_data in position_dict.items():
            for base, base_data in pos_data['bases'].items():
                if avg_prob_weighted_depth > 0:
                    base_data['normalized_depth'] = base_data['prob_weighted_depth'] / avg_prob_weighted_depth
        
        # Convert variant and non-variant positions to DataFrames
        variant_positions = set(zip(variants_df['chr'], variants_df['start']))
        
        variant_rows = []
        depth_rows = []
        
        for pos_key, pos_data in position_dict.items():
            chr_name, pos = pos_key
            chr_dmr = pos_data['chr_dmr']
            start_dmr = pos_data['start_dmr']
            end_dmr = pos_data['end_dmr']
            
            for base, base_data in pos_data['bases'].items():
                row = {
                    'chr_dmr': chr_dmr,
                    'start_dmr': start_dmr,
                    'end_dmr': end_dmr,
                    'chr': chr_name,
                    'start': pos,
                    'end': pos + 1,
                    'base': base,
                    'raw_depth': base_data['raw_depth'],
                    'prob_weighted_depth': base_data['prob_weighted_depth'],
                    'normalized_depth': base_data['normalized_depth']
                }
                
                if pos_key in variant_positions:
                    variant_rows.append(row)
                else:
                    depth_rows.append(row)
        
        # Convert to DataFrames
        depth_df = pd.DataFrame(depth_rows)
        variant_df = pd.DataFrame(variant_rows)
        
        # Sort the dataframes by chromosomal coordinates
        if not depth_df.empty:
            depth_df = depth_df.sort_values(by=['chr', 'start', 'end'])
        
        if not variant_df.empty:
            variant_df = variant_df.sort_values(by=['chr', 'start', 'end'])
        
        logger.info(f"Processed {len(depth_df) + len(variant_df)} positions")
        logger.info(f"Found {len(variant_df)} variant positions and {len(depth_df)} non-variant positions")
        
        return depth_df, variant_df
    
    except Exception as e:
        logger.error(f"Error processing parquet file: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@click.command()
@click.option('--parquet', required=True, type=click.Path(exists=True), help='Path to the parquet file')
@click.option('--fasta', required=True, type=click.Path(exists=True), help='Path to the reference genome FASTA file')
@click.option('--dmr', required=True, type=click.Path(exists=True), help='Path to the DMR BED file')
@click.option('--vcf', required=True, type=click.Path(exists=True), help='Path to the VCF file')
@click.option('--output', required=True, type=str, help='Prefix for output files')
@click.option('--batch-size', default=50, type=int, help='Number of records to process in each batch')
@click.option('--num-workers', default=None, type=int, help='Number of worker processes for parallel processing')
@click.option('--n-bp-downstream', default=20, type=int, help='Number of base pairs downstream to include in the reference extract')
@click.option('--n-bp-upstream', default=20, type=int, help='Number of base pairs upstream to include in the reference extract')
def main(parquet, fasta, dmr, vcf, output, batch_size, num_workers, n_bp_downstream, n_bp_upstream):
    try:
        # Read reference genome
        reference_genome = read_reference_genome(fasta)

        # Read DMR file
        dmr_df = read_dmr_file(dmr)

        # Read VCF file
        vcf_df = read_vcf_file(vcf)

        # Extract variants that are inside the DMR regions
        variants_df = extract_vcf_inside_dmr(vcf_df, dmr_df)
        
        # Process parquet file
        depth_data_df, variation_df = process_parquet_file(
            parquet, reference_genome, dmr_df, variants_df,
            batch_size=batch_size, 
            num_workers=num_workers,
            n_bp_downstream=n_bp_downstream,
            n_bp_upstream=n_bp_upstream
        )
        
        # Write results to parquet files
        depth_output = f"{output}_depth.parquet"
        variation_output = f"{output}_variants.parquet"
        
        # Write to parquet files
        pq.write_table(pa.Table.from_pandas(depth_data_df), depth_output)
        pq.write_table(pa.Table.from_pandas(variation_df), variation_output)
        
        logger.info(f"Wrote depth statistics to: {depth_output}")
        logger.info(f"Wrote variant statistics to: {variation_output}")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        logger.exception("An error occurred during processing")
        sys.exit(1)

if __name__ == '__main__':
    main()
