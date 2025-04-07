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

# Configure logging with rich
console = Console()
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

def extract_region_from_reference(reference, chrom, start, end):
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
    aligner = PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
    
    # Perform alignment
    alignments = aligner.align(ref_seq, seq_for_alignment)
    
    # Get the best alignment
    if not alignments:
        return None
    
    return alignments[0]

def initialize_depth_data_from_reference(reference_genome, dmr_info, coordinate_map):
    """
    Initialize depth data with all bases from the reference genome for all DMR regions.
    
    Args:
        reference_genome: Dictionary containing reference genome sequences.
        dmr_info: List of DMR region information.
        coordinate_map: Mapping from original to virtual coordinates.
        
    Returns:
        Tuple of (depth_map, prob_weighted_depth_map, cpg_sites_count)
        - depth_map: Dictionary mapping virtual positions to base counts (initialized with 0)
        - prob_weighted_depth_map: Dictionary mapping virtual positions to probability-weighted base counts (initialized with 0)
        - cpg_sites_count: Number of CpG sites found in the reference genome
    """
    logger.info("Initializing depth data from reference genome")
    
    # Initialize depth maps
    depth_map = {}
    prob_weighted_depth_map = {}
    cpg_sites_count = 0
    
    # Process each DMR region
    for dmr in tqdm(dmr_info, desc="Processing DMR regions"):
        chrom = dmr['chr_dmr']
        start = dmr['start_dmr']
        end = dmr['end_dmr']
        
        # Extract the region from reference
        region_seq = extract_region_from_reference(reference_genome, chrom, start, end)
        if region_seq is None or len(region_seq) == 0:
            logger.warning(f"Could not extract region for {chrom}:{start}-{end}")
            continue
        
        # Process each position in the region
        for i, base in enumerate(region_seq):
            genome_pos = start + i
            virtual_pos = coordinate_map.get((chrom, genome_pos))
            
            if virtual_pos is None:
                continue
            
            # Initialize the position if not already done
            if virtual_pos not in depth_map:
                depth_map[virtual_pos] = {}
                prob_weighted_depth_map[virtual_pos] = {}
            
            # Add the reference base with zero counts
            if base not in depth_map[virtual_pos]:
                depth_map[virtual_pos][base] = 0
                prob_weighted_depth_map[virtual_pos][base] = 0
            
            # Check for CpG sites (where current base is C and next base is G)
            if base == 'C' and i < len(region_seq) - 1 and region_seq[i+1] == 'G':
                cpg_sites_count += 1
                
                # For CpG sites, also add M (methylated cytosine) as a possible base
                if 'M' not in depth_map[virtual_pos]:
                    depth_map[virtual_pos]['M'] = 0
                    prob_weighted_depth_map[virtual_pos]['M'] = 0
    
    return depth_map, prob_weighted_depth_map, cpg_sites_count

def create_dmr_coordinate_map(df):
    """
    Create a mapping from original genome coordinates to virtual coordinates 
    by concatenating DMR regions.
    
    Args:
        df: DataFrame containing DMR regions with chr_dmr, start_dmr, and end_dmr columns.
        
    Returns:
        A tuple of (coordinate_map, virtual_length, dmr_info, total_dmr_length):
            - coordinate_map: Dict mapping (chr, pos) to virtual position
            - virtual_length: Total length of virtual coordinate space
            - dmr_info: List of DMR region information
            - total_dmr_length: Sum of all DMR region lengths
    """
    logger.info("Creating DMR coordinate mapping")
    
    # Get unique DMR regions
    dmr_regions = df[['chr_dmr', 'start_dmr', 'end_dmr']].drop_duplicates().reset_index(drop=True)
    
    # Sort DMR regions by chromosome and start position
    dmr_regions = dmr_regions.sort_values(['chr_dmr', 'start_dmr']).reset_index(drop=True)
    
    coordinate_map = {}
    dmr_info = []
    virtual_pos = 0
    total_dmr_length = 0
    
    for _, dmr in dmr_regions.iterrows():
        chrom = dmr['chr_dmr']
        start = dmr['start_dmr']
        end = dmr['end_dmr']
        length = end - start
        total_dmr_length += length
        
        # Map each position in this DMR to a virtual coordinate
        for i in range(length):
            coordinate_map[(chrom, start + i)] = virtual_pos + i
        
        dmr_info.append({
            'chr_dmr': chrom,
            'start_dmr': start,
            'end_dmr': end,
            'virtual_start': virtual_pos,
            'virtual_end': virtual_pos + length
        })
        
        virtual_pos += length
    
    return coordinate_map, virtual_pos, dmr_info, total_dmr_length

def update_base_depth_maps(reference_seq, alignment, ref_start_pos, original_seq, coordinate_map, 
                          depth_map, prob_weighted_depth_map, dmr_info, prob_class_1):
    """
    Update depth maps for each base in the alignment.
    
    Args:
        reference_seq: Original reference sequence
        alignment: Alignment object from PairwiseAligner
        ref_start_pos: Start position in the reference
        original_seq: Original query sequence with 'M' representing methylated cytosines
        coordinate_map: Mapping from original to virtual coordinates
        depth_map: Map to track raw depth at each virtual position
        prob_weighted_depth_map: Map to track probability-weighted depth
        dmr_info: Information about DMR regions
        prob_class_1: Probability value for weighting
    """
    # Extract aligned sequences with gaps
    aligned_ref = alignment.target
    aligned_seq = alignment.query
    
    # Map aligned positions to reference positions
    ref_to_align_pos = {}
    ref_idx = 0
    
    for align_idx, ref_char in enumerate(aligned_ref):
        if ref_char != '-':
            ref_to_align_pos[ref_idx] = align_idx
            ref_idx += 1
    
    # Map aligned positions to sequence positions
    align_to_seq_pos = {}
    seq_idx = 0
    
    for align_idx, seq_char in enumerate(aligned_seq):
        if seq_char != '-':
            align_to_seq_pos[align_idx] = seq_idx
            seq_idx += 1
    
    # Process all bases in the reference sequence
    for i in range(len(reference_seq)):
        ref_pos = ref_start_pos + i
        
        # Find the chromosome and DMR region for this position
        dmr_region = None
        for dmr in dmr_info:
            if (dmr['chr_dmr'] == dmr['chr_dmr'] and 
                dmr['start_dmr'] <= ref_pos < dmr['end_dmr']):
                dmr_region = dmr
                break
        
        if dmr_region is None:
            continue  # This position is not in any DMR region
        
        # Check if there's a virtual coordinate for this position
        coord_key = (dmr_region['chr_dmr'], ref_pos)
        if coord_key not in coordinate_map:
            continue
        
        virtual_pos = coordinate_map[coord_key]
        
        # Check if this position is aligned
        if i in ref_to_align_pos:
            align_idx = ref_to_align_pos[i]
            
            # Check if there's a corresponding position in the sequence
            if align_idx in align_to_seq_pos:
                seq_idx = align_to_seq_pos[align_idx]
                
                # Determine the base at this position
                base = original_seq[seq_idx] if seq_idx < len(original_seq) else reference_seq[i]
                
                # Update depth maps
                depth_map[virtual_pos][base] = depth_map[virtual_pos].get(base, 0) + 1
                prob_weighted_depth_map[virtual_pos][base] = prob_weighted_depth_map[virtual_pos].get(base, 0) + prob_class_1

def worker_init():
    """
    Initialize worker process by setting lower recursion limit.
    This helps prevent stack overflows in worker processes.
    """
    # Set a lower recursion limit to avoid stack overflow
    sys.setrecursionlimit(1000)  # Default is typically 1000

def process_batch_safe(batch_data):
    """
    Safe wrapper around process_batch to catch exceptions in worker processes.
    
    Args:
        batch_data: Tuple containing (batch, reference_genome, coordinate_map, dmr_info, n_bp_downstream, n_bp_upstream)
        
    Returns:
        Same as process_batch or error information if an exception occurs
    """
    try:
        batch, reference_genome, coordinate_map, dmr_info, n_bp_downstream, n_bp_upstream = batch_data
        return process_batch(batch, reference_genome, coordinate_map, dmr_info, n_bp_downstream, n_bp_upstream), None
    except Exception as e:
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        return None, error_info

def process_batch(batch, reference_genome, coordinate_map, dmr_info, n_bp_downstream=20, n_bp_upstream=20):
    """
    Process a batch of sequences from the parquet file.
    
    Args:
        batch: Pandas DataFrame containing a batch of sequence records
        reference_genome: Dictionary of chromosome sequences
        coordinate_map: Mapping from original to virtual coordinates
        dmr_info: Information about DMR regions
        n_bp_downstream: Number of base pairs downstream to include in reference
        n_bp_upstream: Number of base pairs upstream to include in reference
        
    Returns:
        Tuple of (depth_map, prob_weighted_depth_map)
    """
    # Initialize local depth maps for this batch
    virtual_length = max([dmr['virtual_end'] for dmr in dmr_info]) if dmr_info else 0
    depth_map = {}
    prob_weighted_depth_map = {}
    
    for _, row in batch.iterrows():
        chr_name = row['chr']
        start_pos = row['start']
        end_pos = row['end']
        sequence = row['seq']
        prob = row['prob_class_1']
        
        # Extract reference region with extra bases upstream and downstream
        ref_region = extract_region_from_reference(
            reference_genome, chr_name, start_pos - n_bp_upstream, end_pos + n_bp_downstream
        )
        
        if ref_region is None or len(ref_region) == 0:
            continue
        
        # Perform local re-alignment
        alignment = local_realign(sequence, ref_region)
        
        if alignment:
            # Extract aligned sequences with gaps
            aligned_ref = alignment.target
            aligned_seq = alignment.query
            
            # Map aligned positions to reference positions
            ref_to_align_pos = {}
            ref_idx = 0
            
            for align_idx, ref_char in enumerate(aligned_ref):
                if ref_char != '-':
                    ref_to_align_pos[ref_idx] = align_idx
                    ref_idx += 1
            
            # Map aligned positions to sequence positions
            align_to_seq_pos = {}
            seq_idx = 0
            
            for align_idx, seq_char in enumerate(aligned_seq):
                if seq_char != '-':
                    align_to_seq_pos[align_idx] = seq_idx
                    seq_idx += 1
            
            # Process all bases in the reference sequence
            for i in range(len(ref_region)):
                ref_pos = start_pos - n_bp_upstream + i
                
                # Find the chromosome and DMR region for this position
                dmr_region = None
                for dmr in dmr_info:
                    if (dmr['chr_dmr'] == chr_name and 
                        dmr['start_dmr'] <= ref_pos < dmr['end_dmr']):
                        dmr_region = dmr
                        break
                
                if dmr_region is None:
                    continue  # This position is not in any DMR region
                
                # Check if there's a virtual coordinate for this position
                coord_key = (dmr_region['chr_dmr'], ref_pos)
                if coord_key not in coordinate_map:
                    continue
                
                virtual_pos = coordinate_map[coord_key]
                
                # Initialize maps for this position if needed
                if virtual_pos not in depth_map:
                    depth_map[virtual_pos] = {}
                if virtual_pos not in prob_weighted_depth_map:
                    prob_weighted_depth_map[virtual_pos] = {}
                
                # Check if this position is aligned
                if i in ref_to_align_pos:
                    align_idx = ref_to_align_pos[i]
                    
                    # Check if there's a corresponding position in the sequence
                    if align_idx in align_to_seq_pos:
                        seq_idx = align_to_seq_pos[align_idx]
                        
                        # Determine the base at this position
                        base = sequence[seq_idx] if seq_idx < len(sequence) else ref_region[i]
                        
                        # Update local depth maps
                        depth_map[virtual_pos][base] = depth_map[virtual_pos].get(base, 0) + 1
                        prob_weighted_depth_map[virtual_pos][base] = prob_weighted_depth_map[virtual_pos].get(base, 0) + prob
    
    return depth_map, prob_weighted_depth_map

def process_parquet_file(parquet_file, reference_genome, batch_size=10000, num_workers=None, 
                       n_bp_downstream=0, n_bp_upstream=20, use_parallel=True):
    """
    Process sequences from a parquet file to determine depths and methylation status.
    
    Args:
        parquet_file: Path to the parquet file
        reference_genome: Dictionary of chromosome sequences
        batch_size: Number of records to process in each batch
        num_workers: Number of worker processes for parallel processing
        n_bp_downstream: Number of base pairs downstream to include in reference
        n_bp_upstream: Number of base pairs upstream to include in reference
        use_parallel: Whether to use parallel processing or sequential processing
        
    Returns:
        Tuple of (depth_data_df, num_reads, total_dmr_length, cpg_sites_count)
        - depth_data_df: DataFrame containing depth statistics
        - num_reads: Number of reads in the parquet file
        - total_dmr_length: Total length of all DMR regions
        - cpg_sites_count: Number of CpG sites found in the reference genome
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
    
    if use_parallel:
        logger.info(f"Processing parquet file: {parquet_file} with {num_workers} workers")
    else:
        logger.info(f"Processing parquet file: {parquet_file} sequentially")
    
    # Read parquet file
    logger.info(f"Reading parquet file: {parquet_file}")
    table = pq.read_table(parquet_file)
    df = table.to_pandas()
    num_reads = len(df)
    logger.info(f"Loaded {num_reads} reads from parquet file")
    
    # Verify required columns
    required_columns = ['chr', 'start', 'end', 'seq', 'prob_class_1', 'chr_dmr', 'start_dmr', 'end_dmr']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {', '.join(missing_columns)}")
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Create coordinate mapping
    coordinate_map, virtual_length, dmr_info, total_dmr_length = create_dmr_coordinate_map(df)
    logger.info(f"Created coordinate mapping with {virtual_length} virtual positions")
    logger.info(f"Total DMR regions length: {total_dmr_length} bp")
    
    # Initialize depth maps with reference genome data
    global_depth_map, global_prob_weighted_depth_map, cpg_sites_count = initialize_depth_data_from_reference(
        reference_genome, dmr_info, coordinate_map
    )
    logger.info(f"Found {cpg_sites_count} CpG sites in reference genome")
    
    # Process data in batches
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    # Prepare batches
    batches = []
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        batch = df.iloc[start_idx:end_idx]
        batches.append((batch, reference_genome, coordinate_map, dmr_info, n_bp_downstream, n_bp_upstream))
    
    if use_parallel:
        # Process in parallel
        with tqdm(total=total_batches, desc="Processing batches") as pbar:
            errors = []
            
            with ProcessPoolExecutor(max_workers=num_workers, initializer=worker_init) as executor:
                for batch_idx, future in enumerate(executor.map(process_batch_safe, batches)):
                    result, error = future
                    
                    if error:
                        error_msg = f"Error in batch {batch_idx}: {error['error']}"
                        logger.error(error_msg)
                        logger.debug(error['traceback'])
                        errors.append(error_msg)
                        pbar.update(1)
                        continue
                    
                    batch_depth_map, batch_prob_weighted_depth_map = result
                    
                    # Merge depth maps back to global maps
                    for pos in batch_depth_map:
                        if pos not in global_depth_map:
                            global_depth_map[pos] = {}
                        if pos not in global_prob_weighted_depth_map:
                            global_prob_weighted_depth_map[pos] = {}
                            
                        for base, count in batch_depth_map[pos].items():
                            global_depth_map[pos][base] = global_depth_map[pos].get(base, 0) + count
                    
                    # Merge probability-weighted depth maps
                    for pos in batch_prob_weighted_depth_map:
                        if pos not in global_prob_weighted_depth_map:
                            global_prob_weighted_depth_map[pos] = {}
                            
                        for base, prob_sum in batch_prob_weighted_depth_map[pos].items():
                            global_prob_weighted_depth_map[pos][base] = global_prob_weighted_depth_map[pos].get(base, 0) + prob_sum
                    
                    # Clean up batch-specific data to free memory
                    del batch_depth_map, batch_prob_weighted_depth_map
                    gc.collect()

                    pbar.update(1)
            
            if errors:
                logger.warning(f"Completed with {len(errors)} batch errors")
    else:
        # Process sequentially
        with tqdm(total=total_batches, desc="Processing batches") as pbar:
            for batch_data in batches:
                try:
                    batch, reference_genome, coordinate_map, dmr_info, n_bp_downstream, n_bp_upstream = batch_data
                    batch_depth_map, batch_prob_weighted_depth_map = process_batch(
                        batch, reference_genome, coordinate_map, dmr_info, n_bp_downstream, n_bp_upstream
                    )
                    
                    # Merge depth maps back to global maps
                    for pos in batch_depth_map:
                        if pos not in global_depth_map:
                            global_depth_map[pos] = {}
                        if pos not in global_prob_weighted_depth_map:
                            global_prob_weighted_depth_map[pos] = {}
                            
                        for base, count in batch_depth_map[pos].items():
                            global_depth_map[pos][base] = global_depth_map[pos].get(base, 0) + count
                    
                    # Merge probability-weighted depth maps
                    for pos in batch_prob_weighted_depth_map:
                        if pos not in global_prob_weighted_depth_map:
                            global_prob_weighted_depth_map[pos] = {}
                            
                        for base, prob_sum in batch_prob_weighted_depth_map[pos].items():
                            global_prob_weighted_depth_map[pos][base] = global_prob_weighted_depth_map[pos].get(base, 0) + prob_sum

                    # Clean up batch-specific data to free memory
                    del batch_depth_map, batch_prob_weighted_depth_map
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    logger.debug(traceback.format_exc())
                
                pbar.update(1)
    
    # Log some statistics about the depths
    total_depth = sum(sum(bases.values()) for bases in global_depth_map.values())
    logger.info(f"Total raw depth across all positions: {total_depth}")
    total_prob_depth = sum(sum(bases.values()) for bases in global_prob_weighted_depth_map.values())
    logger.info(f"Total probability-weighted depth: {total_prob_depth:.2f}")
    
    # Create depth data DataFrame
    depth_data = []
    
    logger.info("Preparing depth data")
    
    # Calculate total average prob-weighted depth for normalization
    total_prob_weighted_depth = 0
    total_positions_with_depth = 0
    
    # First pass: calculate total prob-weighted depth
    for virtual_pos in range(virtual_length):
        if virtual_pos in global_prob_weighted_depth_map:
            pos_depth = sum(global_prob_weighted_depth_map[virtual_pos].values())
            if pos_depth > 0:
                total_prob_weighted_depth += pos_depth
                total_positions_with_depth += 1
    
    # Calculate average prob-weighted depth (avoid division by zero)
    avg_prob_weighted_depth = total_prob_weighted_depth / total_positions_with_depth if total_positions_with_depth > 0 else 1
    logger.info(f"Average prob-weighted depth across all positions: {avg_prob_weighted_depth:.4f}")
    
    # Add virtual positions with their depths
    for virtual_pos in range(virtual_length):
        # Find the original coordinates for this virtual position
        original_coords = None
        for dmr in dmr_info:
            if dmr['virtual_start'] <= virtual_pos < dmr['virtual_end']:
                original_coords = {
                    'chr_dmr': dmr['chr_dmr'],
                    'start_dmr': dmr['start_dmr'],
                    'end_dmr': dmr['end_dmr'],
                    'chr': dmr['chr_dmr'],
                    'start': dmr['start_dmr'] + (virtual_pos - dmr['virtual_start']),
                    'end': dmr['start_dmr'] + (virtual_pos - dmr['virtual_start']) + 1
                }
                break
        
        if original_coords is None:
            continue
        
        # Get the bases and depths at this position
        if virtual_pos in global_depth_map:
            base_depths = global_depth_map[virtual_pos]
            base_prob_weighted_depths = global_prob_weighted_depth_map[virtual_pos]
            
            # Add a row for each base at this position
            for base, depth in base_depths.items():
                prob_weighted_depth = base_prob_weighted_depths.get(base, 0)
                
                # Normalize using total average prob-weighted depth
                normalized_depth = prob_weighted_depth / avg_prob_weighted_depth if avg_prob_weighted_depth > 0 else 0
                
                depth_data.append({
                    'index': virtual_pos,
                    'chr_dmr': original_coords['chr_dmr'],
                    'start_dmr': original_coords['start_dmr'],
                    'end_dmr': original_coords['end_dmr'],
                    'chr': original_coords['chr'],
                    'start': original_coords['start'],
                    'end': original_coords['end'],
                    'base': base,
                    'raw_depth': depth,
                    'prob_weighted_depth': prob_weighted_depth,
                    'normalized_depth': normalized_depth
                })
    
    # Convert to DataFrame
    logger.info("Creating depth data DataFrame")
    depth_data_df = pd.DataFrame(depth_data)
    
    return depth_data_df, num_reads, total_dmr_length, cpg_sites_count

@click.command()
@click.option('--parquet', required=True, type=click.Path(exists=True), help='Path to the parquet file')
@click.option('--fasta', required=True, type=click.Path(exists=True), help='Path to the reference genome FASTA file')
@click.option('--output', required=True, type=str, help='Prefix for output files')
@click.option('--batch-size', default=10000, type=int, help='Number of records to process in each batch')
@click.option('--num-workers', default=None, type=int, help='Number of worker processes for parallel processing')
@click.option('--n-bp-downstream', default=20, type=int, help='Number of base pairs downstream to include in the reference extract')
@click.option('--n-bp-upstream', default=0, type=int, help='Number of base pairs upstream to include in the reference extract')
@click.option('--sequential', is_flag=True, help='Run processing sequentially (no parallel workers)')
def main(parquet, fasta, output, batch_size, num_workers, n_bp_downstream, n_bp_upstream, sequential):
    """
    Process methylation sequencing data to calculate depth statistics at each position.
    
    This tool follows these steps:
    1. Reads methylation sequencing data from a parquet file
    2. Initializes positions with reference genome bases, including CpG sites
    3. Performs local realignment against the reference genome
    4. Creates a virtual coordinate system by concatenating DMR regions
    5. Calculates raw depth and probability-weighted depth at each position
    6. Normalizes the probability-weighted depth using the total average
    7. Outputs results to CSV
    """
    console.print(f"[bold green]Starting depth statistics calculation[/bold green]")
    
    try:
        # Read reference genome
        reference_genome = read_reference_genome(fasta)
        
        # Process parquet file
        depth_data_df, num_reads, total_dmr_length, cpg_sites_count = process_parquet_file(
            parquet, reference_genome, 
            batch_size=batch_size, 
            num_workers=num_workers,
            n_bp_downstream=n_bp_downstream,
            n_bp_upstream=n_bp_upstream,
            use_parallel=not sequential
        )
        
        # Write results to CSV
        csv_output = f"{output}_depth_stats.csv"
        depth_data_df.to_csv(csv_output, index=False)
        
        console.print(f"[bold green]Results written to: {csv_output}[/bold green]")
        console.print(f"[green]Total positions processed: {len(depth_data_df)}[/green]")
        console.print(f"[green]Total reads in parquet file: {num_reads}[/green]")
        console.print(f"[green]Total DMR regions length: {total_dmr_length} bp[/green]")
        console.print(f"[green]Total CpG sites identified: {cpg_sites_count}[/green]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        logger.exception("An error occurred during processing")
        sys.exit(1)

if __name__ == '__main__':
    main()
