import pandas as pd
import polars as pl
import numpy as np
import click
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial
import sys
import os
import logging
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.console import Console
import gc

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

def process_batch(batch, reference_genome, seq_column_name, n_bp_downstream=20, n_bp_upstream=20):
    """
    Process a batch of sequences from the parquet file with optimized memory usage.
    
    Args:
        batch: Pandas DataFrame containing a batch of sequence records
        reference_genome: Dictionary of chromosome sequences
        seq_column_name: Name of the column containing sequences ('seq' or 'text')
        n_bp_downstream: Number of base pairs downstream to include in the reference extract
        n_bp_upstream: Number of base pairs upstream to include in the reference extract
        
    Returns:
        List of dictionaries with CpG methylation data
    """
    results = []
    
    # Convert to numpy arrays for faster iteration
    chr_names = batch['chr'].values
    start_positions = batch['start'].values
    end_positions = batch['end'].values
    sequences = batch[seq_column_name].values
    probs = batch['prob_class_1'].values
    names = batch['name'].values if 'name' in batch.columns else [None] * len(batch)
    insert_sizes = batch['insert_size'].values if 'insert_size' in batch.columns else [None] * len(batch)
    chr_dmrs = batch['chr_dmr'].values
    start_dmrs = batch['start_dmr'].values
    end_dmrs = batch['end_dmr'].values
    
    # Process each row
    for i in range(len(batch)):
        chr_name = chr_names[i]
        start_pos = start_positions[i]
        end_pos = end_positions[i]
        sequence = sequences[i]
        prob = probs[i]
        name = names[i]
        insert_size = insert_sizes[i]
        chr_dmr = chr_dmrs[i]
        start_dmr = start_dmrs[i]
        end_dmr = end_dmrs[i]
        
        # Skip if sequence is None or empty
        if not sequence or pd.isna(sequence):
            continue
            
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
            
            # Add data to results (pre-allocate if possible for better performance)
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
    
    # Clear batch from memory in this process
    del batch
    gc.collect()
    
    return results

def process_parquet_file(parquet_file, reference_genome, batch_size=10000, num_workers=None, n_bp_downstream=20, n_bp_upstream=20):
    """
    Process sequences from a parquet file to determine CpG methylation status using streaming approach.
    
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
        num_workers = min(multiprocessing.cpu_count(), 8)  # Limit to avoid excessive memory usage
    
    console = Console()
    console.print(f"[green]Processing parquet file: {parquet_file} with {num_workers} workers[/green]")
    
    # First, scan the file to get total rows and detect sequence column
    console.print("[cyan]Scanning file to detect columns and count rows...[/cyan]")
    
    # Use polars to efficiently scan file metadata
    df_lazy = pl.scan_parquet(parquet_file)
    columns = df_lazy.columns
    
    # Automatically detect sequence column name
    seq_column_name = None
    if 'seq' in columns:
        seq_column_name = 'seq'
    elif 'text' in columns:
        seq_column_name = 'text'
    else:
        logger.error("No sequence column found. Expected either 'seq' or 'text' column.")
        raise ValueError("No sequence column found. Expected either 'seq' or 'text' column.")
    
    console.print(f"[green]Using sequence column: {seq_column_name}[/green]")
    
    # Verify required columns
    required_columns = ['chr', 'start', 'end', seq_column_name, 'prob_class_1', 'chr_dmr', 'start_dmr', 'end_dmr']
    missing_columns = [col for col in required_columns if col not in columns]
    if missing_columns:
        logger.error(f"Missing required columns: {', '.join(missing_columns)}")
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Get total row count efficiently
    total_rows = df_lazy.select(pl.len()).collect().item()
    total_batches = (total_rows + batch_size - 1) // batch_size
    
    console.print(f"[cyan]Total rows: {total_rows:,}, Processing in {total_batches} batches of {batch_size:,} records[/cyan]")
    
    # Process data in streaming batches with progress tracking
    all_results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Processing batches", total=total_batches)
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            batch_processor = partial(
                process_batch, 
                reference_genome=reference_genome, 
                seq_column_name=seq_column_name, 
                n_bp_downstream=n_bp_downstream, 
                n_bp_upstream=n_bp_upstream
            )
            
            # Submit all batches as futures
            future_to_batch = {}
            
            for i in range(total_batches):
                offset = i * batch_size
                
                # Read batch using polars streaming
                batch_df = (
                    df_lazy
                    .slice(offset, batch_size)
                    .select(required_columns)
                    .collect()
                    .to_pandas()
                )
                
                # Only submit if batch is not empty
                if len(batch_df) > 0:
                    future = executor.submit(batch_processor, batch_df)
                    future_to_batch[future] = i + 1
                    
                    # Clear batch from memory
                    del batch_df
                    gc.collect()
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_num = future_to_batch[future]
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    progress.advance(task)
                    
                    # Periodic garbage collection to manage memory
                    if batch_num % 10 == 0:
                        gc.collect()
                        
                except Exception as exc:
                    logger.error(f'Batch {batch_num} generated an exception: {exc}')
                    raise exc
    
    console.print(f"[green]Processed {len(all_results):,} CpG sites[/green]")
    
    # Convert results to DataFrame efficiently
    if all_results:
        results_df = pd.DataFrame(all_results)
    else:
        # Return empty DataFrame with correct columns if no results
        results_df = pd.DataFrame(columns=['chr', 'start', 'end', 'status', 'prob_class_1', 'name', 'insert_size', 'chr_dmr', 'start_dmr', 'end_dmr'])
    
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
    console = Console()
    console.print("[cyan]Calculating weighted methylation rates...[/cyan]")
    
    if len(results_df) == 0:
        console.print("[yellow]No CpG sites found to calculate methylation rates[/yellow]")
        return pd.DataFrame(columns=['chr', 'start', 'end', 'rate'])
    
    # Group by CpG site location
    grouped = results_df.groupby(['chr', 'start', 'end'])
    
    console.print(f"[cyan]Found {len(grouped)} unique CpG sites[/cyan]")
    
    # Calculate weighted methylation rate
    weighted_meth = grouped.apply(
        lambda x: (np.sum(x['status'] * x['prob_class_1']) / np.sum(x['prob_class_1'])) * 100
    ).reset_index(name='rate')
    
    console.print(f"[green]Calculated methylation rates for {len(weighted_meth)} CpG sites[/green]")
    
    return weighted_meth

@click.command()
@click.option('--parquet', required=True, type=click.Path(exists=True), help='Path to the parquet file')
@click.option('--fasta', required=True, type=click.Path(exists=True), help='Path to the reference genome FASTA file')
@click.option('--output', required=True, type=str, help='Prefix for output files')
@click.option('--batch-size', default=10000, type=int, help='Number of records to process in each batch')
@click.option('--num-workers', default=None, type=int, help='Number of worker processes for parallel processing')
@click.option('--n-bp-downstream', default=20, type=int, help='Number of base pairs downstream to include in the reference extract')
@click.option('--n-bp-upstream', default=0, type=int, help='Number of base pairs upstream to include in the reference extract')
def main(parquet, fasta, output, batch_size, num_workers, n_bp_downstream, n_bp_upstream):
    """
    Process sequencing data to determine CpG methylation status with optimized performance.
    """
    console = Console()
    
    console.print("[bold green]Starting CpG methylation analysis[/bold green]")
    console.print(f"Input file: {parquet}")
    console.print(f"Reference genome: {fasta}")
    console.print(f"Output prefix: {output}")
    console.print(f"Batch size: {batch_size:,}")
    console.print(f"Workers: {num_workers or 'auto'}")
    
    # Read reference genome
    console.print("\n[cyan]Reading reference genome...[/cyan]")
    reference_genome = read_reference_genome(fasta)
    console.print(f"[green]Loaded {len(reference_genome)} chromosomes[/green]")
    
    # Process parquet file
    console.print("\n[cyan]Processing parquet file...[/cyan]")
    results_df = process_parquet_file(
        parquet, reference_genome, 
        batch_size=batch_size, 
        num_workers=num_workers,
        n_bp_downstream=n_bp_downstream,
        n_bp_upstream=n_bp_upstream
    )
    
    # Write detailed results to CSV
    console.print("\n[cyan]Writing detailed results...[/cyan]")
    csv_output = f"{output}_cpg_prob.csv"
    results_df.to_csv(csv_output, index=False)
    console.print(f"[green]Detailed results written to: {csv_output}[/green]")
    
    # Calculate weighted methylation rates
    console.print("\n[cyan]Calculating weighted methylation rates...[/cyan]")
    weighted_meth_df = calculate_weighted_methylation(results_df)
    
    # Write weighted methylation rates to bedGraph file
    console.print("[cyan]Writing bedGraph output...[/cyan]")
    bedgraph_output = f"{output}_CpG.bedGraph"
    if len(weighted_meth_df) > 0:
        weighted_meth_df.to_csv(
            bedgraph_output, 
            columns=['chr', 'start', 'end', 'rate'],
            sep='\t',
            header=False,
            index=False
        )
        console.print(f"[green]Weighted methylation rates written to: {bedgraph_output}[/green]")
    else:
        console.print(f"[yellow]No methylation data to write to bedGraph file[/yellow]")
    
    console.print("\n[bold green]Analysis completed successfully![/bold green]")

if __name__ == '__main__':
    main()
