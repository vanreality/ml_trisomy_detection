import pandas as pd
import numpy as np
import pysam
import click
from Bio import SeqIO

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

def get_methylation_status(bam, txt, cpg_site_df):
    """Extracts read methylation status based on CIGAR and txt file."""
    txt_df = pd.read_csv(txt, sep='\t')
    # Add 'chr' prefix if not present
    if not txt_df['chr'].astype(str).str.contains('chr').all():
        txt_df['chr'] = 'chr' + txt_df['chr'].astype(str)
    
    read_names = set(txt_df['name'].drop_duplicates())
    bam_file = pysam.AlignmentFile(bam, "rb")
    
    cpg_prob_list = []
    for read in bam_file.fetch():
        read_name = read.query_name.split('/')[0]
        if read_name not in read_names:
            continue
        
        cigar_tuples = read.cigartuples
        
        # Find corresponding row in txt_df
        read_data = txt_df[
            (txt_df['name'] == read_name) & 
            (txt_df['chr'] == read.reference_name) & 
            (txt_df['start'] == read.reference_start)
        ]
        
        if read_data.empty:
            continue
            
        prob1 = read_data['prob_class_1'].values[0]
        sequence = read_data['text'].values[0]
        chrom = read_data['chr'].values[0]
        start = read_data['start'].values[0]
        end = read_data['end'].values[0]
        
        # Find CpG sites in the read mapping region
        cpg_sites = cpg_site_df[
            (cpg_site_df['chr'] == chrom) & 
            (cpg_site_df['start'] >= start) & 
            (cpg_site_df['end'] <= end + 1)
        ]
        
        # Convert CIGAR to reference positions
        ref_pos = start
        seq_pos = 0
        for op, length in cigar_tuples:
            # Match/mismatch or sequence match
            if op in [0, 7, 8]:
                for cpg in cpg_sites.itertuples():
                    if cpg.start >= ref_pos and cpg.start < ref_pos + length:
                        # Calculate position in sequence
                        seq_offset = cpg.start - ref_pos
                        seq_index = seq_pos + seq_offset
                        
                        # Check methylation status
                        if sequence[seq_index] == 'M':
                            status = 1
                        elif sequence[seq_index] == 'C':
                            status = 0
                        else:
                            status = np.nan
                            
                        cpg_prob_list.append([
                            cpg.chr,
                            cpg.start,
                            cpg.end,
                            read_name,
                            status,
                            prob1
                        ])
                
                ref_pos += length
                seq_pos += length
            # Deletion
            elif op == 2:
                ref_pos += length
            # Insertion or soft clip
            elif op in [1, 4]:
                seq_pos += length
   
    bam_file.close()
    return pd.DataFrame(cpg_prob_list, columns=['chr', 'start', 'end', 'name', 'status', 'prob_class_1']).dropna()

def compute_methylation_rate(cpg_site_df, cpg_site_prob_df):
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
def main(bam, txt, bed, fasta, output):
    cpg_site_df = get_cpg_sites(bed, fasta)
    cpg_site_prob_df = get_methylation_status(bam, txt, cpg_site_df)
    cpg_site_df = compute_methylation_rate(cpg_site_df, cpg_site_prob_df)
    
    cpg_site_df.to_csv(f"{output}_cpg_sites.csv", index=False)
    cpg_site_prob_df.to_csv(f"{output}_cpg_prob.csv", index=False)

if __name__ == '__main__':
    main()
    