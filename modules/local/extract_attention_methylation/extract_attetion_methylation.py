import pandas as pd
import pysam
import click
from Bio import SeqIO

def get_cpg_sites(bed, fasta):
    """Extracts CpG sites from the DMR regions in the reference genome."""
    cpg_sites = []
    fasta_sequences = SeqIO.to_dict(SeqIO.parse(fasta, "fasta"))
    
    bed_df = pd.read_csv(bed, sep='\t', header=None, names=['chr', 'start', 'end'])
    for _, row in bed_df.iterrows():
        chr_seq = str(fasta_sequences[row['chr']].seq)
        for i in range(row['start'], row['end'] - 1):  # -1 to ensure CG pair
            if chr_seq[i:i+2] == "CG":
                cpg_sites.append([row['chr'], i, i + 2])
    
    return pd.DataFrame(cpg_sites, columns=['chr', 'start', 'end'])

def get_methylation_status(bam, txt, cpg_site_df):
    """Extracts read methylation status based on CIGAR and reference genome."""
    txt_df = pd.read_csv(txt, sep='\t')
    read_names = set(txt_df['name'].drop_duplicates())
    bam_file = pysam.AlignmentFile(bam, "rb")
    
    cpg_prob_list = []
    for read in bam_file.fetch():
        read_name = read.query_name.split('/')[0]
        if read_name not in read_names:
            continue
        
        seq = read.query_sequence
        chrom = read.reference_name
        start = read.reference_start
        prob0, prob1 = txt_df.loc[txt_df['name'] == read_name, ['prob_class_0', 'prob_class_1']].values[0]
        
        for _, site in cpg_site_df[cpg_site_df['chr'] == chrom].iterrows():
            if start <= site['start'] < read.reference_end:
                rel_pos = site['start'] - start
                if 0 <= rel_pos < len(seq):
                    status = 1 if seq[rel_pos] == 'M' else 0
                    cpg_prob_list.append([chrom, site['start'], site['end'], read_name, status, prob0, prob1])
    
    bam_file.close()
    return pd.DataFrame(cpg_prob_list, columns=['chr', 'start', 'end', 'name', 'status', 'prob_class_0', 'prob_class_1'])

def compute_methylation_rate(cpg_site_df, cpg_site_prob_df):
    """Calculates methylation rate for each CpG site."""
    rates = cpg_site_prob_df.groupby(['chr', 'start', 'end']).apply(
        lambda x: (x['prob_class_1'] * x['status']).sum() / x['prob_class_1'].sum() * 100 if x['prob_class_1'].sum() > 0 else 0
    ).reset_index(name='methylation_rate')
    
    return cpg_site_df.merge(rates, on=['chr', 'start', 'end'], how='left').fillna({'methylation_rate': 0})

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
    