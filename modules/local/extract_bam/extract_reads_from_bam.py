import pandas as pd
import pysam
import click

def extract_reads_from_bam(bam, txt, threshold, output):
    # Get read names set
    txt_df = pd.read_csv(txt, sep='\t')
    
    for label in ['0', '1']:
        txt_df_filtered = txt_df[txt_df[f'prob_class_{label}'] >= threshold] # Filter predicted reads
        read_names = set(txt_df_filtered['name'].drop_duplicates())
        
        # Extract reads
        bam_file = pysam.AlignmentFile(bam, "rb")
        extracted_reads = [read for read in bam_file.fetch() if read.query_name in read_names]
        
        # Sort the extracted reads
        extracted_reads.sort(key=lambda x: (x.reference_name, x.reference_start))

        # Output a bam
        out_name = f'{output}_target.bam' if label == '1' else f'{output}_background.bam'
        output = pysam.AlignmentFile(out_name, "wb", template=bam_file)
        for read in extracted_reads:
            output.write(read)

        bam_file.close()
        output.close()

@click.command()
@click.option('--bam', required=True, type=click.Path(exists=True), help='Path to the bam file')
@click.option('--txt', required=True, type=click.Path(exists=True), help='Path to the txt file')
@click.option('--threshold', required=True, type=click.FLOAT, help='Threshold for probability')
@click.option('--output', required=True, type=click.STRING, help='Output file prefix')
def main(bam, txt, threshold, output):
    extract_reads_from_bam(bam, txt, threshold, output)

if __name__ == '__main__':
    main()
    