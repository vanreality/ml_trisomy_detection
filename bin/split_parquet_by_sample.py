import os
import pandas as pd
import click
from pathlib import Path


@click.command()
@click.option('--parquet-file-path', type=click.Path(exists=True))
@click.option('--output-dir', type=click.Path(), default='.')
def split_parquet_by_sample(parquet_file_path, output_dir):
    """
    Read a parquet file and split it by sample.
    
    For each sample:
    1. Group by seqname column
    2. Calculate the average value of prob_class_1 column for each group
    3. Replace prob_class_1 column value of each group by the average value
    4. Output to {output_dir}/{sample}_{label}.parquet
    
    Also creates a samplesheet.csv with columns: sample, label, parquet_file_path
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read parquet file
    print(f"Reading parquet file: {parquet_file_path}")
    df = pd.read_parquet(parquet_file_path)
    
    # Verify required columns exist
    required_columns = ['sample', 'seqname', 'prob_class_1', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Group by sample
    samples = df['sample'].unique()
    print(f"Found {len(samples)} unique samples")
    
    # Create a list to store sample information for samplesheet
    samplesheet_data = []
    
    for sample in samples:
        print(f"Processing sample: {sample}")
        
        # Filter data for this sample
        sample_df = df[df['sample'] == sample].copy()
        
        # Group by seqname and calculate average prob_class_1
        seqname_groups = sample_df.groupby('seqname')
        avg_probs = seqname_groups['prob_class_1'].mean()
        
        # Replace prob_class_1 values with the average for each seqname group
        for seqname, avg_prob in avg_probs.items():
            sample_df.loc[sample_df['seqname'] == seqname, 'prob_class_1'] = avg_prob
        
        # Determine output filename
        label = sample_df['label'].iloc[0]
        output_file = os.path.join(output_dir, f"{sample}_{label}.parquet")
        
        # Get absolute path for samplesheet
        absolute_path = os.path.abspath(output_file)
        
        # Add to samplesheet data
        samplesheet_data.append({
            'sample': sample,
            'label': label,
            'parquet': absolute_path
        })
        
        # Write to parquet
        print(f"Writing to: {output_file}")
        sample_df.to_parquet(output_file, index=False)
    
    # Create and write the samplesheet
    samplesheet_path = os.path.join(output_dir, 'samplesheet.csv')
    samplesheet_df = pd.DataFrame(samplesheet_data)
    samplesheet_df.to_csv(samplesheet_path, index=False)
    print(f"Samplesheet created at: {samplesheet_path}")
    
    print("Processing complete")


if __name__ == "__main__":
    split_parquet_by_sample()
