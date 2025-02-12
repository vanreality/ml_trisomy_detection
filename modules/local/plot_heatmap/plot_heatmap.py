#!/usr/bin/env python3

import click
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np
import re

@click.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True, help='Input TSV file path')
@click.option('--output', '-o', default='heatmap.pdf', help='Output file path')
@click.option('--format', '-f', default='pdf', help='Output format (pdf, png, jpg)')
@click.option('--chr', '-c', default='chr16', help='Chromosome to select for specific plot')
@click.option('--figsize', default=(10, 8), type=(float, float), help='Figure size (width, height)')
def plot_heatmap(input, output, format, chr, figsize):
    """
    Draw a heatmap from a TSV file containing methylation data.
    The TSV file should have columns: 'sample', 'label', and genomic regions.
    """
    # Read the TSV file
    df = pd.read_csv(input, sep='\t')
    
    # Extract sample and label columns
    samples = df['sample']
    labels = df['label']
    
    # Parse chromosome information from column names
    data_cols = df.columns[2:]  # Skip 'sample' and 'label'
    chr_info = {}
    for col in data_cols:
        match = re.match(r'(chr\w+):.*', col)
        if match:
            chr_info[col] = match.group(1)
    
    # Create chromosome categories
    chr_categories = pd.Series(chr_info)
    selected_chr_cols = chr_categories[chr_categories == chr].index
    other_cols = chr_categories[chr_categories != chr].index
    
    # Reorder columns: selected chr followed by others
    ordered_cols = list(selected_chr_cols) + list(other_cols)
    
    # Create column colors for chromosome information
    col_colors = pd.Series('other', index=ordered_cols)
    col_colors[selected_chr_cols] = chr
    col_palette = {chr: sns.color_palette('Set1')[0], 'other': sns.color_palette('Set1')[1]}
    col_colors = col_colors.map(col_palette)
    
    # Ensure output has the correct format
    if not output.endswith(f'.{format}'):
        output = f"{output.rsplit('.', 1)[0]}.{format}"
    
    def create_plot(data_matrix, output_path):
        # Create row colors for labels
        label_colors = pd.Series(df['label'].factorize()[0], index=df.index)
        label_palette = sns.color_palette('Set2', len(label_colors.unique()))
        row_colors = pd.DataFrame({'Label': label_colors.map(dict(enumerate(label_palette)))},
                                index=df.index)
        
        # Create the clustermap
        g = sns.clustermap(
            data_matrix,
            cmap=custom_cmap,
            vmin=0,
            vmax=100,
            row_cluster=True,
            col_cluster=False,
            yticklabels=df['sample'],
            xticklabels=False,
            figsize=figsize,
            row_colors=row_colors,
            col_colors=col_colors[data_matrix.columns],
        )
        
        # Create and add label legend
        legend_patches = [Patch(color=color, label=label) 
                        for color, label in zip(label_palette, label_order)]
        
        # Add chromosome legend
        chr_patches = [Patch(color=color, label=chr_type) 
                      for chr_type, color in col_palette.items()]
        legend_patches.extend(chr_patches)
        
        g.ax_heatmap.legend(handles=legend_patches, 
                           loc='center left', 
                           bbox_to_anchor=(1.2, 0.5), 
                           title='Labels & Chromosomes', 
                           frameon=False)
        
        # Hide axes ticks
        g.ax_heatmap.set_xticks([])
        g.ax_heatmap.tick_params(left=False, right=False, top=False, bottom=False)
        g.ax_row_colors.set_xticks([])
        g.ax_row_colors.set_yticks([])
        g.ax_col_colors.set_xticks([])
        g.ax_col_colors.set_yticks([])
        g.ax_heatmap.set_ylabel('')
        
        # Save the plot
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    # Define label order and create categorical labels
    label_order = sorted(labels.unique())
    labels = pd.Categorical(labels, categories=label_order, ordered=True)
    df['label'] = labels
    df.sort_values(by='label', inplace=True)
    
    # Get the data matrix (excluding sample and label columns)
    data_matrix = df.drop(['sample', 'label'], axis=1)
    data_matrix = data_matrix[ordered_cols]  # Reorder columns
    
    # Fill NA values with column medians
    data_matrix = data_matrix.fillna(data_matrix.median())
    
    # Create custom colormap from blue to yellow
    colors = ['#0000FF', '#FFD700']
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors)
    
    # Create full plot
    create_plot(data_matrix, output)
    
    # Create chromosome-specific plot
    if len(selected_chr_cols) > 0:
        chr_matrix = data_matrix[selected_chr_cols]
        chr_output = f"{output.rsplit('.', 1)[0]}_{chr}.{format}"
        create_plot(chr_matrix, chr_output)
    else:
        click.echo(f"Warning: No regions found for chromosome {chr}")

if __name__ == '__main__':
    plot_heatmap()
