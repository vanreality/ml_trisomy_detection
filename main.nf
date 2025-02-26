// Import required modules with logical grouping
// Core samtools modules
include { SAMTOOLS_FAIDX } from './modules/nf-core/samtools/faidx/main'
include { SAMTOOLS_INDEX } from './modules/nf-core/samtools/index/main'
include { SAMTOOLS_INDEX as SAMTOOLS_INDEX_RAW } from './modules/nf-core/samtools/index/main'
include { SAMTOOLS_INDEX as SAMTOOLS_INDEX_TARGET } from './modules/nf-core/samtools/index/main'

// Methylation analysis modules
include { METHYLDACKEL_EXTRACT as METHYLDACKEL_EXTRACT_RAW } from './modules/nf-core/methyldackel/extract/main'
include { METHYLDACKEL_EXTRACT as METHYLDACKEL_EXTRACT_TARGET } from './modules/nf-core/methyldackel/extract/main'

// Custom analysis modules
include { SPLIT_PARQUET_BY_SAMPLE } from './modules/local/split_parquet_by_sample/main'
include { EXTRACT_READS_FROM_BAM } from './modules/local/extract_reads_from_bam/main'
include { CALCULATE_PROB_WEIGHTED_METHYLATION_FROM_BED } from './modules/local/calculate_prob_weighted_methylation_from_bed/main'
include { CALCULATE_PROB_WEIGHTED_METHYLATION_FROM_PARQUET } from './modules/local/calculate_prob_weighted_methylation_from_parquet/main'
include { GENERATE_METHYLATION_MATRIX as GENERATE_METHYLATION_MATRIX_TARGET } from './modules/local/generate_methylation_matrix/main'
include { GENERATE_METHYLATION_MATRIX as GENERATE_METHYLATION_MATRIX_RAW } from './modules/local/generate_methylation_matrix/main'
include { GENERATE_METHYLATION_MATRIX as GENERATE_METHYLATION_MATRIX_ATTENTION } from './modules/local/generate_methylation_matrix/main'

// Add aliases for DMR heatmap plotting
include { PLOT_HEATMAP as PLOT_HEATMAP_TARGET_CPGS } from './modules/local/plot_heatmap/main'
include { PLOT_HEATMAP as PLOT_HEATMAP_RAW_CPGS } from './modules/local/plot_heatmap/main'
include { PLOT_HEATMAP as PLOT_HEATMAP_ATTENTION_CPGS } from './modules/local/plot_heatmap/main'
include { PLOT_HEATMAP as PLOT_HEATMAP_TARGET_DMRS } from './modules/local/plot_heatmap/main'
include { PLOT_HEATMAP as PLOT_HEATMAP_RAW_DMRS } from './modules/local/plot_heatmap/main'
include { PLOT_HEATMAP as PLOT_HEATMAP_ATTENTION_DMRS } from './modules/local/plot_heatmap/main'

workflow {
    // 1. Input processing
    // Determine input source (samplesheet or parquet)
    if (params.input_parquet) {
        // Process parquet input and create samplesheet
        SPLIT_PARQUET_BY_SAMPLE(
            [[ id: 'samples' ], file(params.input_parquet)],
            file("${workflow.projectDir}/bin/split_parquet_by_sample.py")
        )
        
        // Read the generated samplesheet
        Channel
            .fromPath(SPLIT_PARQUET_BY_SAMPLE.out.samplesheet.map { meta, samplesheet -> samplesheet })
            .splitCsv(header: true)
            .map { row -> 
                def meta = [id: row.sample, label: row.label]
                return [meta, file(row.parquet)]
            }
            .set { ch_parquet_samplesheet }
    }
    
    // Read and parse input CSV file (original behavior)
    Channel
        .fromPath(params.input_samplesheet)
        .splitCsv(header: true)
        .map { row -> 
            def meta = [id: row.sample, label: row.label]
            return [meta, file(row.bam), file(row.txt)]
        }
        .set { ch_samplesheet }

    // 2. Initial BAM processing
    // Index input BAM files
    SAMTOOLS_INDEX(ch_samplesheet.map {meta, bam, txt -> tuple(meta, bam)})
    ch_raw_samplesheet = ch_samplesheet.join(SAMTOOLS_INDEX.out.bai)

    // If threshold is provided, extract target regions from the BAM files
    if ( params.threshold ) {
        // Extract BAM regions based on threshold
        EXTRACT_READS_FROM_BAM(
            ch_raw_samplesheet, 
            params.threshold, 
            file("${workflow.projectDir}/bin/extract_reads_from_bam.py")
        )
        ch_target_bam = EXTRACT_READS_FROM_BAM.out.target_bam
    }

    // 3. Reference genome preparation
    // Index reference genome
    SAMTOOLS_FAIDX([[:], file(params.reference)], [[:], []])
    ch_fasta_index = SAMTOOLS_FAIDX.out.fai

    // 4. Target BAM analysis
    // Process extracted target regions
    if ( params.threshold ) {
        // Process extracted target regions
        SAMTOOLS_INDEX_TARGET(ch_target_bam)
        ch_target_bam_bai = SAMTOOLS_INDEX_TARGET.out.bai
        ch_target_bam = ch_target_bam.join(ch_target_bam_bai)

        METHYLDACKEL_EXTRACT_TARGET(
            ch_target_bam, 
            file(params.reference), 
            ch_fasta_index.map { meta, fasta_index -> fasta_index }
        )
    }

    // 5. Raw BAM analysis
    // Process original BAM files
    ch_raw_bam = ch_samplesheet.map {meta, bam, txt -> tuple(meta, bam)}
    SAMTOOLS_INDEX_RAW(ch_raw_bam)
    ch_raw_bam_bai = SAMTOOLS_INDEX_RAW.out.bai
    ch_raw_bam = ch_raw_bam.join(ch_raw_bam_bai)

    METHYLDACKEL_EXTRACT_RAW(
        ch_raw_bam, 
        file(params.reference), 
        ch_fasta_index.map {meta, fasta_index -> fasta_index}
    )

    // 6. Attention methylation analysis
    if (params.input_parquet) {
        // Use parquet-based methylation calculation if parquet input is provided
        CALCULATE_PROB_WEIGHTED_METHYLATION_FROM_PARQUET(
            ch_parquet_samplesheet,
            file(params.reference),
            ch_fasta_index.map {meta, fasta_index -> fasta_index},
            file("${workflow.projectDir}/bin/calculate_prob_weighted_methylation_from_parquet.py")
        )
    } else {
        // Use original BED-based methylation calculation
        CALCULATE_PROB_WEIGHTED_METHYLATION_FROM_BED(
            ch_raw_samplesheet,
            file(params.dmr_bed),
            file(params.reference),
            ch_fasta_index.map {meta, fasta_index -> fasta_index},
            file("${workflow.projectDir}/bin/calculate_prob_weighted_methylation_from_bed.py")
        )
    }

    // 7. Metadata preparation
    // Create meta CSV files for each analysis type
    def create_meta_csv = { stream, prefix ->
        stream
            .map { meta, bedgraph -> 
                [meta.id, meta.label, bedgraph.toString()].join(',')
            }
            .collectFile(
                name: "${prefix}_meta.csv",
                newLine: true,
                seed: 'sample,label,bedgraph_file_path'
            )
            .map { csv -> [[ id: prefix ], csv] }
    }

    def create_attention_meta_csv = { stream ->
        stream
            .map { meta, bedgraph, prob_file -> 
                [meta.id, meta.label, bedgraph.toString(), prob_file.toString()].join(',')
            }
            .collectFile(
                name: "attention_meta.csv",
                newLine: true,
                seed: 'sample,label,bedgraph_file_path,prob_file_path'
            )
            .map { csv -> [[ id: 'attention' ], csv] }
    }

    ch_target_meta = params.threshold ? create_meta_csv(METHYLDACKEL_EXTRACT_TARGET.out.bedgraph, 'target') : Channel.empty()
    ch_raw_meta = create_meta_csv(METHYLDACKEL_EXTRACT_RAW.out.bedgraph, 'raw')
    
    // Use the appropriate channel for attention meta based on the input source
    ch_attention_meta = params.input_parquet ? 
        create_attention_meta_csv(
            CALCULATE_PROB_WEIGHTED_METHYLATION_FROM_PARQUET.out.cpg_sites.join(CALCULATE_PROB_WEIGHTED_METHYLATION_FROM_PARQUET.out.cpg_prob)
        ) :
        create_attention_meta_csv(
            CALCULATE_PROB_WEIGHTED_METHYLATION_FROM_BED.out.cpg_sites.join(CALCULATE_PROB_WEIGHTED_METHYLATION_FROM_BED.out.cpg_prob)
        )

    // 8. Matrix generation
    // Generate methylation matrices for each analysis type
    if ( params.threshold ) {
        GENERATE_METHYLATION_MATRIX_TARGET(ch_target_meta, file(params.dmr_bed))
    }
    GENERATE_METHYLATION_MATRIX_RAW(ch_raw_meta, file(params.dmr_bed))
    GENERATE_METHYLATION_MATRIX_ATTENTION(ch_attention_meta, file(params.dmr_bed))

    // 9. Visualization
    // Generate heatmaps for CpG matrices
    if ( params.threshold ) {
        // Generate heatmaps for CpG matrices for target analysis
        PLOT_HEATMAP_TARGET_CPGS(
            GENERATE_METHYLATION_MATRIX_TARGET.out.cpgs_matrix,
            params.chr,
            params.format
        )
    }
    PLOT_HEATMAP_RAW_CPGS(
        GENERATE_METHYLATION_MATRIX_RAW.out.cpgs_matrix,
        params.chr,
        params.format
    )
    PLOT_HEATMAP_ATTENTION_CPGS(
        GENERATE_METHYLATION_MATRIX_ATTENTION.out.cpgs_matrix,
        params.chr,
        params.format
    )

    // Generate heatmaps for DMR matrices
    if ( params.threshold ) {
        // Generate heatmaps for DMR matrices for target analysis
        PLOT_HEATMAP_TARGET_DMRS(
            GENERATE_METHYLATION_MATRIX_TARGET.out.dmrs_matrix,
            params.chr,
            params.format
        )
    }
    PLOT_HEATMAP_RAW_DMRS(
        GENERATE_METHYLATION_MATRIX_RAW.out.dmrs_matrix,
        params.chr,
        params.format
    )
    PLOT_HEATMAP_ATTENTION_DMRS(
        GENERATE_METHYLATION_MATRIX_ATTENTION.out.dmrs_matrix,
        params.chr,
        params.format
    )
}
