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
include { GENERATE_METHYLATION_MATRIX as GENERATE_METHYLATION_MATRIX_PROB_WEIGHTED } from './modules/local/generate_methylation_matrix/main'
include { STAT_DEPTH } from './modules/local/stat_depth/main'
include { GENERATE_DEPTH_MATRIX } from './modules/local/generate_depth_matrix/main'
include { GENERATE_CHR_LEVEL_METHYLATION_MATRIX } from './modules/local/generate_chr_level_methylation_matrix/main'

// Add aliases for DMR heatmap plotting
include { PLOT_HEATMAP as PLOT_HEATMAP_TARGET_CPGS } from './modules/local/plot_heatmap/main'
include { PLOT_HEATMAP as PLOT_HEATMAP_RAW_CPGS } from './modules/local/plot_heatmap/main'
include { PLOT_HEATMAP as PLOT_HEATMAP_PROB_WEIGHTED_CPGS } from './modules/local/plot_heatmap/main'
include { PLOT_HEATMAP as PLOT_HEATMAP_TARGET_DMRS } from './modules/local/plot_heatmap/main'
include { PLOT_HEATMAP as PLOT_HEATMAP_RAW_DMRS } from './modules/local/plot_heatmap/main'
include { PLOT_HEATMAP as PLOT_HEATMAP_PROB_WEIGHTED_DMRS } from './modules/local/plot_heatmap/main'

workflow {
    // Parse mode parameter to determine which parts to run
    def run_raw = false
    def run_target = false
    def run_prob_weighted = false
    def run_depth = false
    def run_chr_level = false

    if (!params.mode) {
        // Default: run all modes if not specified
        run_raw = true
        run_target = params.threshold != null
        run_prob_weighted = true
        run_depth = true
        run_chr_level = true
    } else {
        // Parse the mode parameter
        def modes = params.mode.split('\\|')
        run_raw = modes.contains('raw')
        run_target = modes.contains('target') && params.threshold != null
        run_prob_weighted = modes.contains('prob_weighted')
        run_depth = modes.contains('depth')
        run_chr_level = modes.contains('chr_level')
    }
    
    // 1. Input processing and common setup
    // ====================================
    
    // Reference genome preparation - common for all modes
    SAMTOOLS_FAIDX([[:], file(params.reference)], [[:], []])
    ch_fasta_index = SAMTOOLS_FAIDX.out.fai
    
    // Determine input source (samplesheet or parquet)
    if (params.input_parquet_samplesheet) {
        // Skip SPLIT_PARQUET_BY_SAMPLE process and directly parse the provided CSV
        Channel
            .fromPath(params.input_parquet_samplesheet)
            .splitCsv(header: true)
            .map { row -> 
                def meta = [id: row.sample, label: row.label]
                // Check if parquet file exists
                def parquetFile = file(row.parquet)
                if (!parquetFile.exists()) {
                    error "Parquet file not found: ${row.parquet}"
                }
                return [meta, parquetFile]
            }
            .set { ch_parquet_samplesheet }
    } else if (params.input_parquet) {
        // Run split_parquet_by_sample process to split input parquet file by sample
        SPLIT_PARQUET_BY_SAMPLE(
            [[id: "sample"], file(params.input_parquet)],
            file("${workflow.projectDir}/bin/split_parquet_by_sample.py")
        )

        // Parse the generated samplesheet CSV to create channel
        SPLIT_PARQUET_BY_SAMPLE.out.samplesheet
            .map { meta, samplesheet -> samplesheet }
            .splitCsv(header: true)
            .map { row -> 
                def meta = [id: row.sample, label: row.label]
                // Check if parquet file exists
                def parquetFile = file(row.parquet)
                if (!parquetFile.exists()) {
                    error "Parquet file not found: ${row.parquet}"
                }
                return [meta, parquetFile]
            }
            .set { ch_parquet_samplesheet }
    }
    
    // Index input BAM files - common for raw and target processing
    if (run_raw || run_target) {
        // Read and parse input CSV file 
        Channel
            .fromPath(params.input_samplesheet)
            .splitCsv(header: true)
            .set { input_channel}
        
        if (run_raw) {
            // For raw processing, we only need the BAM files
            input_channel
                .map { row -> 
                    def meta = [id: row.sample, label: row.label]
                    return [meta, file(row.bam)]
                }
                .set { ch_raw_bam_input }
                
            SAMTOOLS_INDEX_RAW(ch_raw_bam_input)
            ch_raw_bam = ch_raw_bam_input.join(SAMTOOLS_INDEX_RAW.out.bai)
        }
        
        if (run_target) {
            // For target processing, we need both BAM and TXT files
            input_channel
                .map { row -> 
                    def meta = [id: row.sample, label: row.label]
                    return [meta, file(row.bam), file(row.txt)]
                }
                .set { ch_target_input }
                
            SAMTOOLS_INDEX(ch_target_input.map {meta, bam, txt -> tuple(meta, bam)})
            ch_target_samplesheet = ch_target_input.join(SAMTOOLS_INDEX.out.bai)
        }
    }
    
    // 2. RAW processing part
    // =====================
    if (run_raw) {
        // ch_raw_bam is already prepared with necessary BAM and BAI files
        
        METHYLDACKEL_EXTRACT_RAW(
            ch_raw_bam, 
            file(params.reference), 
            ch_fasta_index.map {meta, fasta_index -> fasta_index}
        )
        
        // Create meta CSV files for raw analysis
        ch_raw_meta = create_meta_csv(METHYLDACKEL_EXTRACT_RAW.out.bedgraph, 'raw')
        
        // Generate methylation matrices for raw analysis
        GENERATE_METHYLATION_MATRIX_RAW(ch_raw_meta, file(params.dmr_bed))
        
        // Generate heatmaps for CpG matrices
        PLOT_HEATMAP_RAW_CPGS(
            GENERATE_METHYLATION_MATRIX_RAW.out.cpgs_matrix,
            params.chr,
            params.format
        )
        
        // Generate heatmaps for DMR matrices
        PLOT_HEATMAP_RAW_DMRS(
            GENERATE_METHYLATION_MATRIX_RAW.out.dmrs_matrix,
            params.chr,
            params.format
        )
    }
    
    // 3. TARGET processing part
    // =======================
    if (run_target) {
        // Extract BAM regions based on threshold
        EXTRACT_READS_FROM_BAM(
            ch_target_samplesheet, 
            params.threshold, 
            file("${workflow.projectDir}/bin/extract_reads_from_bam.py")
        )
        ch_target_bam = EXTRACT_READS_FROM_BAM.out.target_bam
        
        // Process extracted target regions
        SAMTOOLS_INDEX_TARGET(ch_target_bam)
        ch_target_bam_bai = SAMTOOLS_INDEX_TARGET.out.bai
        ch_target_bam = ch_target_bam.join(ch_target_bam_bai)

        METHYLDACKEL_EXTRACT_TARGET(
            ch_target_bam, 
            file(params.reference), 
            ch_fasta_index.map { meta, fasta_index -> fasta_index }
        )
        
        // Create meta CSV files for target analysis
        ch_target_meta = create_meta_csv(METHYLDACKEL_EXTRACT_TARGET.out.bedgraph, 'target')
        
        // Generate methylation matrices for target analysis
        GENERATE_METHYLATION_MATRIX_TARGET(ch_target_meta, file(params.dmr_bed))
        
        // Generate heatmaps for target analysis
        PLOT_HEATMAP_TARGET_CPGS(
            GENERATE_METHYLATION_MATRIX_TARGET.out.cpgs_matrix,
            params.chr,
            params.format
        )
        
        PLOT_HEATMAP_TARGET_DMRS(
            GENERATE_METHYLATION_MATRIX_TARGET.out.dmrs_matrix,
            params.chr,
            params.format
        )
    }
    
    // 4. PROB_WEIGHTED processing part
    // ==============================
    if (run_prob_weighted) {
        if (params.input_parquet || params.input_parquet_samplesheet) {
            // Use parquet-based methylation calculation if parquet input is provided
            CALCULATE_PROB_WEIGHTED_METHYLATION_FROM_PARQUET(
                ch_parquet_samplesheet,
                file(params.reference),
                ch_fasta_index.map {meta, fasta_index -> fasta_index},
                file("${workflow.projectDir}/bin/calculate_prob_weighted_methylation_from_parquet.py")
            )
            
            // Create meta CSV for prob-weighted analysis with parquet input
            ch_prob_weighted_meta = create_prob_weighted_meta_csv(
                CALCULATE_PROB_WEIGHTED_METHYLATION_FROM_PARQUET.out.cpg_sites.join(
                    CALCULATE_PROB_WEIGHTED_METHYLATION_FROM_PARQUET.out.cpg_prob
                )
            )
        } else {
            // For bed-based calculations, either raw or target mode must be active
            if (!run_raw && !run_target) {
                error "Cannot run probability weighted methylation calculation with bedGraph input. Either 'raw' or 'target' mode must be active."
            }
            
            // Use original BED-based methylation calculation
            if (run_target) {
                CALCULATE_PROB_WEIGHTED_METHYLATION_FROM_BED(
                    ch_target_samplesheet,
                    file(params.dmr_bed),
                    file(params.reference),
                    ch_fasta_index.map {meta, fasta_index -> fasta_index},
                    file("${workflow.projectDir}/bin/calculate_prob_weighted_methylation_from_bed.py")
                )
            } else if (run_raw) {
                // For raw mode, we need to ensure the channel has the right format (meta, bam, txt, bai)
                // We need to read the input sample sheet again to get the txt files
                Channel
                    .fromPath(params.input_samplesheet)
                    .splitCsv(header: true)
                    .map { row -> 
                        def meta = [id: row.sample, label: row.label]
                        return [meta, file(row.bam), file(row.txt)]
                    }
                    .join(ch_raw_bam.map { meta, bam, bai -> tuple(meta, bam) })
                    .map { meta, bam, txt, bai -> tuple(meta, bam, txt, bai) }
                    .set { ch_prob_weighted_input }
                    
                CALCULATE_PROB_WEIGHTED_METHYLATION_FROM_BED(
                    ch_prob_weighted_input,
                    file(params.dmr_bed),
                    file(params.reference),
                    ch_fasta_index.map {meta, fasta_index -> fasta_index},
                    file("${workflow.projectDir}/bin/calculate_prob_weighted_methylation_from_bed.py")
                )
            } else {
                error "Neither 'raw' nor 'target' mode is active. Cannot run probability weighted methylation calculation."
            }
            
            // Create meta CSV for prob-weighted analysis with bedgraph input
            ch_prob_weighted_meta = create_prob_weighted_meta_csv(
                CALCULATE_PROB_WEIGHTED_METHYLATION_FROM_BED.out.cpg_sites.join(
                    CALCULATE_PROB_WEIGHTED_METHYLATION_FROM_BED.out.cpg_prob
                )
            )
        }
        
        // Generate methylation matrices for prob-weighted analysis
        GENERATE_METHYLATION_MATRIX_PROB_WEIGHTED(ch_prob_weighted_meta, file(params.dmr_bed))

        if (run_chr_level && params.hypo_dmr_bed != null && params.hyper_dmr_bed != null) {
            // Generate methylation matrix for chr level analysis
            GENERATE_CHR_LEVEL_METHYLATION_MATRIX(
                ch_prob_weighted_meta, 
                file(params.hypo_dmr_bed), 
                file(params.hyper_dmr_bed),
                file("${workflow.projectDir}/bin/generate_chr_level_methylation_matrix.py")
            )
        }
        
        // Generate heatmaps for prob-weighted analysis
        PLOT_HEATMAP_PROB_WEIGHTED_CPGS(
            GENERATE_METHYLATION_MATRIX_PROB_WEIGHTED.out.cpgs_matrix,
            params.chr,
            params.format
        )
        
        PLOT_HEATMAP_PROB_WEIGHTED_DMRS(
            GENERATE_METHYLATION_MATRIX_PROB_WEIGHTED.out.dmrs_matrix,
            params.chr,
            params.format
        )
    }

    // 5. DEPTH processing part
    // =======================
    if (run_depth) {
        STAT_DEPTH(
            ch_parquet_samplesheet,
            file(params.dmr_bed),
            file(params.vcf),
            file(params.reference),
            ch_fasta_index.map {meta, fasta_index -> fasta_index},
            file("${workflow.projectDir}/bin/stat_depth.py")
        )

        ch_depth_samplesheet = STAT_DEPTH.out.depth_stats
            .map { meta, depth_stats -> 
                def parquet_path = depth_stats.toAbsolutePath().toString()
                if (!file(parquet_path).exists()) {
                    error "Parquet file does not exist: ${parquet_path}"
                }
                [meta.id, meta.label, parquet_path].join(',')
            }
            .collectFile(
                name: "depth_meta.csv",
                newLine: true,
                seed: 'sample,label,parquet_file_path'
            )
            .map { csv -> [[ id: 'depth_samplesheet' ], csv] }

        GENERATE_DEPTH_MATRIX(
            ch_depth_samplesheet,
            file("${workflow.projectDir}/bin/generate_depth_matrix.py")
        )
    }
}

// Helper functions for creating meta CSV files
def create_meta_csv(stream, prefix) {
    return stream
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

def create_prob_weighted_meta_csv(stream) {
    return stream
        .map { meta, bedgraph, prob_file -> 
            [meta.id, meta.label, bedgraph.toString(), prob_file.toString()].join(',')
        }
        .collectFile(
            name: "prob_weighted_meta.csv",
            newLine: true,
            seed: 'sample,label,bedgraph_file_path,prob_file_path'
        )
        .map { csv -> [[ id: 'prob_weighted' ], csv] }
}
