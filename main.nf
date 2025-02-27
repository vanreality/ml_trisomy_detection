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
    
    if (!params.mode) {
        // Default: run all modes if not specified
        run_raw = true
        run_target = params.threshold != null
        run_prob_weighted = true
    } else {
        // Parse the mode parameter
        def modes = params.mode.split('\\|')
        run_raw = modes.contains('raw')
        run_target = modes.contains('target') && params.threshold != null
        run_prob_weighted = modes.contains('prob_weighted')
    }
    
    // 1. Input processing and common setup
    // ====================================
    
    // Reference genome preparation - common for all modes
    SAMTOOLS_FAIDX([[:], file(params.reference)], [[:], []])
    ch_fasta_index = SAMTOOLS_FAIDX.out.fai
    
    // Determine input source (samplesheet or parquet)
    if (params.input_parquet) {
        // Process parquet input and create samplesheet
        SPLIT_PARQUET_BY_SAMPLE(
            [[ id: 'samples' ], file(params.input_parquet)],
            file("${workflow.projectDir}/bin/split_parquet_by_sample.py")
        )
        
        // Read the generated samplesheet
        SPLIT_PARQUET_BY_SAMPLE.out.samplesheet
            .map { meta, samplesheet -> samplesheet }
            .splitCsv(header: true)
            .map { row -> 
                def meta = [id: row.sample, label: row.label]
                // Ensure file path is correctly resolved with absolute path if needed
                def parquetFile = file(row.parquet, checkIfExists: true)
                return [meta, parquetFile]
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
    
    // Index input BAM files - common for raw and target processing
    if (run_raw || run_target) {
        SAMTOOLS_INDEX(ch_samplesheet.map {meta, bam, txt -> tuple(meta, bam)})
        ch_raw_samplesheet = ch_samplesheet.join(SAMTOOLS_INDEX.out.bai)
    }
    
    // 2. RAW processing part
    // =====================
    if (run_raw) {
        ch_raw_samplesheet
            .map { meta, bam, txt, bai -> tuple(meta, bam, bai) }
            .set { ch_raw_bam }

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
            ch_raw_samplesheet, 
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
        if (params.input_parquet) {
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
            // Use original BED-based methylation calculation
            CALCULATE_PROB_WEIGHTED_METHYLATION_FROM_BED(
                ch_raw_samplesheet,
                file(params.dmr_bed),
                file(params.reference),
                ch_fasta_index.map {meta, fasta_index -> fasta_index},
                file("${workflow.projectDir}/bin/calculate_prob_weighted_methylation_from_bed.py")
            )
            
            // Create meta CSV for prob-weighted analysis with bedgraph input
            ch_prob_weighted_meta = create_prob_weighted_meta_csv(
                CALCULATE_PROB_WEIGHTED_METHYLATION_FROM_BED.out.cpg_sites.join(
                    CALCULATE_PROB_WEIGHTED_METHYLATION_FROM_BED.out.cpg_prob
                )
            )
        }
        
        // Generate methylation matrices for prob-weighted analysis
        GENERATE_METHYLATION_MATRIX_PROB_WEIGHTED(ch_prob_weighted_meta, file(params.dmr_bed))
        
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
