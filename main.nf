include { EXTRACT_BAM } from './modules/local/extract_bam/main'
include { SAMTOOLS_FAIDX } from './modules/nf-core/samtools/faidx/main'
include { SAMTOOLS_INDEX } from './modules/nf-core/samtools/index/main'
include { SAMTOOLS_INDEX as SAMTOOLS_INDEX_RAW } from './modules/nf-core/samtools/index/main'
include { SAMTOOLS_INDEX as SAMTOOLS_INDEX_TARGET } from './modules/nf-core/samtools/index/main'
include { METHYLDACKEL_EXTRACT as METHYLDACKEL_EXTRACT_RAW } from './modules/nf-core/methyldackel/extract/main'
include { METHYLDACKEL_EXTRACT as METHYLDACKEL_EXTRACT_TARGET } from './modules/nf-core/methyldackel/extract/main'
include { EXTRACT_ATTENTION_METHYLATION } from './modules/local/extract_attention_methylation/main'
include { GENERATE_METHYLATION_MATRIX as GENERATE_METHYLATION_MATRIX_TARGET } from './modules/local/generate_methylation_matrix/main'
include { GENERATE_METHYLATION_MATRIX as GENERATE_METHYLATION_MATRIX_RAW } from './modules/local/generate_methylation_matrix/main'
include { GENERATE_METHYLATION_MATRIX as GENERATE_METHYLATION_MATRIX_ATTENTION } from './modules/local/generate_methylation_matrix/main'
include { PLOT_HEATMAP as PLOT_HEATMAP_TARGET } from './modules/local/plot_heatmap/main'
include { PLOT_HEATMAP as PLOT_HEATMAP_RAW } from './modules/local/plot_heatmap/main'
include { PLOT_HEATMAP as PLOT_HEATMAP_ATTENTION } from './modules/local/plot_heatmap/main'

workflow {
    Channel
        .fromPath(params.input)
        .splitCsv(header: true)
        .map { row -> 
            def meta = [id: row.sample, label: row.label]
            return [meta, file(row.bam), file(row.txt)]
        }
        .set { ch_samplesheet }

    SAMTOOLS_INDEX(ch_samplesheet.map {meta, bam, txt -> tuple(meta, bam)})
    ch_raw_samplesheet = ch_samplesheet.join(SAMTOOLS_INDEX.out.bai)

    EXTRACT_BAM(ch_raw_samplesheet, params.threshold)
    ch_target_bam = EXTRACT_BAM.out.target_bam

    SAMTOOLS_FAIDX([[:], file(params.reference)], [[:], []])
    ch_fasta_index = SAMTOOLS_FAIDX.out.fai

    SAMTOOLS_INDEX_TARGET(ch_target_bam)
    ch_target_bam_bai = SAMTOOLS_INDEX_TARGET.out.bai
    ch_target_bam = ch_target_bam.join(ch_target_bam_bai)

    METHYLDACKEL_EXTRACT_TARGET(
        ch_target_bam, 
        file(params.reference), 
        ch_fasta_index.map {meta, fasta_index -> fasta_index}
    )

    ch_raw_bam = ch_samplesheet.map {meta, bam, txt -> tuple(meta, bam)}
    SAMTOOLS_INDEX_RAW(ch_raw_bam)
    ch_raw_bam_bai = SAMTOOLS_INDEX_RAW.out.bai
    ch_raw_bam = ch_raw_bam.join(ch_raw_bam_bai)

    METHYLDACKEL_EXTRACT_RAW(
        ch_raw_bam, 
        file(params.reference), 
        ch_fasta_index.map {meta, fasta_index -> fasta_index}
    )

    EXTRACT_ATTENTION_METHYLATION(
        ch_raw_samplesheet,
        file(params.dmr_bed),
        file(params.reference),
        ch_fasta_index.map {meta, fasta_index -> fasta_index}
    )

    // Collect bedGraph files and create meta CSV for each type
    METHYLDACKEL_EXTRACT_TARGET.out.bedgraph
        .map { meta, bedgraph -> 
            [meta.id, meta.label, bedgraph.toString()].join(',')
        }
        .collectFile(
            name: 'target_meta.csv',
            newLine: true,
            seed: 'sample,label,bedgraph_file_path'
        )
        .map { csv -> [[ id: 'target' ], csv] }
        .set { ch_target_meta }

    METHYLDACKEL_EXTRACT_RAW.out.bedgraph
        .map { meta, bedgraph -> 
            [meta.id, meta.label, bedgraph.toString()].join(',')
        }
        .collectFile(
            name: 'raw_meta.csv',
            newLine: true,
            seed: 'sample,label,bedgraph_file_path'
        )
        .map { csv -> [[ id: 'raw' ], csv] }
        .set { ch_raw_meta }

    EXTRACT_ATTENTION_METHYLATION.out.cpg_sites
        .map { meta, bedgraph -> 
            [meta.id, meta.label, bedgraph.toString()].join(',')
        }
        .collectFile(
            name: 'attention_meta.csv',
            newLine: true,
            seed: 'sample,label,bedgraph_file_path'
        )
        .map { csv -> [[ id: 'attention' ], csv] }
        .set { ch_attention_meta }

    // Generate methylation matrices
    GENERATE_METHYLATION_MATRIX_TARGET(
        ch_target_meta,
        file(params.dmr_bed)
    )
    GENERATE_METHYLATION_MATRIX_RAW(
        ch_raw_meta,
        file(params.dmr_bed)
    )
    GENERATE_METHYLATION_MATRIX_ATTENTION(
        ch_attention_meta,
        file(params.dmr_bed)
    )

    // Plot heatmaps for each matrix type
    PLOT_HEATMAP_TARGET(
        GENERATE_METHYLATION_MATRIX_TARGET.out.cpgs_matrix,
        params.chr,
        params.format
    )
    PLOT_HEATMAP_RAW(
        GENERATE_METHYLATION_MATRIX_RAW.out.cpgs_matrix,
        params.chr,
        params.format
    )
    PLOT_HEATMAP_ATTENTION(
        GENERATE_METHYLATION_MATRIX_ATTENTION.out.cpgs_matrix,
        params.chr,
        params.format
    )
}
