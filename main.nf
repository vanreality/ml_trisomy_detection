include { EXTRACT_BAM } from './modules/local/extract_bam/main'
include { SAMTOOLS_FAIDX } from './modules/nf-core/samtools/faidx/main'
include { SAMTOOLS_INDEX } from './modules/nf-core/samtools/index/main'
include { SAMTOOLS_INDEX as SAMTOOLS_INDEX_RAW } from './modules/nf-core/samtools/index/main'
include { SAMTOOLS_INDEX as SAMTOOLS_INDEX_TARGET } from './modules/nf-core/samtools/index/main'
include { METHYLDACKEL_EXTRACT as METHYLDACKEL_EXTRACT_RAW } from './modules/nf-core/methyldackel/extract/main'
include { METHYLDACKEL_EXTRACT as METHYLDACKEL_EXTRACT_TARGET } from './modules/nf-core/methyldackel/extract/main'

workflow {
    Channel
        .fromPath(params.input)
        .splitCsv(header: true)
        .map { row -> 
            def meta = [id: row.sample]
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

    // TODO: use customed script to extract methylation rate using proba as attention score

    // TODO: merge methylation output to a matrix
}
