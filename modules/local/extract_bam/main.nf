process EXTRACT_BAM {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(bam), path(txt)
    val(threshold)

    output:
    tuple val(meta), path("*_0.bam"), emit: background_bam
    tuple val(meta), path("*_1.bam"), emit: target_bam

    script:
    def args = task.ext.args ?: ''
    def script = "${projectDir}/modules/local/extract_bam/extract_reads_from_bam.py"
    def prefix = task.ext.prefix ?: "$meta.id"
    """
    python3 ${script} --bam ${bam} --txt ${txt} --threshold ${threshold} --output ${prefix}
    """
}
