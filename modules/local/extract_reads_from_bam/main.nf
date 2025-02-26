process EXTRACT_READS_FROM_BAM {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(bam), path(txt), path(bai)
    val(threshold)
    path(script)

    output:
    tuple val(meta), path("*_target.bam"), emit: target_bam
    tuple val(meta), path("*_background.bam"), emit: background_bam

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "$meta.id"
    """
    python3 ${script} ${args} --bam ${bam} --txt ${txt} --threshold ${threshold} --output ${prefix}
    """
}
