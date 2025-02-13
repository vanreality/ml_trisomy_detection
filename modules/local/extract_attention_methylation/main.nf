process EXTRACT_ATTENTION_METHYLATION {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(bam), path(txt), path(bai)
    path bed
    path fasta
    path fai

    output:
    tuple val(meta), path("*_CpG.bedGraph"), emit: cpg_sites
    tuple val(meta), path("*_cpg_prob.csv"), emit: cpg_prob

    script:
    def args = task.ext.args ?: ''
    def script = "${workflow.projectDir}/bin/extract_attetion_methylation.py"
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    python3 ${script} \\
        ${args} \\
        --bam ${bam} \\
        --txt ${txt} \\
        --bed ${bed} \\
        --fasta ${fasta} \\
        --output ${prefix} \\
        --ncpus ${task.cpus}
    """
}
