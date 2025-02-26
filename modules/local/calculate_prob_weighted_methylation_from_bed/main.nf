process CALCULATE_PROB_WEIGHTED_METHYLATION_FROM_BED {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(bam), path(txt), path(bai)
    path bed
    path fasta
    path fai
    path script

    output:
    tuple val(meta), path("*_CpG.bedGraph"), emit: cpg_sites
    tuple val(meta), path("*_cpg_prob.csv"), emit: cpg_prob

    script:
    def args = task.ext.args ?: ''
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
