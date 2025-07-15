process CALCULATE_PROB_WEIGHTED_METHYLATION_FROM_PARQUET {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(parquet)
    path fasta
    path fai
    path script

    output:
    tuple val(meta), path("*_CpG.bedGraph"), emit: cpg_sites
    tuple val(meta), path("*_cpg_prob.csv"), emit: cpg_prob
    tuple val(meta), path("${meta.id}.log"), emit: log_file

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    python3 ${script} \\
        ${args} \\
        --parquet ${parquet} \\
        --fasta ${fasta} \\
        --output ${prefix} \\
        --num-workers ${task.cpus} \\
        > ${meta.id}.log 2>&1
    """
}
