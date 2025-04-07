process STAT_DEPTH {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(parquet)
    path fasta
    path fai
    path script

    output:
    tuple val(meta), path("*.csv"), emit: depth_stats

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    def num_workers = task.cpus > 1 ? task.cpus - 1 : 1
    """
    python3 ${script} \\
        ${args} \\
        --parquet ${parquet} \\
        --fasta ${fasta} \\
        --output ${prefix} \\
        --num-workers ${num_workers}
    """
}
