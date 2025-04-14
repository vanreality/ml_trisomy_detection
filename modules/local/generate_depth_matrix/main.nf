process GENERATE_DEPTH_MATRIX {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(meta_csv)
    path script

    output:
    tuple val(meta), path("*.parquet"), emit: depth_matrices

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    python3 ${script} \\
        ${args} \\
        --meta ${meta_csv} \\
        --output ${prefix}
    """
}
