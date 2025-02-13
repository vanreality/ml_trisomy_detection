process PLOT_HEATMAP {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(tsv)
    val(chr)
    val(format)

    output:
    tuple val(meta), path("*.${format}"), emit: plots
    
    script:
    def args = task.ext.args ?: ''
    def script = "${projectDir}/modules/local/plot_heatmap/plot_heatmap.py"
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    python3 ${script} \\
        --input ${tsv} \\
        --output ${prefix}.${format} \\
        --format ${format} \\
        --chr ${chr} \\
        ${args}
    """
}
