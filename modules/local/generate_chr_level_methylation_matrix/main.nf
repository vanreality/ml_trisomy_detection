process GENERATE_CHR_LEVEL_METHYLATION_MATRIX {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(meta_csv)
    path hypo_dmr_bed
    path hyper_dmr_bed
    path script

    output:
    tuple val(meta), path("*.tsv"), emit: chr_level_matrices

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    python3 ${script} \\
        ${args} \\
        --meta-file ${meta_csv} \\
        --hypo-dmr-bed ${hypo_dmr_bed} \\
        --hyper-dmr-bed ${hyper_dmr_bed} \\
        --centralize \\
        --ncpus ${task.cpus} \\
        --output-prefix ${prefix}
    """
}
