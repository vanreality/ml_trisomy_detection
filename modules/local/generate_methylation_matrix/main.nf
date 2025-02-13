process GENERATE_METHYLATION_MATRIX {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(meta_csv)
    path(dmr_bed)
    
    output:
    tuple val(meta), path("*_cpgs.tsv"), emit: cpgs_matrix
    tuple val(meta), path("*_dmrs.tsv"), emit: dmrs_matrix
    
    script:
    def args = task.ext.args ?: ''
    def script = "${workflow.projectDir}/bin/generate_methylation_matrix.py"
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    python3 ${script} \\
        --meta-file ${meta_csv} \\
        --dmr-file ${dmr_bed} \\
        --output-prefix ${prefix} \\
        ${args}
    """
}
