#!/usr/bin/bash
#SBATCH -J generate
#SBATCH --partition=cn-long
#SBATCH -o generate.log
#SBATCH -e generate.log
#SBATCH --cpus-per-task=32

singularity exec -B /lustre1,/lustre2,/appsnew /lustre1/cqyi/syfan/images/common_tools.sif \
python3 generate_chr_level_methylation_matrix.py \
--meta-file samplesheet.csv \
--insert-size-cutoff 10000 \
--output-prefix $1 \
--ncpus 32