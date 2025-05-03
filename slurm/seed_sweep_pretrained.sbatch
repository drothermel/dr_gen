#!/bin/bash
#SBATCH --job-name="cifar10_p"
#SBATCH --array=1
#SBATCH --open-mode=append
#SBATCH --output=/scratch/ddr8143/logs/slurm_logs/%x_s%a_%A.out
#SBATCH --error=/scratch/ddr8143/logs/slurm_logs/%x_s%a_%A.err
#SBATCH --export=ALL
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=cds

singularity exec --nv --overlay $SCRATCH/continual_learning.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
source /scratch/ddr8143/repos/dr_gen/.venv/bin/activate
CUBLAS_WORKSPACE_CONFIG=:4096:8 python /scratch/ddr8143/repos/dr_gen/scripts/train.py paths.proj_dir_name=cifar10 seed=${SLURM_ARRAY_TASK_ID} optim.lr=${LR} optim.weight_decay=${WD} model.weights=DEFAULT weight_type=pretrained
"
