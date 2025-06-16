#!/bin/bash
#SBATCH --job-name="scratch_smooth_3"
#SBATCH --open-mode=append
#SBATCH --output=/scratch/ddr8143/logs/slurm_logs/%x_s%a_%A.out
#SBATCH --error=/scratch/ddr8143/logs/slurm_logs/%x_s%a_%A.err
#SBATCH --export=ALL
#SBATCH --time=47:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --account=cds

singularity exec --nv --overlay $SCRATCH/continual_learning_v2.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
source /scratch/ddr8143/repos/dr_gen/.venv/bin/activate
CUBLAS_WORKSPACE_CONFIG=:4096:8 python /scratch/ddr8143/repos/dr_gen/scripts/parallel_runs.py -p 6 --start_seed 0 --max_seed 3 --val_bs 500 --proj_name cifar10 --bs 500 --lr 1e-4 --wd 0.01,0.001,0.0001 --wtype scratch --wn None --ws torchvision --xft pycil --use_percent 1.0 --epochs 50 
"
