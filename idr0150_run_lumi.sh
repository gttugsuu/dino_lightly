#!/bin/bash
#SBATCH --job-name=DINO_idr0150_rcrop   # Job name
#SBATCH --output=lumi_logs/%jo_DINO_idr0150.log
#SBATCH --error=lumi_logs/%je_DINO_idr0150.log
#SBATCH --account=project_462000147
#SBATCH --partition=dev-g # dev-g or small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=3:00:00

module use /appl/local/csc/modulefiles
module load pytorch/2.0

srun python3 main.py --n_nodes 1 --n_devices 8 \
    --wandb_project_name DINO_Lighlty \
    --wandb_name idr0150_largehead_rcrop \
    --dataset idr0150 \
    --data_path ... \
    --output_dir ... \
    --batch_size 125 \
    --max_epochs 500 \
    --seed 9 \
    --lr 0.001 \
    --momentum 0.996