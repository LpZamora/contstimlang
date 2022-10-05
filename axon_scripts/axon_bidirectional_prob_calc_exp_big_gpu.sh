#!/bin/sh
#
#
#SBATCH --account=nklab
#SBATCH --job-name=exp_a40    # The job name.
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16G
#SBATCH --error="slurm-%A_%a.err"
#SBATCH --time=24:00:00
#SBATCH --exclude=ax13
module load anaconda3-2019.03

cd /scratch/nklab/projects/contstimlang/contstimlang

conda activate contstimlang

python -u batch_bidirectional_prob_calc_exp.py
echo "python script terminated"

# End of script
