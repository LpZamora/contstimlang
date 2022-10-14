#!/bin/sh
#
#
#SBATCH --account=nklab
#SBATCH --job-name=exp_2gpu    # The job name.
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --error="slurm-%A_%a.err"
#SBATCH --time=24:00:00
#SBATCH --exclude=ax11,ax12,ax13,ax14,ax15,ax16
module load anaconda3-2019.03

cd /scratch/nklab/projects/contstimlang/contstimlang

conda activate contstimlang

python -u batch_bidirectional_prob_calc_exp2.py
echo "python script terminated"

# End of script
