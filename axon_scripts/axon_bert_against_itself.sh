#!/bin/sh
#
#
#SBATCH --account=nklab
#SBATCH --job-name=make_model_sent    # The job name.
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --error="slurm-%A_%a.err"
#SBATCH --time=24:00:00
#SBATCH --exclude=ax11,ax12,ax13
module load anaconda3-2019.03

cd /scratch/nklab/projects/contstimlang/contstimlang

conda activate contstimlang

python -u batch_bert_against_itself.py
echo "python script terminated"

# End of script
