#!/bin/sh
#
#
#SBATCH --account=nklab
#SBATCH --job-name=make_model_sent    # The job name.
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --error="slurm-%A_%a.err"
#SBATCH --time=4-0:0

module load anaconda3-2019.03 

cd /scratch/nklab/projects/contstimlang/contstim

. activate contstimlang

python -u make_model_sentences.py
echo "python script terminated"

# End of script
