#!/bin/sh
#
#
#SBATCH --account=nklab
#SBATCH --job-name=make_model_sent    # The job name.
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --error="slurm-%A_%a.err"
#SBATCH --time=24:00:00
#SBATCH --output="slurm-%A_%a.out"
#SBATCH --exclude=ax01

module load anaconda3-2019.03

cd /scratch/nklab/projects/contstimlang/contstim

conda activate contstimlang

python -u batch_synthesize_absolute_rating_controversial_sentences.py
echo "python script terminated"

# End of script
