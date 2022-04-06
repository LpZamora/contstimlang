#!/bin/sh
#
#
#SBATCH --account=nklab
#SBATCH --job-name=get_probs    # The job name.
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=72:00:00

module load anaconda3-2019.03

cd /scratch/nklab/projects/contstimlang/contstim

conda activate contstimlang

python -u get_reddit_sent_probs.py --model $MODEL
echo "python script terminated"

# End of script
