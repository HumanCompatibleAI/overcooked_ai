#!/bin/bash
#SBATCH --qos=blanca-kann
#SBATCH --time=01:00:00
#SBATCH --gres=gpu
#SBATCH --ntasks=4
#SBATCH --mem=40G
#SBATCH --job-name=oai
#SBATCH --output=oai.%j.out
source /curc/sw/anaconda3/latest
conda activate arl
python agents/behavioral_cloning.py --base-dir /projects/star7023/oai --layout asymmetric_advantages --dataset 2019_hh_trials_all.pickle --encoding-fn dense_lossless
