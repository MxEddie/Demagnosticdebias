#!/bin/bash
#SBATCH -A NLP-CDT-SL2-GPU
#SBATCH -p ampere
#SBATCH -N1
#SBATCH -n1
#SBATCH --gres=gpu:1
#SBATCH -t 1:0:0

. /etc/profile.d/modules.sh
module load python

source ../dad/bin/activate

python3 code/ceat_new.py 
