#!/bin/bash
#SBATCH -A NLP-CDT-SL2-GPU
#SBATCH -p ampere
#SBATCH -N1
#SBATCH -n1
#SBATCH --gres=gpu:1
#SBATCH -t 4:0:0

. /etc/profile.d/modules.sh
module load rhel8/default-amp
module load python/3.6

source ../dad/bin/activate

python3 --version

python3 code/generate_ebd_bert_new.py 
