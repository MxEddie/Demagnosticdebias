#!/bin/bash
#SBATCH -A NLP-CDT-SL2-GPU
#SBATCH -p ampere
#SBATCH -N1
#SBATCH -n1
#SBATCH --gres=gpu:1
#SBATCH -t 12:0:0

. /etc/profile.d/modules.sh
module load rhel8/default-amp
module load python/3.6

source ../../cd/bin/activate

model_type=bert
seed=$1
block_size=512
OUTPUT_DIR=/rds/user/$USER/hpc-work/preprocess/$seed/$model_type/data.bin

rm -r $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

python3 -u ../src/2preprocess.py --input /rds/user/hpcungl1/hpc-work/reddit_cd/RC_2018 \
                        --stereotypes ../data/warm_stereotypes.txt,../data/comp_stereotypes.txt \
                        --attributes ../data/EA_CD.txt,../data/AA_MF_CD.txt \
                        --output $OUTPUT_DIR \
                        --seed $seed \
                        --block_size $block_size \
                        --model_type $model_type \

