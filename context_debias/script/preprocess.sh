model_type=bert
seed=$1
block_size=512
OUTPUT_DIR=/output_dir/

rm -r $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

python3 -u ../src/2preprocess.py --input /input_data \
                        --stereotypes ../data/warm_stereotypes.txt,../data/comp_stereotypes.txt \
                        --attributes ../data/EA_CD.txt,../data/AA_MF_CD.txt \
                        --output $OUTPUT_DIR \
                        --seed $seed \
                        --block_size $block_size \
                        --model_type $model_type \

