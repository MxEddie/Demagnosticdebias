model_type=bert
seed=21

gpu=0
debias_layer=all # first last all
loss_target=token # token sentence
dev_data_size=1000
alpha=0.2
beta=0.8
lr=0.00005

if [ $model_type = 'bert' ]; then
    model_name_or_path=bert-base-cased
fi

TRAIN_DATA=/preprocess/$seed/$model_type/data.bin/data.bin
OUTPUT_DIR=/debiased_models/$seed/$model_type/$lr/$alpha

rm -r $OUTPUT_DIR
echo $model_type $seed

CUDA_VISIBLE_DEVICES=$gpu python -u ../src/run.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=$model_type \
    --model_name_or_path=$model_name_or_path \
    --do_train \
    --data_file=$TRAIN_DATA \
    --do_eval \
    --evaluate_during_training \
    --learning_rate $lr \
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 32 \
    --num_train_epochs 3 \
    --block_size 512 \
    --loss_target $loss_target \
    --debias_layer $debias_layer \
    --seed $seed \
    --weighted_loss $alpha $beta \
    --dev_data_size $dev_data_size \
    --square_loss \
    --line_by_line \
    #--overwrite_cache \
    #--max_steps 10000
    
