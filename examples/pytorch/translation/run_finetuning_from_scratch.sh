export CUDA_VISIBLE_DEVICES=0
python run_finetuning_from_scratch.py \
    --model_name_or_path t5-small \
    --config_name ./config_12_2.json \
    --do_train \
    --do_eval \
    --source_lang de \
    --target_lang en \
    --dataset_name iwslt2017 \
    --dataset_config_name iwslt2017-de-en \
    --output_dir /scratch/chooper/iwslt/tst-translation \
    --source_prefix 'translate English to German: ' \
    --per_device_train_batch_size=48 \
    --per_device_eval_batch_size=48 \
    --overwrite_output_dir \
    --cache_dir /scratch/chooper/iwslt/cache-translation \
    --num_train_epochs 50 \
    --predict_with_generate
