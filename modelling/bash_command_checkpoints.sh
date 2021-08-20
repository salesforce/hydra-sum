for folder in 1 2
do
  python3 train_seq2seq.py \
    --model_type bart_mult_heads_2 \
    --model_name_or_path facebook/bart-large \
    --input_dir ../../data/xsum/model-bart-2heads-8layers-2/$folder \
    --train_data_file ../../data/xsum/train.tsv \
    --eval_data_file ../../data/xsum/dev.tsv \
    --test_data_file ../../data/xsum/test.tsv \
    --per_gpu_eval_batch_size=15 \
    --per_gpu_train_batch_size=2 \
    --gradient_accumulation_steps=8 \
    --num_train_epochs 10 \
    --learning_rate 1e-5 \
    --output_dir  ../../data/xsum/model-bart-2heads-8layers-2/$folder/head1 \
    --overwrite_output_dir \
    --gpu_device 1   --use_head 1  --generate
done