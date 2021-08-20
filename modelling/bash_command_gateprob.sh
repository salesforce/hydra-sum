for gate_prob in 0.25 0.5 0.75
do
  echo $gate_prob
  python3 train_seq2seq.py \
    --model_type bart_mult_heads_2 \
    --model_name_or_path facebook/bart-large \
    --input_dir ../../data/newsroom/mixed/lexical/model-bart-2heads-8layers/3 \
    --train_data_file ../../data/newsroom/mixed/lexical/train.tsv \
    --eval_data_file ../../data/newsroom/mixed/lexical/dev.tsv \
    --test_data_file ../../data/newsroom/mixed/lexical/test.tsv \
    --per_gpu_eval_batch_size=15 \
    --per_gpu_train_batch_size=2 \
    --gradient_accumulation_steps=8 \
    --num_train_epochs 10 \
    --learning_rate 1e-5 \
    --output_dir  ../../data/newsroom/mixed/lexical//model-bart-2heads-8layers/3/$gate_prob \
    --overwrite_output_dir \
    --save_steps 500  --gpu_device 4   --use_mixed --gate_probability $gate_prob  --generate  --do_eval  --max_seq_length 512
done