# HydraSum - Disentangling Stylistic Features in Text Summarization using Multi-Decoder Models
Paper Link: https://arxiv.org/abs/2110.04400
Authors: Tanya Goyal, Nazneen Fatema Rajani, Wenhao Liu, Wojciech Kryściński

Environment base is Python 3.6. Also see requirements.txt

## Introduction
We introduce a new summarization model HydraSum, that extends the single decoder framework of current models, e.g. BART, to a mixture-of-experts version consisting of multiple decoders. Our proposed model encourages each expert, i.e. decoder, to learn and generate stylistically-distinct summaries along dimensions such as abstractiveness, length, specificity, and others. At each time step, HydraSum employs a gating mechanism that decides the contribution of each individual decoder to the next token's output probability distribution. 

This repository contains code and model checkpoints to train and evaluate 2 kinds of models:
1. Unguided Training: 2- and 3- decoder models trained without explicitly controlling how summary styles get partitioned between multiple decoders. These models correspond to results in Section 3.1 of the paper. 
2. Guided Training: 2- deocder models trained to partition along certain target styles, namely abstractivess and specificity. E.g., if the target style is abstractiveness, decoder 0 is trained to generate highly abstractive summaries and decoder 1 is trained to generate highly extractive summarires. These models correspond to results in Section 3.2 of the paper. 

## Running Code
Download the data from the link above (to be  updated). For the three datasets - CNN, Newsroom, XSum, we've included subfolders: unguided, abstractiveness, specificity corresponding to data for the unguided setting, abstractiveness guided setting and specificity guided setting respectively. Each tsv file includes input article, summary, and gate probabilites (i.e. the gate probability derived from the percentile split it belongs to, see Section 2 of the paper for details) for train, dev and test examples.

### Training
Let DATA_DIR point to the folder with train.tsv, dev.tsv and test.tsv files. To train models under the unguided setting, run the following command:
```
python3 train_seq2seq.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path facebook/bart-large \
    --do_train \
    --train_data_file $DATA_DIR/train.tsv \
    --eval_data_file $DATA_DIR/dev.tsv \
    --test_data_file $DATA_DIR/test.tsv \
    --per_gpu_eval_batch_size=2 \
    --per_gpu_train_batch_size=2 \
    --gradient_accumulation_steps=1 \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --output_dir $OUTPUT_DIR \
    --num_decoder_layers_shared 8
```
The var $MODEL_TYPE can be set to ```bart_mult_heads_2```, ```bart_mult_heads_3``` or ```bart``` to train 2-decoder, 3-decoder or the standard bart models respectively. For the first two models, i.e. the HydraSum models, set the ```num_decoder_layers_shared``` argument to specify the number of shared layers between the multiple decoders.

To train under the guided setting, additionally set the ```use_sentence_gate_supervision```. The control style is determined by the $DATA_DIR folder. 

### Inference
As outlined in Section 2.2, we can use 3 inference strategies to generate summaries:

1. (Inference Strategy 1) Generate summaries using **Individual Decoders**. The decoder to use can be specified using the ```use_head``` argument. Choose from 0/1 for a 2-decoder model, or 0/1/2 for a 3-decoder model.
```
python3 train_seq2seq.py \
--model_type $MODEL_TYPE \
--model_name_or_path facebook/bart-large \
--input_dir $MODEL_DIR \
--test_data_file $DATA_DIR/test.tsv \
--per_gpu_eval_batch_size=8 \
--output_dir $OUTPUT_DIR \
--overwrite_output_dir \  
--generate  --use_head 0
```

2. (Inference Strategy 2) **Mixture using G**, i.e., the mixture weights are decided by the model. Only valid when using a model trained under the unguided setting.
```
python3 train_seq2seq.py \
--model_type $MODEL_TYPE \
--model_name_or_path facebook/bart-large \
--input_dir $MODEL_DIR \
--test_data_file $DATA_DIR/test.tsv \
--per_gpu_eval_batch_size=8 \
--output_dir $OUTPUT_DIR \
--overwrite_output_dir \  
--generate  --use_mixed
```

3. (Inference Strategy 3) **Mixture using manually-specified g**. The contribution of each decoder can be specified by g, i.e. the mixture coefficients are [1-g, g]
```
python3 train_seq2seq.py \
--model_type $MODEL_TYPE \
--model_name_or_path facebook/bart-large \
--input_dir $MODEL_DIR \
--test_data_file $DATA_DIR/test.tsv \
--per_gpu_eval_batch_size=8 \
--output_dir $OUTPUT_DIR \
--overwrite_output_dir \  
--generate  --use_mixed   --gate_probability g
```
 
 ## Citation
 ```
 @article{goyal2021hydrasum,
      title={HydraSum -- Disentangling Stylistic Features in Text Summarization using Multi-Decoder Models}, 
      author={Tanya Goyal and Nazneen Fatema Rajani and Wenhao Liu and Wojciech Kryściński},
      year={2021},
      journal={arXiv preprint arXiv:2110.04400},
}
```
