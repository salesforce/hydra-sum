import argparse
import json
import logging
import os
import random
from typing import Dict
import numpy as np
import torch
import math
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import train_seq2seq_utils
import single_head_utils
import multi_head_utils
import topic_utils
from torch import nn

from transformers import (
    AdamW,
    PreTrainedModel,
    PreTrainedTokenizer,
    BartConfig,
    BartTokenizer,
    PegasusConfig,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)
MODEL_CLASSES = {"bart": (BartConfig,
                          single_head_utils.ConditionalGenerationCustomBart,
                          BartTokenizer),
                 "pegasus": (PegasusConfig,
                             PegasusForConditionalGeneration,
                             PegasusTokenizer),
                 "bart_mult_heads": (BartConfig,
                                     multi_head_utils.ConditionalGenerationCustomBartMultHeads,
                                     BartTokenizer),
                 "bart_topic": (BartConfig,
                                topic_utils.ConditionalGenerationCustomBartTopic,
                                BartTokenizer)
                 }


class BartModelCombined(nn.Module):
    def __init__(self, model1, model2):
        super().__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, **model_kwargs):
        return


def load_model(path):
    args = json.load(open(path))
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config)
    return model




def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--base_model",
        default=None,
        type=str,
        help="base model, used to load tokenizer",
    )
    parser.add_argument(
        "--model_1_config",
        default=None,
        type=str,
        help="Path to model 1 config",
    )
    parser.add_argument(
        "--model_2_config",
        default=None,
        type=str,
        required=True,
        help="Path to model 2 config",
    )
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        required=True,
        help="Evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=1024,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--max_decoder_length",
        default=128,
        type=int,
        help="The maximum total decoder sequence length after tokenization.",
    )
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int, help="Batch size evaluation.", )
    parser.add_argument("--gpu_device", type=int, default=0, help="gpu device")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached data sets", )

    # custom flags
    parser.add_argument("--generate", action="store_true", help="Generate summaries for dev set", )
    parser.add_argument("--dump_posteriors", action="store_true", help="Dump posterior probs at intermediate steps", )
    parser.add_argument("--gate_probability", type=float, default=None, help="gate prob")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.n_gpu = 1
    device = torch.device("cuda", args.gpu_device)
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=os.path.join(args.output_dir, 'model.log')
    )

    # Set seed
    model1 = load_model(args.model_1_config)
    model1.to(args.device)

    model2 = load_model(args.model_2_config)
    model2.to(args.device)


    if args.base_model == 'bart':
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    elif args.base_model == 'pegasus':
        tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')

    eval_dataset = train_seq2seq_utils.load_and_cache_examples(args, tokenizer, evaluate=True)
    # evaluate(args, eval_dataset, model1, model2, tokenizer, 'final')

    logger.info("Training/evaluation parameters %s", args)


if __name__ == "__main__":
    main()
