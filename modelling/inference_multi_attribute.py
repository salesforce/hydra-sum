import argparse
import json
import logging
import os
import torch
from transformers.file_utils import ModelOutput
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader, SequentialSampler
from transformers.modeling_outputs import Seq2SeqLMOutput
import train_seq2seq_utils
import single_head_utils
import multi_head_utils
from torch import nn
from generation_utils_multi_attribute import GenerationMixinCustomCombined

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    BartConfig,
    BartTokenizer
)

logger = logging.getLogger(__name__)
MODEL_CLASSES = {"bart_mult_heads_2": (BartConfig,
                                     multi_head_utils.ConditionalGenerationCustomBartMultHeads,
                                     BartTokenizer),
                 }


class Seq2SeqLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values_1: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values_2: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


class BartModelCombined(GenerationMixinCustomCombined, nn.Module):
    def __init__(self, model1, model2, config: BartConfig):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.config = config
        self.device = model2.device

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs_1=None,
            encoder_outputs_2=None,
            past_key_values_1=None,
            past_key_values_2=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=None,
            use_mixed=False,
            use_head_1=0,
            use_head_2=0,
            gate_prob=0.5,
    ):
        args1 = {'input_ids': input_ids,
                 'attention_mask': attention_mask,
                 'decoder_input_ids': decoder_input_ids,
                 'decoder_attention_mask': decoder_attention_mask,
                 'head_mask': head_mask,
                 'decoder_head_mask': decoder_head_mask,
                 'cross_attn_head_mask': cross_attn_head_mask,
                 'encoder_outputs': encoder_outputs_1,
                 'past_key_values': past_key_values_1,
                 'inputs_embeds': inputs_embeds,
                 'use_cache': use_cache,
                 'output_attentions': False,
                 'output_hidden_states': False,
                 'return_dict': None,
                 'use_mixed': False,
                 'use_head': use_head_1,
                 }

        out1 = self.model1(**args1)
        softmax_0 = torch.exp(out1.logits)

        args2 = {'input_ids': input_ids,
                 'attention_mask': attention_mask,
                 'decoder_input_ids': decoder_input_ids,
                 'decoder_attention_mask': decoder_attention_mask,
                 'head_mask': head_mask,
                 'decoder_head_mask': decoder_head_mask,
                 'cross_attn_head_mask': cross_attn_head_mask,
                 'encoder_outputs': encoder_outputs_2,
                 'past_key_values': past_key_values_2,
                 'inputs_embeds': inputs_embeds,
                 'use_cache': use_cache,
                 'output_attentions': output_attentions,
                 'output_hidden_states': output_hidden_states,
                 'return_dict': None,
                 'use_mixed': False,
                 'use_head': use_head_2,
                 }

        out2 = self.model2(**args2)
        softmax_1 = torch.exp(out2.logits)

        softmax_0 = softmax_0 * gate_prob
        softmax_1 = softmax_1 * (1 - gate_prob)

        lm_logits = torch.log(softmax_0 + softmax_1)
        return_output = Seq2SeqLMOutput(
            logits=lm_logits,
            past_key_values_1=out1.past_key_values,
            past_key_values_2=out2.past_key_values)

        return return_output

    # unchanged
    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past_1=None,
            past_2=None,
            attention_mask=None,
            head_mask=None,
            use_cache=None,
            encoder_outputs_1=None,
            encoder_outputs_2=None,
            **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past_1 is not None and past_2 is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs_1": encoder_outputs_1,
            "encoder_outputs_2": encoder_outputs_2,
            "past_key_values_1": past_1,
            "past_key_values_2": past_2,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }


def load_model(path):
    args = json.load(open(path))
    config_class, model_class = BartConfig, multi_head_utils.ConditionalGenerationCustomBartMultHeads
    config = config_class.from_pretrained(args['path'])
    model = model_class.from_pretrained(
        args['path'],
        from_tf=bool(".ckpt" in args['path']),
        config=config)
    return model, args, config


def evaluate(args, eval_dataset, model: PreTrainedModel, args1, args2, tokenizer: PreTrainedTokenizer,
             suffix="") -> Dict:
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    if args.generate:
        f_out = open(os.path.join(eval_output_dir, 'test_out%s.txt' % suffix), 'w')
        print(eval_output_dir)
        k = 0

        with torch.no_grad():
            model.eval()

            for batch in eval_dataloader:

                batch = tuple(t.to(args.device) for t in batch)
                input_ids, input_attention_mask, decoder_ids = batch[0], batch[1], batch[2]

                for j in range(input_ids.shape[0]):
                    gold = tokenizer.decode(decoder_ids[j], skip_special_tokens=True)
                    input = tokenizer.decode(input_ids[j], skip_special_tokens=True)

                    input_args = {'input_ids': input_ids[j].unsqueeze(0),
                                  'attention_mask': input_attention_mask[j].unsqueeze(0), 'num_beams': 6,
                                  'length_penalty': 2, 'no_repeat_ngram_size': 3, 'max_length': 200, 'min_length': 12,
                                  'top_k': 30, 'top_p': 0.5, 'do_sample': True,
                                  'decoder_start_token_id': tokenizer.bos_token_id, 'num_return_sequences': 1,
                                  'gate_prob': args.gate_probability, 'use_head_1': args1['use_head'],
                                  'use_head_2': args2['use_head']}

                    gen = model.generate(**input_args)

                    gen = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                           gen]

                    # gen = gen[0]
                    print(gen[0].strip())

                    f_out.write(input + '\n')
                    f_out.write(gold + '\n')
                    for g in gen:
                        f_out.write(g.strip() + '\n')
                    f_out.write('\n')

                k += 1
                if k > 1000:
                    break

            f_out.close()


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
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
        "--test_data_file",
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
    model1, args1, config = load_model(args.model_1_config)
    model1.to(args.device)

    model2, args2, _ = load_model(args.model_2_config)
    model2.to(args.device)

    f_out = open(os.path.join(args.output_dir, 'model_configs.json'), 'w')
    json.dump(args1, f_out)
    f_out.write('\n')
    json.dump(args2, f_out)
    f_out.write('\n')
    json.dump({'gate_prob': args.gate_probability}, f_out)
    f_out.write('\n')
    f_out.close()

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    model = BartModelCombined(model1, model2, config)
    eval_dataset = train_seq2seq_utils.load_and_cache_examples(args, tokenizer, 'test')
    evaluate(args, eval_dataset, model, args1, args2, tokenizer, 'final')

    logger.info("Training/evaluation parameters %s", args)


if __name__ == "__main__":
    main()
