"""Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause"""

import os, csv, torch, copy
import logging
from torch.utils.data import TensorDataset
from torch import nn
from nltk import sent_tokenize

logger = logging.getLogger(__name__)


def _read_tsv(input_file, quoting=csv.QUOTE_MINIMAL):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=quoting)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def get_examples(filename):
    return _read_tsv(os.path.join(filename))


class InputFeatures(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def convert_examples_to_features(examples, tokenizer, max_length=512, max_decoder_length=128):
    features = []
    pad_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % ex_index)

        input = example['article']
        output = example['summary']
        id = example['id']

        if input == '' or output == '':
            continue

        input_ids = tokenizer.encode(input, add_prefix_space=True)

        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length - 1]

        padding_length_a = max_length - len(input_ids)
        input_attention_mask = [1] * len(input_ids) + ([0] * padding_length_a)
        input_ids = input_ids + ([pad_id] * padding_length_a)

        """
        decoder_ids = tokenizer.encode(output, add_prefix_space=True)

        if 'gate' in example.keys():
            decoder_ids_2 = [tokenizer.encode(x, add_special_tokens=False, add_prefix_space=True) for x in
                             output.split(' ')]

            decoder_ids_2_flattened = [item for sublist in decoder_ids_2 for item in sublist]
            assert decoder_ids_2_flattened == decoder_ids[1:-1], 'mismatch in splitting w/ gating_supervision'

            gate_tokenlevel = [int(x) for x in example['gate'].split(' ')]
            gate_wplevel = []
            assert len(decoder_ids_2) == len(gate_tokenlevel), 'mismatch in splitting w/ gating_supervision'
            for idx in range(len(decoder_ids_2)):
                gate_wplevel += [gate_tokenlevel[idx]] * len(decoder_ids_2[idx])
            gate_wplevel += [-1, -1]  # makin length equal to decoder_ids
        else:
            gate_wplevel = [-1] * len(decoder_ids)

        assert len(gate_wplevel) == len(decoder_ids), 'mismatch in splitting w/ gating_supervision'"""

        if 'gate_sent' in example.keys():
            sent_gates = [float(g) for g in example['gate_sent'].split()] # previously int
            output_sents = sent_tokenize(output)
            assert len(sent_gates) == len(output_sents), 'mismatch in splitting w/ gating_supervision'

            decoder_ids = []
            gate_sent = []
            for sent, g in zip(output_sents, sent_gates):
                decoder_ids_sent = tokenizer.encode(sent, add_prefix_space=True)
                gate_sent += [g] * len(decoder_ids_sent)
                decoder_ids += decoder_ids_sent

        else:
            decoder_ids = tokenizer.encode(output, add_prefix_space=True)
            gate_sent = [0] * len(decoder_ids)

        if len(decoder_ids) > max_decoder_length:
            decoder_ids = decoder_ids[:max_decoder_length - 1]
            # gate_wplevel = gate_wplevel[:max_decoder_length - 1]
            gate_sent = gate_sent[:max_decoder_length - 1]

        padding_length_b = max_decoder_length - len(decoder_ids)
        decoder_attention_mask = [1] * len(decoder_ids) + ([0] * padding_length_b)
        decoder_ids = decoder_ids + ([pad_id] * padding_length_b)
        # gate_wplevel = gate_wplevel + ([-1] * padding_length_b)
        gate_sent = gate_sent + ([0] * padding_length_b)

        features.append(InputFeatures(input_ids=input_ids,
                                      attention=input_attention_mask,
                                      decoder_attention=decoder_attention_mask,
                                      decoder_ids=decoder_ids,
                                      id=id,
                                      #gate=gate_wplevel,
                                      sent_gate=gate_sent))

    print(len(features))
    return features


def convert_examples_to_features_pegasus(examples, tokenizer, max_length=512, max_decoder_length=128):
    features = []
    pad_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % ex_index)

        input = example['article']
        output = example['summary']
        try:
            id = example['id']
        except:
            id = ex_index

        if input == '' or output == '':
            continue

        input_ids = tokenizer.encode(input)
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length - 1]

        padding_length_a = max_length - len(input_ids)
        input_attention_mask = [1] * len(input_ids) + ([0] * padding_length_a)
        input_ids = input_ids + ([pad_id] * padding_length_a)

        decoder_ids = tokenizer.encode(output)
        if len(decoder_ids) > max_decoder_length:
            decoder_ids = decoder_ids[:max_decoder_length - 1]

        padding_length_b = max_decoder_length - len(decoder_ids)
        decoder_attention_mask = [1] * len(decoder_ids) + ([0] * padding_length_b)
        decoder_ids = decoder_ids + ([pad_id] * padding_length_b)
        gate_wplevel = [-1] * max_decoder_length
        sent_gate = 0

        features.append(InputFeatures(input_ids=input_ids,
                                      attention=input_attention_mask,
                                      decoder_attention=decoder_attention_mask,
                                      decoder_ids=decoder_ids,
                                      id=id,
                                      gate=gate_wplevel,
                                      sent_gate=sent_gate))
    print(len(features))
    return features


def load_and_cache_examples(args, tokenizer, split):
    if split == 'dev':
        data_dir = '/'.join(args.eval_data_file.split('/')[:-1])
        file_name = args.eval_data_file
    elif split == 'test':
        data_dir = '/'.join(args.test_data_file.split('/')[:-1])
        file_name = args.test_data_file
    else:
        data_dir = '/'.join(args.train_data_file.split('/')[:-1])
        file_name = args.train_data_file

    model_type = args.model_type
    if model_type == 'bart_subpop' and split == 'train':
        model_type_prefix = model_type
    else:
        model_type_prefix = model_type.split('_')[0]

    cached_features_file = os.path.join(
        data_dir,
        "cached_{}_{}_{}".format(
            split,
            model_type_prefix,
            str(args.max_seq_length)
        ),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        examples = get_examples(file_name)
        if model_type == 'bart_subpop' and split == 'train':
            gate = args.subpop
            examples_new = []
            for ex in examples:
                if int(ex['gate_sent']) == gate:
                    examples_new.append(ex)

            examples = examples_new
        features = convert_examples_to_features(
            examples,
            tokenizer,
            max_length=args.max_seq_length,
            max_decoder_length=args.max_decoder_length,
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_attention_mask = torch.tensor([f.attention for f in features], dtype=torch.long)
    decoder_ids = torch.tensor([f.decoder_ids for f in features], dtype=torch.long)
    decoder_attention_mask = torch.tensor([f.decoder_attention for f in features], dtype=torch.long)
    # gate = torch.tensor([f.gate for f in features], dtype=torch.long)
    sent_gate = torch.tensor([f.sent_gate for f in features], dtype=torch.float)  # previously long

    dataset = TensorDataset(input_ids, input_attention_mask, decoder_ids, decoder_attention_mask, sent_gate, # FIX THIS
                            sent_gate)

    return dataset


def fix_endtoken_weight(weights, decoder_attention):
    batch_size = weights.shape[0]
    num_decoder_length = torch.sum(decoder_attention, dim=1) - 2  # get the index of </s> token
    j = torch.arange(batch_size).long()
    weights[j, num_decoder_length] = 1
    return weights


def shift_tokens_left(input_ids, pad_token_id):
    """Shift input ids one token to the left"""
    prev_output_tokens = input_ids.clone()
    prev_output_tokens[:, :-1] = input_ids[:, 1:]
    prev_output_tokens[:, -1] = pad_token_id
    return prev_output_tokens


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer
