'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
import csv
import os
import string
from nltk import word_tokenize, ngrams
import numpy as np

punctuations = string.punctuation

def _read_tsv(input_file, quoting=csv.QUOTE_MINIMAL):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=quoting)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


# if token is seen, head1, if token is unseen head0
def get_gate_type0(article, summary):
    gate = []
    article_tokens = set(article.lower().split(' '))
    summary_tokens = summary.lower().split(' ')
    for token in summary_tokens:
        if token in punctuations:
            gate.append('-1')
        elif token in article_tokens:
            gate.append('1')
        else:
            gate.append('0')

    assert len(gate) == len(summary_tokens), 'gate length does not match summary length'
    return gate


# if token is unseen and context is unseen, head1, if token is seen and context is seen head0
def get_gate_type2(article, summary):
    gate = []
    article_tokens = set(article.split(' '))
    summary_tokens = summary.split(' ')
    for idx, token in enumerate(summary_tokens):
        if idx < 3:
            gate.append('-1')
        elif token in punctuations:
            gate.append('-1')
        else:
            prefix_wtoken = ' '.join(summary_tokens[idx-2: idx + 1])
            prefix = ' '.join(summary_tokens[idx-2: idx])
            if prefix_wtoken in article:
                gate.append('1')
            elif prefix not in article and token not in article:
                gate.append('0')
            else:
                gate.append('-1')

    assert len(gate) == len(summary_tokens), 'gate length does not match summary length'
    return gate


def get_overlap(inp, out, ngram=2):
    grams_inp = set(ngrams(word_tokenize(inp.lower()), ngram))
    grams_out = set(ngrams(word_tokenize(out.lower()), ngram))

    total = len(grams_out)
    common = len(grams_inp.intersection(grams_out))
    if total == 0:
        return 0
    else:
        return float(common) / float(total)


def get_gate_type3(article, summary, mean_overlap):
    overlap = get_overlap(article, summary)
    if overlap < mean_overlap:
        return 0
    else:
        return 1



if __name__=='__main__':
    input_folder = '../../data/xsum/original_data'
    output_folder = '../../data/xsum/'

    split = 'train.tsv'
    input_file = os.path.join(input_folder, split)
    data = _read_tsv(input_file)

    outfile = open(os.path.join(output_folder, split), 'w')
    fieldnames = list(data[0].keys()) + ['gate', 'gate_sent']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, delimiter='\t')
    writer.writeheader()

    if split == 'train.tsv':
        mean_overlap = []
        for ex in data:
            article = ex['article']
            summary = ex['summary']
            overlap = get_overlap(article.lower(), summary.lower())
            mean_overlap.append(overlap)
        mean_overlap = np.mean(overlap)
    else:
        print('Please provide mean overlap based on the train file.')

    print(mean_overlap)


    num_0s = 0.
    num_1s = 0.
    num_blank = 0.
    num_1sent = 0.

    for ex in data:
        article = ex['article']
        summary = ex['summary']

        gate = get_gate_type0(article.lower(), summary.lower())
        gate_sent = get_gate_type3(article.lower(), summary.lower(), mean_overlap)

        ex['gate'] = ' '.join(gate)
        ex['gate_sent'] = str(gate_sent)
        num_0s += gate.count('0')
        num_1s += gate.count('1')
        num_blank += gate.count('-1')
        num_1sent += gate_sent

        writer.writerow(ex)


print(num_1s/(num_0s + num_1s + num_blank))
print(num_0s/(num_0s + num_1s + num_blank))
print(num_blank/(num_0s + num_1s + num_blank))

print(num_1sent/len(data))