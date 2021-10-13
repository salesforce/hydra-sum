'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
from nltk import word_tokenize, ngrams


def get_overlap(inp, out, ngram):
    grams_inp = set(ngrams(word_tokenize(inp.lower()), ngram))
    grams_out = set(ngrams(word_tokenize(out.lower()), ngram))

    total = len(grams_out)
    common = len(grams_inp.intersection(grams_out))
    if total == 0:
        return 0
    else:
        return float(common) / float(total)


def get_overlap_file(input_file, output_filename, ngram=2, graph=False):
    lines = [line.strip() for line in input_file.readlines()]

    overlap_gold = []
    overlap_gen = []
    gen_length = []
    gold_length = []
    for i in range(0, len(lines), 4):
        inp = lines[i]
        gold = lines[i + 1]
        out = lines[i + 2]

        overlap_gold.append(get_overlap(inp, gold, ngram))
        overlap_gen.append(get_overlap(inp, out, ngram))

        gen_length.append(len(out.split(' ')))
        gold_length.append(len(gold.split(' ')))

    print(len(lines))

    overlap_gold_mean = np.mean(overlap_gold)
    overlap_gen_mean = np.mean(overlap_gen)
    gen_length = np.mean(gen_length)
    gold_length = np.mean(gold_length)


    print(f'Gold overlap %dgram = %f' % (ngram, overlap_gold_mean))
    print(f'Generated overlap %dgram = %f' % (ngram, overlap_gen_mean))

    print(f'Gold length = %f' % gold_length)
    print(f'Generated length = %f' % gen_length)

    if graph:
        # the histogram of the data
        kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)

        weights = np.ones_like(overlap_gold) / float(len(overlap_gold))
        plt.hist(overlap_gold, **kwargs, label='gold', weights=weights)

        weights = np.ones_like(overlap_gen) / float(len(overlap_gold))
        plt.hist(overlap_gen, **kwargs, label='generated', weights=weights)

        plt.xlabel(f'{ngram}-gram overlap')
        plt.ylim(0, 8)
        #plt.xlim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.savefig(output_filename)
    return overlap_gold, overlap_gen


if __name__ == '__main__':
    """
    folder = '../../data/xsum/model-bart-3heads-8layers/3/head1'
    print(folder)
    input_file = open(os.path.join(folder, 'test_outfinal.txt'))
    output_filename = os.path.join(folder, 'lexical.png')
    overlap_gold, overlap_gen = get_overlap_file(input_file, output_filename, 2, graph=False)
    exit()"""

    folder = '../../data/cnndm/cnn/lexical/model-bart-2heads-8layers-2/'
    overlap_head0 = []
    overlap_head1 = []
    for idx in range(1, 11, 1):
        file_head0 = open(os.path.join(folder, str(idx), 'head0/dev_outfinal.txt'))
        overlap_gold, overlap_gen0 = get_overlap_file(file_head0, None, 2, graph=False)
        overlap_head0.append(np.mean(overlap_gen0))
        overlap_gold = np.mean(overlap_gold)

        file_head1 = open(os.path.join(folder, str(idx), 'head1/dev_outfinal.txt'))
        _, overlap_gen1 = get_overlap_file(file_head1, None, 2, graph=False)
        overlap_head1.append(np.mean(overlap_gen1))

    plt.figure()
    x = np.linspace(1, 11, 1)
    x_labels = list(range(1, 11, 1))
    plt.plot(x_labels, overlap_head0, label='head-0', color='r')
    plt.plot(x, [overlap_gold] * len(x), color='g', linestyle='--')
    plt.plot(x_labels, overlap_head1, label='head-1', color='b')

    plt.legend()
    plt.ylim([0, 1])
    plt.savefig(os.path.join(folder, 'lexical_overlap_checkpoints.png'))

    exit()
    """
    folder = '../../data/newsroom/mixed/20k/model-bart-multheads-8/'
    print(folder)
    output_filename = os.path.join(folder, 'lexical.png')

    ngram = 2
    input_file = open(os.path.join(folder, 'head0', 'dev_outfinal.txt'))
    overlap_gold, overlap_gen_head0 = get_overlap_file(input_file, None, ngram, graph=False)

    input_file = open(os.path.join(folder, 'head1', 'dev_outfinal.txt'))
    _, overlap_gen_head1 = get_overlap_file(input_file, None, ngram, graph=False)

    kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)

    plt.hist(overlap_gold, **kwargs, label='gold')
    plt.hist(overlap_gen_head0, **kwargs, label='gen head 0')
    plt.hist(overlap_gen_head1, **kwargs, label='gen head 1')

    plt.xlabel(f'{ngram}-gram overlap')

    plt.legend()
    plt.grid(True)
    plt.savefig(output_filename)"""

