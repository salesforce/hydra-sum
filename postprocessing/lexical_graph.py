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
from scipy.stats import gaussian_kde
import seaborn as sns

def get_overlap(inp, out, ngram):
    grams_inp = set(ngrams(word_tokenize(inp.lower()), ngram))
    grams_out = set(ngrams(word_tokenize(out.lower()), ngram))

    total = len(grams_out)
    common = len(grams_inp.intersection(grams_out))
    if total == 0:
        return 0
    else:
        return float(common) / float(total)


def read_file_and_compute_overlap(folder, ngram=2, return_gold=False):
    baseline_file = open(os.path.join(folder, 'test_outfinal.txt'))
    lines = [line.strip() for line in baseline_file.readlines()]

    overlap_gold = []
    overlap_gen = []
    gen_comp = []
    gold_comp = []
    for i in range(0, len(lines), 4):
        inp = lines[i]
        gold = lines[i + 1]
        out = lines[i + 2]

        article_length = float(len(inp.split(' ')))

        overlap_gen.append(get_overlap(inp, out, ngram))
        gen_comp.append(float(len(out.split(' ')))/article_length)

        if return_gold:
            overlap_gold.append(get_overlap(inp, gold, ngram))
            gold_comp.append(float(len(gold.split(' ')))/article_length)

    print(len(lines))

    if return_gold:
        return overlap_gen, gen_comp, overlap_gold, gold_comp
    else:
        return overlap_gen, gen_comp


def generate_graph(folder, baseline_folder,  ngram=2):
    baseline_overlap_gen, base_line_gen_comp, overlap_gold, gold_comp = read_file_and_compute_overlap(baseline_folder,
                                                                                                      return_gold=True)
    heads = ['head0', 'head1']
    overlaps = []
    compressions = []
    for head in heads:
        subfolder = os.path.join(folder, head)
        ov, comp = read_file_and_compute_overlap(subfolder, return_gold=False)
        overlaps.append(ov)
        compressions.append(comp)

    # the histogram of the data
    #kwargs = dict(kind='kde', alpha=0.3, density=True, bins=40)
    sns.set_style('whitegrid')

    sns.kdeplot(overlap_gold, bw=0.5, label='gold')
    sns.kdeplot(baseline_overlap_gen, bw=0.5, label='baseline')

    for idx, head in enumerate(heads):
        sns.kdeplot(overlaps[idx], bw=0.5, label=head)

    plt.xlabel(f'{ngram}-gram overlap')
    plt.ylabel('Fraction of summaries')
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True)
    output_filename = os.path.join(folder, 'lexical.png')
    plt.savefig(output_filename)

    plt.clf()

    sns.set_style('whitegrid')
    sns.kdeplot(gold_comp, bw=0.5, label='gold')
    sns.kdeplot(base_line_gen_comp, bw=0.5, label='baseline')

    for idx, head in enumerate(heads):
        sns.kdeplot(compressions[idx], bw=0.5, label=head)

    plt.xlabel('Compression Ratio')
    plt.ylabel('Fraction of summaries')
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True)
    output_filename = os.path.join(folder, 'compression.png')
    plt.savefig(output_filename)


def generate_graph_prob(folder,  ngram=2):

    heads = ['head0', '0.25', '0.5', '0.75', 'head1']
    overlaps = []
    compressions = []
    for head in heads:
        subfolder = os.path.join(folder, head)
        ov, comp = read_file_and_compute_overlap(subfolder, return_gold=False)
        overlaps.append(ov)
        compressions.append(comp)


    sns.set_style('whitegrid')

    for idx, head in enumerate(heads):
        sns.kdeplot(overlaps[idx], bw=0.5, label=head)

    plt.xlabel(f'{ngram}-gram overlap')
    plt.ylabel('Fraction of summaries')
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True)
    output_filename = os.path.join(folder, 'lexical_prob.png')
    plt.savefig(output_filename)


if __name__ == '__main__':

    folder = '../../data/cnndm/cnn/lexical/model-bart-2heads-8layers/3'

    #baseline_folder = '../../data/newsroom/mixed/lexical/model-bart/3'

    generate_graph_prob(folder, 2)
