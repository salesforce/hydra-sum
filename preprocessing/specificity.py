'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
import csv
import os
import time
import subprocess
import argparse
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
from metrics.lexical_overlap import get_overlap
from scipy.stats import spearmanr
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


def run_speciteller(filename):
    bashCommand = f'python speciteller.py --inputfile ../{filename} --outputfile ../{filename}_out'
    print(bashCommand)
    subprocess.run(bashCommand.split(), cwd='./speciteller/')

    file = open(f'{filename}_out')
    scores = [float(s.strip()) for s in file.readlines()]

    bashCommand = f'rm {filename} {filename}_out'
    subprocess.run(bashCommand.split())

    return scores


def generate_specificity_file(input_file, output_file):
    data = csv.DictReader(input_file, delimiter='\t')

    temp_file = open(output_file, 'w')
    for d in data:
        temp_file.write(f'{d["summary"]}\n')


def combine_scores(score_file, input_file, output_file_name):
    scores_file_gold = open(score_file)
    gold_scores = [float(s.strip()) for s in scores_file_gold.readlines()]
    mean_score = np.mean(gold_scores)
    sd = np.std(gold_scores)

    input_file_data = open(input_file)
    data = csv.DictReader(input_file_data, delimiter='\t')

    output_file = open(output_file_name, 'w')
    fieldnames = ['id', 'article', 'summary', 'sent_gate', 'compression_bin', 'coverage', 'coverage_bin', 'density', 'density_bin', 'compression']
    writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()

    lengths0 = []
    lengths1 = []

    for d, sc in zip(data, gold_scores):
        if sc > mean_score + sd:
            d['sent_gate'] = 1
            writer.writerow(d)
            lengths1.append(len(d['summary'].split(' ')))
        elif sc < mean_score - sd:
            d['sent_gate'] = 0
            writer.writerow(d)
            lengths0.append(len(d['summary'].split(' ')))
        else:
            continue

    print(np.mean(lengths0))
    print(np.mean(lengths1))

def correlation(score_file, input_file, output_file_name):
    scores_file_gold = open(score_file)
    gold_scores = [float(s.strip()) for s in scores_file_gold.readlines()]

    input_file_data = open(input_file)
    data = csv.DictReader(input_file_data, delimiter='\t')

    lexical_overlaps = []
    lengths = []
    for d in data:
        summary = d['summary']
        article = d['article']

        lexical_overlaps.append(get_overlap(article, summary, 2))
        lengths.append(len(summary.split(' ')))

    assert len(lexical_overlaps) == len(gold_scores) == len(lengths), 'mismatch in length'

    corr_sp_lo, pval_sp_lo = spearmanr(lexical_overlaps, gold_scores)
    corr_sp_length, pval_sp_length = spearmanr(lengths, gold_scores)
    corr_length_lo, pval_length_lo = spearmanr(lengths, lexical_overlaps)

    print(f'b/w lexical overlap and specificity: corr = {corr_sp_lo:.6f}, p-value={pval_sp_lo:.6f}')
    print(f'b/w specificity and length: corr = {corr_sp_length:.6f}, p-value={pval_sp_length:.6f}')
    print(f'b/w lexical overlap and length: corr = {corr_length_lo:.6f}, p-value={pval_length_lo:.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--input_file",
        default=None,
        type=str,
        required=False,
        help="Input file ",
    )
    # Required parameters
    parser.add_argument(
        "--output_file",
        default=None,
        type=str,
        required=False,
        help="Input directory with the dave_outfinal.txt file",
    )
    parser.add_argument(
        "--intermediate_file",
        default=None,
        type=str,
        required=False,
        help="int files, will be deleted",
    )
    parser.add_argument(
        "--function",
        default=None,
        type=str,
        required=False,
        help="function",
    )

    args = parser.parse_args()

    function = args.function

    if function == 'write_files':
        input_file = open(args.input_file)
        output_file = args.intermediate_file
        generate_specificity_file(input_file, output_file)
    elif function == 'combine_scores':
        combine_scores(args.intermediate_file, args.input_file, args.output_file)
    elif function == 'correlation':
        correlation(args.intermediate_file, args.input_file, args.output_file)
