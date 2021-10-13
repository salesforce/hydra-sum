'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
from nltk import word_tokenize
import os
import csv


def get_extractive_fragments(article, summary):
    article_tokens = word_tokenize(article.lower())
    summary_tokens = word_tokenize(summary.lower())

    F = []
    i, j = 0, 0
    while i < len(summary_tokens):
        f = []
        while j < len(article_tokens):
            if summary_tokens[i] == article_tokens[j]:
                i_, j_ = i, j
                while summary_tokens[i_] == article_tokens[j_]:
                    i_, j_ = i_ + 1, j_ + 1
                    if i_ >= len(summary_tokens) or j_ >= len(article_tokens):
                        break
                if len(f) < (i_ - i):
                    f = list(range(i, i_))
                j = j_
            else:
                j = j + 1
        i, j = i + max(len(f), 1), 1
        F.append(f)

    return F, article_tokens, summary_tokens


def get_extractive_coverage(article, summary):
    frags, article_tokens, summary_tokens = get_extractive_fragments(article, summary)
    coverage = float(sum([len(f) for f in frags]))/float(len(summary_tokens))
    return coverage

def get_fragment_density(article, summary):
    frags, article_tokens, summary_tokens = get_extractive_fragments(article, summary)
    density = float(sum([len(f)**2 for f in frags]))/float(len(summary_tokens))
    return density


def _read_tsv(input_file, quoting=csv.QUOTE_MINIMAL):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=quoting)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


if __name__=='__main__':
    input_file = '../../data/newsroom/mixed/train.tsv'
    data = _read_tsv(input_file)

    for d in data:
        article = d['article']
        summary = d['summary']

        density = get_fragment_density(article, summary)
        converage = get_extractive_coverage(article, summary)

        break
