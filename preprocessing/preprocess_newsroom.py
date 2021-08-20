'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
import json
import os
import csv
import re
from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')

input_folder = '../../data/newsroom_old'
output_folder_root = '../../data/newsroom'
splits = ['train', 'dev', 'test']


def remove_whitespace(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text


def get_tokenized(text):
    tokenized_json = nlp.annotate(text, properties={'annotators': 'tokenize', 'outputFormat': 'json',
                                                    'ssplit.isOneSentence': True})
    tokenized_text = []
    # print(tokenized_json)
    try:
        for tok in tokenized_json['tokens']:
            tokenized_text.append(tok['word'])
        tokenized_text = ' '.join(tokenized_text[:1500])  # truncating articles
    except:
        tokenized_text = ''
    return tokenized_text


bins = ['extractive', 'abstractive', 'mixed', 'all']
for bin in bins:
    for split in splits:
        input_file = open(os.path.join(input_folder, f'{split}.jsonl'))
        output_folder = os.path.join(output_folder_root, bin)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_file = open(os.path.join(output_folder, f'{split}.tsv'), 'w')
        fieldnames = ['id', 'article', 'summary', 'compression', 'coverage', 'density', 'compression_bin',
                      'coverage_bin', 'density_bin']
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()

        idx = 0
        for line in input_file.readlines():
            line = json.loads(line)
            row = {'compression': line['compression'],
                   'coverage': line['coverage'],
                   'density': line['density'],
                   'compression_bin': line['compression_bin'],
                   'coverage_bin': line['coverage_bin'],
                   'density_bin': line['density_bin']}

            if bin == 'all':
                article = get_tokenized(remove_whitespace(line['text']))
                summary = get_tokenized(remove_whitespace(line['summary']))
                if article == '' or summary == '':
                    continue
                row.update({'id': idx, 'article': article, 'summary': summary})
                writer.writerow(row)
            elif line['density_bin'] == bin:
                article = get_tokenized(remove_whitespace(line['text']))
                summary = get_tokenized(remove_whitespace(line['summary']))
                if article == '' or summary == '':
                    continue
                row.update({'id': idx, 'article': article, 'summary': summary})
                writer.writerow(row)
