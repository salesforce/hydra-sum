from pycorenlp import StanfordCoreNLP
import csv
from nltk import Tree
import numpy as np
import os
import nltk
nlp = StanfordCoreNLP('http://localhost:9000')



def compute_perplexity(ll):
    return np.exp2(-np.sum(ll)/len(ll))


def get_top_parse(input_text):
    tokenized_json = nlp.annotate(input_text, properties={'annotators': 'tokenize, parse', 'outputFormat': 'json',
                                                          'ssplit.isOneSentence': True})
    parse = tokenized_json['sentences'][0]['parse']
    tree = Tree.fromstring(parse)
    productions = tree.productions()
    base_l = productions[0].lhs()
    base_r = productions[0].rhs()
    top_level_l = productions[1].lhs()
    top_level_r = productions[1].rhs()

    return top_level_r


def get_top_level_parse(input_file):
    lines = [l.strip() for l in input_file.readlines()]

    top_level_parses_file = {}

    for i in range(0, len(lines), 4):
        article_text = lines[i].strip()
        gold = lines[i + 1].strip()
        summary = lines[i + 2].strip()

        top_parse = str(get_top_parse(summary))
        if top_parse not in top_level_parses_file:
            top_level_parses_file[top_parse] = 0

        top_level_parses_file[top_parse] += 1

    sorted_parses = dict(sorted(top_level_parses_file.items(), key=lambda item: item[1]))
    print(sorted_parses)


if __name__ == '__main__':

    folder = '../../data/cnndm/cnn/lexical/model-bart-2heads-8layers/3/head1/'
    print(folder)

    input_file = open(os.path.join(folder, 'test_outfinal.txt'))
    output_filename = os.path.join(folder, 'lexical.png')
    get_top_level_parse(input_file)
