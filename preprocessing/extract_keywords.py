'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
# repurposed from the ctrl-sum code.


# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
import csv
import os
import argparse
import random
import pickle
import json
import spacy
import re
import sys
import subprocess
import numpy as np
import pandas as pd

import stanza

from typing import List, Dict, Any, Optional, Tuple
from collections import (
    defaultdict,
    OrderedDict,
    namedtuple,
    Counter,
)

from multiprocessing import Pool

from spacy.lang.en import English
from spacy.tokens import Token, Doc
from spacy.tokenizer import Tokenizer


def tokenize(inp_file: str,
             tokenizer: Tokenizer,
             split: str,
             annotator: Optional[str] = None,
             batch_size=100,
             max_position=1024,
             max_tgt_position=256,
             save_to_file=True,
             datadir: Optional[str] = None) -> Dict:
    """perform tokenization and sentence segmentation,
    return results or save results to a pickle file
    (if save_to_file is True).
    Args:
        src_file: the summarization source file
        tgt_file: the summarization target file
        tokenizer: the spacy tokenizer
        split: split name
        annotator: some annotation names to disambiguate saved files
            (only used when save_to_file is True)
        max_position: the maximum length of the source file (before tokenization),
            the source will be truncated automatically
        max_tgt_position: the maximum length of the target file (before tokenization),
            the target will be truncated automatically
        save_to_file: whether to save the tokenization results into a pickle file
        datadir: the dataset directory (only used when save_to_file is True)
    Returns:
        a dictionary that contains the tokenized spacy.tokens.Doc objects
            for every source and target example
    """

    def tokenize_batch(batched, data):
        src_docs = tokenizer.pipe([x[1] for x in batched], batch_size=batch_size)
        tgt_docs = tokenizer.pipe([x[2] for x in batched], batch_size=batch_size)
        id_list = [x[0] for x in batched]
        for id_, src_doc, tgt_doc in zip(id_list, src_docs, tgt_docs):
            assert id_ not in data
            data[id_] = {'id': id_,
                         'src_doc': src_doc.to_bytes() if save_to_file else src_doc,
                         'tgt_doc': tgt_doc.to_bytes() if save_to_file else tgt_doc,
                         }

    def truncate(x, max_len):
        x_s = x.rstrip().split()
        max_len = min(len(x_s), max_len)
        return ' '.join(x_s[:max_len])

    data = {}
    f = csv.DictReader(open(inp_file), delimiter='\t')
    batched = []
    for i, d in enumerate(f):
        src_l = d['article']
        tgt_l = d['summary']
        batched.append((i, truncate(src_l, max_position), truncate(tgt_l, max_tgt_position)))
        if (i + 1) % batch_size == 0:
            tokenize_batch(batched, data)
            batched = []

        if i % 1000 == 0:
            print("processed {} lines".format(i))

    if len(batched) > 0:
        tokenize_batch(batched, data)
        batched = []

    return data


def get_tokens(text, tokens):
    """get list of tokenized text
    """

    # remove the linebreak symbol
    return [text[tok['start']:tok['end']] for tok in tokens[:-1]]


def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
      n: which n-grams to calculate
      text: An array of tokens
    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def _greedy_selection(doc_sent_list: List[List[str]],
                      abstract_sent_list: List[List[str]],
                      summary_size: int) -> List[int]:
    """select sentences from the source to maximum its ROUGE scores with the oracle summries.
    Borrowed from BertSum: https://github.com/nlpyang/BertSum/blob/9aa6ab84fa/src/prepro/data_builder.py.
    we select candidate sentences to maximum ROUGE-Recall scores.
    Args:
        doc_sent_list: the source list of sentences
        abstract_sent_list: the target list of sentences
        summary_size: the number of maximum source sentences that can be selected
    Returns:
        the sorted id of selected sentences
    """

    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    # Following is from BertSum to maximum rouge score w.r.t. the whole summary
    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))

            # use recall score
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['r']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['r']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            # return selected
            return sorted(selected)
            # maybe should be sorted(selected)
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def oracle_sent(split,
                annotator: Optional[str] = None,
                summary_size=3,
                data: Optional[Dict] = None,
                save_to_file=True,
                datadir=None) -> Dict:
    """The sentence selection step in the pipeline. This step is after the tokenization step
    This function greedily selects source sentences to maximize the ROUGE scores with the
    summaries.
    Args:
        split: the split name
        annotator: the annotation name
        summary_size: the maximum number of selected source sentences
        data: the tokenized data dict from last step in the pipeline
        save_to_file: whether to save the tokenization results into a pickle file
        datadir: the dataset directory (only used when save_to_file is True)
    Returns:
        a dictionary that contains all the preprocessing results until this stepss
    """
    nlp = English()

    # if test_extract:
    #     fout_src = open('{}.extsents'.format(split) ,'w')
    #     fout_tgt = open('{}.extsents.gold'.format(split) ,'w')
    from_file = False
    if data is None:
        with open('{}.tok.pickle'.format(split), 'rb') as fin:
            data = pickle.load(fin)
        from_file = True

    new_data = {}
    print('finish loading pickle data from {}'.format(split))

    for i in range(len(data)):
        k, v = i, data[i]
        if i % 1000 == 0:
            print("processed {} examples".format(i))
        src_doc = Doc(nlp.vocab).from_bytes(v['src_doc']) if from_file else v['src_doc']
        tgt_doc = Doc(nlp.vocab).from_bytes(v['tgt_doc']) if from_file else v['tgt_doc']
        doc_sent_list = [[token.text for token in sent] for sent in src_doc.sents]
        abs_sent_list = [[token.text for token in sent] for sent in tgt_doc.sents]
        # selected = greedy_selection(doc_sent_list, abs_sent_list, summary_size)
        # summary size equal to the gold
        selected = _greedy_selection(doc_sent_list, abs_sent_list, max(summary_size, len(abs_sent_list)))
        data[k].update({'oracle_sents': selected})

        # debug purpose
        # if test_extract:
        #     ext = sum([doc_sent_list[s] for s in selected], [])
        #     abs = sum(abs_sent_list, [])
        #     fout_src.write('{}\n'.format(' '.join(ext)))
        #     fout_tgt.write('{}\n'.format(' '.join(abs)))
    if save_to_file:
        with(open(f'{datadir}/{split}.{annotator}.pickle', 'wb')) as outfile:
            pickle.dump(data, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    return data


def write_ext_sent(split, datadir):
    """load the preprocessing results from picle file and write the
    selected sentences into text files
    """

    prefix = f'{datadir}/{split}'
    nlp = English()
    with open(f'{prefix}.oracle_ext.pickle', 'rb') as fin, \
            open(f'{prefix}.extsents', 'w') as fout_src, \
            open(f'{prefix}.extsents.gold', 'w') as fout_tgt:
        data = pickle.load(fin)
        print('finish loading pickle data from {}'.format(split))

        for i in range(len(data)):
            k, v = i, data[i]
            if i % 1000 == 0:
                print("processed {} examples".format(i))

            src_doc = Doc(nlp.vocab).from_bytes(v['src_doc'])
            tgt_doc = Doc(nlp.vocab).from_bytes(v['tgt_doc'])
            doc_sent_list = [[token.text for token in sent] for sent in src_doc.sents]
            ext = sum([doc_sent_list[s] for s in v['oracle_sents']], [])
            abs = [tok.text for tok in tgt_doc]
            fout_src.write('{}\n'.format(' '.join(ext)))
            fout_tgt.write('{}\n'.format(' '.join(abs)))


def _extract_word(p_text: List[str],
                  e_text: List[str],
                  p_token: List[spacy.tokens.Token],
                  e_token: List[spacy.tokens.Token]) -> List[Tuple[int, int]]:
    """obtain oracle keywords given source and target sentences
    Args:
        p_text: the source word list
        e_text: the target word list
        p_token: the spacy.Token list corresponding to p_text
        e_token: the spacy.Token list corresponding to e_text
    Returns:
        a list of tuples where the first item is the keyword id in the source document,
            while the second item is the keyword id in the target document
    """

    res = []

    # modified based on
    # https://github.com/sebastianGehrmann/bottom-up-summary/blob/master/preprocess_copy.py
    # tsplit = t.split()
    def getsubidx(x, y):
        if len(y) == 0:
            return None

        l1, l2 = len(x), len(y)
        for i in range(l1):
            if x[i:i + l2] == y:
                return i

        return None

    startix = 0
    endix = 1
    matches = []
    src_set = set()
    tgt_set = set()
    while endix <= len(p_text):
        # last check is to make sure that phrases at end can be copied
        tgt_idx = getsubidx(e_text, p_text[startix: endix])
        if tgt_idx is not None and endix <= len(p_text):
            endix += 1
        else:
            # only phrases, not words
            # uncomment the -1 if you only want phrases > len 1
            if startix >= endix - 1:
                endix += 1
            else:
                # restrict to not select single stop word separately
                if not (endix - 1 == startix + 1 and p_token[startix].is_stop):
                    for offset, loc in enumerate(range(startix, endix - 1)):
                        if loc not in src_set and (prev_idx + offset) not in tgt_set \
                                and not p_token[loc].is_stop:
                            res.append((p_token[loc].i, e_token[prev_idx + offset].i))
                            src_set.update([loc])
                            tgt_set.update([prev_idx + offset])
                # endix += 1
            startix = endix - 1

        prev_idx = tgt_idx

    # deal with the corner case matching to the end
    if endix - 1 > startix and not (endix - 1 == startix + 1 and p_token[startix].is_stop):
        for offset, loc in enumerate(range(startix, endix - 1)):
            if loc not in src_set and (prev_idx + offset) not in tgt_set \
                    and not p_token[loc].is_stop:
                res.append((p_token[loc].i, e_token[prev_idx + offset].i))
                src_set.update([loc])
                tgt_set.update([prev_idx + offset])

    return res


def oracle_keyword(split,
                   annotator=None,
                   data=None,
                   save_to_file=True,
                   datadir=None,
                   ):
    """the keyword extraction step in the preprocessing pipeline. This step is
    after the oracle_sent step
    Args:
        split: the split name
        annotator: the annotation name
        data: the tokenized data dict from last step in the pipeline
        save_to_file: whether to save the tokenization results into a pickle file
        datadir: the dataset directory (only used when save_to_file is True)
    Returns:
        a dictionary that contains all the preprocessing results until this steps
    """
    nlp = English()
    # vocab = defaultdict(lambda: len(vocab))
    from_file = False
    if data is None:
        with open(f'{datadir}/{split}.oracle_ext.pickle', 'rb') as fin:
            data = pickle.load(fin)
            print('finish loading pickle data from {}'.format(split))
        from_file = True

    cnt = 0
    for i in range(len(data)):
        k, v = i, data[i]
        if i % 1000 == 0:
            print("processed {} examples".format(i))

        src_doc = Doc(nlp.vocab).from_bytes(v['src_doc']) if from_file else v['src_doc']
        tgt_doc = Doc(nlp.vocab).from_bytes(v['tgt_doc']) if from_file else v['tgt_doc']
        doc_sent_list = [[token for token in sent] for sent in src_doc.sents]
        ext_sents = sum([doc_sent_list[s] for s in v['oracle_sents']], [])
        ext_sents = [tok for tok in ext_sents if not tok.is_punct]
        tgt_tok = [tok for tok in tgt_doc if not tok.is_punct]
        abs = [tok.text for tok in tgt_tok]
        # print(i)
        ext_sents_text = [x.text for x in ext_sents]
        # ext_sents_id = [vocab[x] for x in ext_sents_text]
        # abs_id = [vocab[x] for x in abs]
        selected = _extract_word(ext_sents_text, abs, ext_sents, tgt_tok)
        for (src_loc, tgt_loc) in selected:
            assert src_doc[src_loc].text == tgt_doc[tgt_loc].text

        data[k].update({'oracle_tok': selected})

    if save_to_file:
        with(open(f'{datadir}/{split}.{annotator}.pickle', 'wb')) as outfile:
            pickle.dump(data, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    return data


def entity_tag(split, src, datadir, filter_user=True):
    """tag entities in the source document and generate a keyword file
    {split}.entitywords{[filter]} that uses the entities as keywords
    Args:
        split: the split name
        src: the source document suffix
        datadir: the dataset directory
        filter_user: if True, only select entites given in the `user_ent_type`
            variable. This is to better simulate user perferences since many
            entity types are unlikely to be specified by users
    """
    # spacy.prefer_gpu()
    # nlp = spacy.load("en_core_web_sm", disable=['tagger', 'parser'])
    user_ent_type = set(['EVENT', 'FAC', 'GPE', 'LAW', 'NORP', 'ORG', 'PERSON', 'PRODUCT', 'WORK_OF_ART'])
    batch_size = 64
    nlp = stanza.Pipeline(lang='en', processors='tokenize,ner', ner_batch_size=batch_size, tokenize_no_ssplit=True)
    batches = []
    no_entity = 0
    prefix = f'{datadir}/{split}'
    postfix = 'filter' if filter_user else ''
    with open(f'{prefix}.{src}') as fin, \
            open(f'{prefix}.entitywords{postfix}', 'w') as fout:
        for i, line in enumerate(fin):
            batches.append(line.strip())

            if (i + 1) % batch_size == 0:
                for doc in nlp('\n\n'.join(batches)).sentences:
                    flag = False
                    for ent in doc.ents:
                        if not filter_user or ent.type in user_ent_type:
                            flag = True
                            fout.write(ent.text + ' ')
                    # for tok in doc:
                    #     if tok.ent_type_ != '':
                    #         flag = True
                    #         fout.write(tok.text + ' ')
                    fout.write('\n')
                    if not flag:
                        no_entity += 1

                batches = []

            if i % 1000 == 0:
                print('processing {} lines'.format(i))

        if batches != []:
            for doc in nlp('\n\n'.join(batches)).sentences:
                flag = False
                for ent in doc.ents:
                    if not filter_user or ent.type in user_ent_type:
                        flag = True
                        fout.write(ent.text + ' ')
                # for tok in doc:
                #     if tok.ent_type_ != '':
                #         flag = True
                #         fout.write(tok.text + ' ')
                fout.write('\n')
                if not flag:
                    no_entity += 1

    print('{} examples without entity detected'.format(no_entity))


def auto_truncate(in_path, out_path, max_len):
    """truncate the input file and write to `out_path`
    """
    with open(in_path) as fin, \
            open(out_path, 'w') as fout:
        for x in fin:
            x_s = x.rstrip().split()
            max_len_s = min(len(x_s), max_len)
            fout.write(' '.join(x_s[:max_len_s]) + '\n')


def add_prefix(split, src, datadir, prefix: str):
    """add a prefix string to every example in the source file.
    Separated with the '=>' token
    """

    datapath = f'{datadir}/{split}'

    with open(f'{datapath}.{src}') as fsrc, \
            open(f'{datapath}.prefix{src}lead', 'w') as fout_src:
        data = fsrc.readlines()
        data = [x.strip() for x in data]

        for line in data:
            fout_src.write(f' {prefix} => {line}\n')


def pipeline(split, outfilename, args):
    """the entire preprocessing pipeline consisting of tokenization,
    selecting oracle key sentences, selecting oracle keywords, and preparing
    sequence labeling dataset to tag keywords. This function should be able
    generate all the required data files to train and evaluate the summarization model
    Args:
        split: the split name
        suffix: output file suffix for disambiguation purpose
        offset: mainly for parallel processing purpose
    """
    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)

    # an issue for spacy default tokenization, see http://www.longest.io/2018/01/27/spacy-custom-tokenization.html
    def custom_tokenizer(nlp):
        prefix_re = spacy.util.compile_prefix_regex(nlp.Defaults.prefixes)
        suffix_re = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes)
        custom_infixes = ['\.\.\.+', '(?<=[0-9])-(?=[0-9])', '[!&:,()]']
        infix_re = spacy.util.compile_infix_regex(tuple(list(nlp.Defaults.infixes) + custom_infixes))

        tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab,
                                              nlp.Defaults.tokenizer_exceptions,
                                              prefix_search=prefix_re.search,
                                              suffix_search=suffix_re.search,
                                              infix_finditer=infix_re.finditer,
                                              token_match=None)

        return tokenizer

    if args.pretokenize:
        nlp.tokenizer = Tokenizer(nlp.vocab)
    else:
        nlp.tokenizer = custom_tokenizer(nlp)

    print(f"----- tokenize split '{split}' -------")
    data = tokenize(f'{args.datadir}/{split}.tsv',
                    nlp, split,
                    max_position=args.max_position,
                    max_tgt_position=args.max_tgt_position,
                    save_to_file=False,
                    datadir=None)

    print(f"----- greedy-selection of sentences split '{split}' --------")
    data = oracle_sent(split, summary_size=args.summary_size, data=data, save_to_file=False)

    print(f"----- extract keywords split '{split}' --------")
    data = oracle_keyword(split, data=data, save_to_file=False)
    print(len(data))

    outfile = open(outfilename, 'w')
    fieldnames = ['id', 'article', 'summary', 'oracle_toks']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()

    for idx in range(len(data)):
        d = data[idx]
        oracle_tokens = ' '.join([str(d['src_doc'][x[0]]) for x in d['oracle_tok']])
        temp = {'id': d['id'], 'article': d['src_doc'], 'summary': d['tgt_doc'], 'oracle_toks': oracle_tokens}
        writer.writerow(temp)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='various preprocessing for summarization task')
    parser.add_argument('--datadir', type=str, help='the data directory')
    parser.add_argument('--mode', type=str, choices=['truncate', 'process_tagger_prediction',
                                                     'prepend_oracle_len', 'human_study_entity', 'human_study_purpose',
                                                     'get_keyword_len',
                                                     'pipeline'],
                        help='preprocessing mode. Please see the comments doc of each function for details')

    parser.add_argument('--max-position', type=int, default=1024,
                        help='maximum source length')
    parser.add_argument('--max-tgt-position', type=int, default=256,
                        help='maximum target length')
    parser.add_argument('--src', type=str, default='source',
                        help='source file suffix')
    parser.add_argument('--tgt', type=str, default='target',
                        help='target file suffix')
    parser.add_argument('--outfix', type=str, default='default',
                        help='output file suffix to disambiguate')
    parser.add_argument('--split', type=str, default=None,
                        help='the specific split to preprocess. If None then processing all splits')
    parser.add_argument('--pretokenize', action='store_true', default=False,
                        help='whether the input data is already tokenized')
    parser.add_argument('--tag-pred', type=str, default=None,
                        help='prediction file from tagger')

    # keyword selection hyperparams
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='the confidence threshold to select keywords from tagger output')
    parser.add_argument('--maximum-word', type=int, default=25,
                        help='maximum number of keywords for tagger')
    parser.add_argument('--summary-size', type=int, default=3,
                        help='maximum number of firstly extracted sentences before keyword extraction')

    parser.add_argument('--sent-separator', type=str, default='|',
                        help='if specified, include a sentence separator in the keywords')
    parser.add_argument('--num-workers', type=int, default=20,
                        help='number of processes in the pipeline mode, 1 to disable')

    args = parser.parse_args()

    mode_to_id = {'truncate': 'trunc', 'tokenize': 'tok', 'greedy_selection':
        'oracle_ext', 'align': 'oracle_word'}
    indicator = mode_to_id.get(args.mode, None)

    print("start mode {}".format(args.mode))

    split_list = ['dev', 'train', 'test'] if args.split is None else args.split.split(',')
    for split in split_list:
        outfile = os.path.join(args.datadir, 'keywords', f'{split}.tsv')
        pipeline(split, outfile, args)

