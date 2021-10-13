"""Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause"""

import argparse
import os
import re
import time
from multiprocessing import Pool

import shutil
from nltk import sent_tokenize
from pyrouge import Rouge155


def process(data):
    candidates, references, pool_id = data
    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = "rouge-tmp-{}-{}".format(current_time, pool_id)
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:
        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])

        r = Rouge155()
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate(rouge_args="-e /export/home/pyrouge/rouge/tools/ROUGE-1.5.5/data -c 95 "
                                                          "-m -n 2 -a")
        # print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def test_rouge(cand, ref, num_processes):
    """Calculate ROUGE scores of sequences passed as an iterator
       e.g. a list of str, an open file, StringIO or even sys.stdin
    """
    candidates = [line.strip() for line in cand]
    references = [line.strip() for line in ref]

    # print(len(candidates))
    # print(len(references))
    assert len(candidates) == len(references)
    candidates_chunks = list(chunks(candidates, int(len(candidates) / num_processes)))
    references_chunks = list(chunks(references, int(len(references) / num_processes)))
    n_pool = len(candidates_chunks)
    arg_lst = []
    for i in range(n_pool):
        arg_lst.append((candidates_chunks[i], references_chunks[i], i))
    pool = Pool(n_pool)
    results = pool.map(process, arg_lst)
    final_results = {}
    for i, r in enumerate(results):
        for k in r:
            if (k not in final_results):
                final_results[k] = r[k] * len(candidates_chunks[i])
            else:
                final_results[k] += r[k] * len(candidates_chunks[i])
    for k in final_results:
        final_results[k] = final_results[k] / len(candidates)
    return final_results


def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-P(1/2/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        # results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,

        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        # results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_recall"] * 100,

        results_dict["rouge_1_precision"] * 100,
        results_dict["rouge_2_precision"] * 100,
        # results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_precision"] * 100
    )


def read_file(input_file):
    lines = [l.strip() for l in input_file.readlines()]

    inputs = []
    candidates = []
    references = []
    num_line_per_ex = 4

    for i in range(0, len(lines), num_line_per_ex):
        input = lines[i].strip()
        input = re.sub('<.*?>', '', input)
        gold = lines[i + 1].strip()
        gold = re.sub('<.*?>', '', gold)
        summary = lines[i + 2].strip()
        summary = re.sub('<.*?>', '', summary)
        inputs.append(input)
        candidates.append(summary)
        references.append(gold)

    return inputs, references, candidates

"""
def get_tokenized(text):
    tokenized_json = nlp.annotate(text, properties={'annotators': 'tokenize', 'outputFormat': 'json',
                                                    'ssplit.isOneSentence': True})
    tokenized_text = []
    # print(tokenized_json)
    for tok in tokenized_json['tokens']:
        tokenized_text.append(tok['word'])
    tokenized_text = ' '.join(tokenized_text)  # truncating articles

    return tokenized_text"""


def clean_addnewline(line):
    line = line.strip().lower()
    # line = get_tokenized(line)
    line_sents = sent_tokenize(line)
    line = '\n'.join(line_sents)  # rouge-l is very sensitive to line breaks. uses \n as identifier
    # line = re.sub('<.*?>', '', line)
    return line


if __name__ == "__main__":
    # init_logger('test_rouge.log')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default="",
                        help='input file')
    parser.add_argument('--p', type=int, default=1,
                        help='number of processes')
    args = parser.parse_args()


    input_file = open(args.input_file)
    lines = [l.strip() for l in input_file.readlines()]

    candidates = []
    references = []
    num_line_per_ex = 4

    for i in range(0, len(lines), num_line_per_ex):
        gold = clean_addnewline(lines[i + 1])
        summary = clean_addnewline(lines[i + 2])
        candidates.append(summary)
        references.append(gold)

        assert lines[i + num_line_per_ex - 1].strip() == '', 'some error in file, pls check generated file'

    print(len(candidates))
    # calculate rouge
    results_dict = test_rouge(candidates, references, args.p)
    print(rouge_results_to_str(results_dict))

    exit()

    folder = args.input_file
    input_file = open(os.path.join(folder, 'head0', 'test_outfinal.txt'))
    inputs, references, candidates0 = read_file(input_file)

    input_file = open(os.path.join(folder, 'head1', 'test_outfinal.txt'))
    _, _, candidates1 = read_file(input_file)

    results_dict = test_rouge(candidates1, candidates0, args.p)
    print(rouge_results_to_str(results_dict))
