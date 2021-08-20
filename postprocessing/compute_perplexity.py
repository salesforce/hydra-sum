'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
import os
import numpy as np


def read_next(file):
    input = file.readline().strip()
    if input == '':
        return None
    gold = file.readline().strip()

    token_level = []
    while True:
        temp = file.readline().strip()
        if temp == '':
            break

        token = temp.split('\t')[0]
        score = float(temp.split('\t')[1])

        token_level.append((token, score))

    file.readline()

    return (input, gold, token_level)


def compute_perplexity(ll):
    return np.exp2(-np.sum(ll)/len(ll))


def get_perplexity_file(input_file):
    log_likelihood_list = []
    examples = []
    while True:
        ex = read_next(input_file)
        if ex is not None:
            examples.append(ex)
        else:
            break

    for e in examples:
        for t in e[2]:
            log_likelihood_list.append(np.log(t[1]))

    perplexity = compute_perplexity(log_likelihood_list)

    return perplexity


def plot_graph(input_folder, range_start, range_end, stepsize, num_line_per_ex=4):

    perplexity_list = []
    for k in range(range_start, range_end, stepsize):
        input_file = open(os.path.join(input_folder, f'prob_out{k}.txt'))
        perp = get_perplexity_file(input_file)

        perplexity_list.append(perp)

    plt.figure()
    x = list(range(range_start, range_end, stepsize))
    plt.plot(x, perplexity_list, label='perplexity', color='r')

    plt.legend()
    plt.savefig(os.path.join(input_folder, 'perplexity.png'))


def plot_graph_paraphrases(root_folder, range_start, range_end, stepsize):

    perplexity_list = []
    for k in range(range_start, range_end, stepsize):
        input_file = open(os.path.join(root_folder, f'prob_out{k}.txt'))
        perp = get_perplexity_file(input_file)

        perplexity_list.append(perp)

    para_perplexity_list = []
    for k in range(range_start, range_end, stepsize):
        input_file = open(os.path.join(root_folder, 'paraphrased_outputs',  f'prob_out{k}.txt'))
        perp = get_perplexity_file(input_file)

        para_perplexity_list.append(perp)

    plt.figure()
    x = list(range(range_start, range_end, stepsize))
    plt.plot(x, perplexity_list, label='original', color='r')
    plt.plot(x, para_perplexity_list, label='paraphrases', color='b')

    plt.legend()
    plt.savefig(os.path.join(root_folder, 'paraphrased_perplexity.png'))


if __name__=='__main__':

    input_file = open('../../../data/cnndm/cnn/football/model-bart/prob_outood.txt')
    perp = get_perplexity_file(input_file)
    print(perp)
