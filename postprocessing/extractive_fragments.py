'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
import os
from postprocessing.extractive_fragments_utils import get_extractive_coverage, get_fragment_density
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np


def get_extraction_file(input_file, output_filename, graph=False):
    lines = [line.strip() for line in input_file.readlines()]

    coverage_gold = []
    coverage_gen = []
    density_gold = []
    density_gen = []
    gen_length = []
    gold_length = []
    for i in range(0, len(lines), 4):
        inp = lines[i]
        gold = lines[i + 1]
        out = lines[i + 2]

        coverage_gold.append(get_extractive_coverage(inp, gold))
        coverage_gen.append(get_extractive_coverage(inp, out))

        density_gold.append(get_fragment_density(inp, gold))
        density_gen.append(get_fragment_density(inp, out))

        gen_length.append(len(out.split(' ')))
        gold_length.append(len(gold.split(' ')))

    print(len(lines))

    coverage_gold_mean = np.mean(coverage_gold)
    coverage_gen_mean = np.mean(coverage_gen)

    density_gold_mean = np.mean(density_gold)
    density_gen_mean = np.mean(density_gen)

    gen_length = np.mean(gen_length)
    gold_length = np.mean(gold_length)


    print(f'Gold coverage = %f' % coverage_gold_mean)
    print(f'Generated coverage = %f' % coverage_gen_mean)

    print(f'Gold density = %f' % density_gold_mean)
    print(f'Generated density = %f' % density_gen_mean)

    print(f'Gold length = %f' % gold_length)
    print(f'Generated length = %f' % gen_length)


    if graph:
        kwargs = dict(alpha=0.3)

        plt.scatter(coverage_gen, density_gen, label='gen', color='red', **kwargs)
        plt.scatter(coverage_gold, density_gold, label='gold', color='blue', **kwargs)

        plt.xlabel('Coverage')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_filename)


if __name__ == '__main__':

    folder = '../data/newsroom_old/mixed/20k/model-bart-multheads-4/head0/'
    print(folder)
    input_file = open(os.path.join(folder, 'dev_outfinal.txt'))
    output_filename = os.path.join(folder, 'extractive.png')
    get_extraction_file(input_file, output_filename, graph=True)
