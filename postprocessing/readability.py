"""Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause"""

from scipy import stats
import os
import csv
import numpy as np
import textstat
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')


def get_readability(file, output_filename, graph=False):
    readability_gold = []
    readability_ref = []
    readability_diff = []
    readability_inp = []

    lines = [x.strip() for x in file.readlines()]

    for i in range(0, len(lines), 4):
        inp = lines[i]
        gold = lines[i + 1]
        out = lines[i + 2]

        r_gold = textstat.flesch_reading_ease(gold)
        r_inp = textstat.flesch_reading_ease(inp)
        r_ref = textstat.flesch_reading_ease(out)
        readability_diff.append(np.abs(r_gold - r_inp))
        readability_gold.append(r_gold)
        readability_ref.append(r_ref)
        readability_inp.append(r_inp)

    correlation, p_val = stats.pearsonr(readability_gold, readability_inp)
    print(correlation)
    print(p_val)

    if graph:
        print('herer')
        # the histogram of the data
        kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)

        plt.hist(readability_ref, **kwargs, label='gold')
        plt.hist(readability_gold, **kwargs, label='gen')
        # plt.hist(readability_inp, **kwargs, label='input')
        #plt.xlim([0, 100])
        plt.ylim([0, 0.05])
        plt.legend()
        plt.grid(True)
        plt.savefig(output_filename)

    return readability_gold


if __name__ == '__main__':
    folder = '../../data/cnndm/cnn/model-bart/6.0/'
    print(folder)
    input_file = open(os.path.join(folder, 'dev_outfinal.txt'))
    output_filename = os.path.join(folder, 'readability.png')
    readability_gold = get_readability(input_file, output_filename, graph=True)
