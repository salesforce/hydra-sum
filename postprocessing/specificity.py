import os
import time
import subprocess
import argparse
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np

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
    lines = [line.strip() for line in input_file.readlines()]

    temp_file_gold = open(output_file, 'w')
    temp_file_gen = open(output_file + 'gen', 'w')
    for i in range(0, len(lines), 4):
        gold = lines[i + 1]
        gen = lines[i + 2]

        temp_file_gold.write(f'{gold}\n')
        temp_file_gen.write(f'{gen}\n')


def combine_scores(input_file, output_dir, graph):
    input_file_gold = open(input_file)
    scores_file_gold = open(input_file + 'out')
    input_file_gen = open(input_file + 'gen')
    scores_file_gen = open(input_file + 'genout')

    gold_scores = [float(s.strip()) for s in scores_file_gold.readlines()]
    gen_scores = [float(s.strip()) for s in scores_file_gen.readlines()]

    print(np.mean(gen_scores))

    output_filename = os.path.join(output_dir, 'specificity.png')
    if graph:
        kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)
        plt.hist(gold_scores, **kwargs, label='gold')
        plt.hist(gen_scores, **kwargs, label='gen')

        plt.xlabel('specificity')
        plt.xlim(0, 1)
        plt.ylim(0, 7)
        plt.legend()
        plt.savefig(output_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="Input directory with the dave_outfinal.txt file",
    )
    parser.add_argument(
        "--input_file",
        default=None,
        type=str,
        required=False,
        help="Input directory with the dave_outfinal.txt file",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        type=str,
        required=False,
        help="Input directory with the dave_outfinal.txt file",
    )
    parser.add_argument(
        "--function",
        default=None,
        type=str,
        required=False,
        help="Input directory with the dave_outfinal.txt file",
    )

    args = parser.parse_args()

    function = args.function

    if function == 'write_files':
        input_file = open(os.path.join(args.input_dir, 'dev_outfinal.txt'))
        output_file = args.output_file
        generate_specificity_file(input_file, output_file)
    elif function == 'combine_scores':
        combine_scores(args.input_file, args.input_dir, graph=True)