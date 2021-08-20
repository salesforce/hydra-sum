import datasets
import argparse
import re
from nltk import sent_tokenize

metric = datasets.load_metric('rouge')

def clean_addnewline(line):
    line = line.strip().lower()
    line_sents = sent_tokenize(line)
    line = '\n'.join(line_sents)  # rouge-l is very sensitive to line breaks. uses \n as identifier
    line = re.sub('<.*?>', '', line)
    return line


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default="",
                        help='input file')
    args = parser.parse_args()

    input_file = open(args.input_file)
    lines = [l.strip() for l in input_file.readlines()]

    candidates = []
    references = []
    num_line_per_ex = 4

    for i in range(0, len(lines), num_line_per_ex):
        print(i)
        gold = lines[i + 1]
        summary = lines[i + 2]
        print(summary)

        candidates.append(summary)
        references.append(gold)
        assert lines[i + num_line_per_ex - 1].strip() == '', 'some error in file, pls check generated file'

    print(len(candidates))
    # calculate rouge

    final_score = metric.compute(predictions=candidates, references=references)
    r1 = final_score['rouge1'].mid.fmeasure
    r2 = final_score['rouge2'].mid.fmeasure
    rl = final_score['rougeL'].mid.fmeasure

    print(f'Rouge1: {r1}')
    print(f'Rouge2: {r2}')
    print(f'RougeL: {rl}')
