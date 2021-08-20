import csv
import os
import textstat


def _read_tsv(input_file, quoting=csv.QUOTE_MINIMAL):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=quoting)
        lines = []
        for line in reader:
            lines.append(line)
        return lines



def get_readability_gate(summary, mean_read):
    r_gold = textstat.flesch_reading_ease(summary)
    if r_gold > mean_read:
        return 1
    else:
        return 0


if __name__ == '__main__':
    input_folder = '../../data/cnndm/cnn/original_data'
    output_folder = '../../data/cnndm/cnn/readability_data'

    split = 'dev.tsv'
    input_file = os.path.join(input_folder, split)
    data = _read_tsv(input_file)

    outfile = open(os.path.join(output_folder, split), 'w')
    fieldnames = list(data[0].keys()) + ['gate_sent']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, delimiter='\t')
    writer.writeheader()

    mean_Readability = 50.

    for ex in data:
        article = ex['article']
        summary = ex['summary']

        gate_sent = get_readability_gate(summary, mean_Readability)

        ex['gate_sent'] = str(gate_sent)

        writer.writerow(ex)
