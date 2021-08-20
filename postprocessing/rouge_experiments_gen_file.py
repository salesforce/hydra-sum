import os
from bert_score import score
from nltk.tokenize import sent_tokenize
import numpy as np

def get_bertscore(candidate, reference):
    (P, R, F), hashname = score(candidate, reference, lang="en", return_hash=True)
    return F.numpy()


def get_extractive_sentences(data):
    extracted_sents = []
    for ex in data:
        input_sents = ex['input_sents']
        output_sents = ex['output_sents']

        for out_sent_idx, out_sent in enumerate(output_sents):
            input_sents_filtered_idx = []
            bert_scores = ex['output_to_input_mappings'][out_sent_idx]
            for idx, (inp, b) in enumerate(zip(input_sents, bert_scores)):
                if b > 0.89:
                    input_sents_filtered_idx.append(idx)

            if len(input_sents_filtered_idx) == 0:
                max_idx = np.argmax(bert_scores)
                input_sents_filtered_idx.append(max_idx)

        input_sents_filtered_idx = sorted(set(input_sents_filtered_idx))
        input_sents_filtered = [input_sents[idx] for idx in input_sents_filtered_idx]

        input_sents_filtered = ' '.join(input_sents_filtered)
        extracted_sents.append(input_sents_filtered)

    return extracted_sents


def get_bert_scores_all(references, candidates):

    input_references = []
    input_candidates = []
    for ref, cand in zip(references, candidates):
        ref_sents = sent_tokenize(ref)
        cand_sents = sent_tokenize(cand)
        for cand_sent in cand_sents:
            for ref_sent in ref_sents:
                input_references.append(ref_sent)
                input_candidates.append(cand_sent)

    bert_score = get_bertscore(input_references, input_candidates)

    mapping = []
    index = 0
    for ref, cand in zip(references, candidates):
        ref_sents = sent_tokenize(ref)
        cand_sents = sent_tokenize(cand)
        temp = {'input': ref, 'output': cand, 'input_sents': ref_sents, 'output_sents': cand_sents,
                'output_to_input_mappings': []}
        for _ in cand_sents:
            candidate_temp = []
            for _ in ref_sents:
                candidate_temp.append(bert_score[index])
                index += 1
            temp['output_to_input_mappings'].append(candidate_temp)
        mapping.append(temp)

    return mapping


def get_extractive_sentences_file(file):
    lines = [line.strip() for line in file.readlines()]

    inputs = []
    references = []
    generations = []
    for i in range(0, len(lines), 4):
        inputs.append(lines[i])
        references.append(lines[i + 1])
        generations.append(lines[i + 2])

    bert_scores_generated = get_bert_scores_all(inputs, generations)
    extractive_sents = get_extractive_sentences(bert_scores_generated)

    return inputs, references, extractive_sents


def write_to_file(inputs, references, extractive_sents, output_file):

    for inp, ref, out in zip(inputs, references, extractive_sents):
        output_file.write(inp + '\n')
        output_file.write(ref + '\n')
        output_file.write(out + '\n\n')


if __name__ == '__main__':
    folder = '../../data/newsroom_old/mixed/20k/model-bart-multheads-4/head0'
    input_file = open(os.path.join(folder, 'dev_outfinal.txt'))

    output_file = open(os.path.join(folder, 'dev_outfinal_extractive.txt'), 'w')

    inputs, references, extractive_sents = get_extractive_sentences_file(input_file)
    write_to_file(inputs, references, extractive_sents, output_file)