import os
from bert_score import score
from nltk.tokenize import sent_tokenize
import numpy as np


def get_bertscore(candidate, reference):
    (P, R, F), hashname = score(candidate, reference, lang="en", return_hash=True)
    return F.numpy()


def get_position_bias(data, num_buckets=10):

    position_based_scores = [0 for _ in range(num_buckets)]
    position_based_counts = [0 for _ in range(num_buckets)]
    for ex in data:
        input_sents = ex['input_sents']
        output_sents = ex['output_sents']

        for out_sent_idx, out_sent in enumerate(output_sents):
            bert_scores = ex['output_to_input_mappings'][out_sent_idx]
            for idx, (_, b) in enumerate(zip(input_sents, bert_scores)):
                bucket_id = int(float(idx) * num_buckets / float(len(input_sents)))
                position_based_scores[bucket_id] += b
                position_based_counts[bucket_id] += 1.

    for idx in range(len(position_based_scores)):
        position_based_scores[idx] = position_based_scores[idx] / position_based_counts[idx]

    return position_based_scores


    exit()
    buckets = [0 for _ in range(num_buckets)]
    min_val = 0
    max_val = 1

    for ex in data:
        input_sents = ex['input_sents']
        output_sents = ex['output_sents']

        for out_sent_idx, out_sent in enumerate(output_sents):
            input_sents_filtered = []
            input_sents_filtered_idx = []
            bert_scores = ex['output_to_input_mappings'][out_sent_idx]
            for idx, (inp, b) in enumerate(zip(input_sents, bert_scores)):
                if b > 0.89:
                    input_sents_filtered.append(inp)
                    input_sents_filtered_idx.append(idx)

            if len(input_sents_filtered) == 0:
                max_idx = np.argmax(bert_scores)
                input_sents_filtered.append(input_sents[max_idx])
                input_sents_filtered_idx.append(max_idx)


            input_sents_filtered_idx = set(input_sents_filtered_idx)
            positions = [float(x)/float(len(input_sents)) for x in input_sents_filtered_idx]

            for p in positions:
                val = min_val
                for idx in range(num_buckets):
                    val = val + (max_val - min_val) / float(num_buckets)
                    if p < val:
                        buckets[idx] += 1
                        break

    sum_buckets = np.sum(buckets)
    buckets = [float(b)/float(sum_buckets) for b in buckets]
    return buckets


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


def get_positionbias_file(file, num_buckets=10):
    lines = [line.strip() for line in file.readlines()]

    inputs = []
    references = []
    generations = []
    for i in range(0, len(lines), 4):
        inputs.append(lines[i])
        references.append(lines[i + 1])
        generations.append(lines[i + 2])

    bert_scores_references = get_bert_scores_all(inputs, references)
    buckets_references = get_position_bias(bert_scores_references, num_buckets)

    bert_scores_generated = get_bert_scores_all(inputs, generations)
    buckets_generated = get_position_bias(bert_scores_generated, num_buckets)

    print(buckets_references)
    print(buckets_generated)


if __name__ == '__main__':
    folder = '../../data/newsroom_old/mixed/20k/model-bart-multheads-4/head0'
    input_file = open(os.path.join(folder, 'dev_outfinal.txt'))
    graph = True
    gen_scores = get_positionbias_file(input_file)
