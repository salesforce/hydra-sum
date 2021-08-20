import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')


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

    return {'input': input, 'gold': gold, 'tokens': token_level}


def read_file(input_file):
    examples = []
    while True:
        ex = read_next(input_file)
        if ex is not None:
            examples.append(ex)
        else:
            break

    return examples


def get_prefix(toks, idx, ngram):

    if not toks[idx][0].startswith('Ġ'):  # don't do this analysis for these words
        return None

    full_token = [toks[idx][0]]
    for t_idx in range(idx + 1, len(toks)):
        if toks[t_idx][0].startswith('Ġ'):
            break
        else:
            next_part = toks[t_idx][0]
            if next_part in ['.', ',', '!', '</s>']:
                break
            full_token.append(next_part)

    full_token = ''.join(full_token)[1:]

    tokens = [full_token]

    incomplete_token = ''
    for t_idx in range(idx-1, 0, -1):
        if len(tokens) >= ngram:
            break
        if toks[t_idx][0].startswith('Ġ'):
            full_token = toks[t_idx][0][1:] + incomplete_token
            tokens.append(full_token)
            incomplete_token = ''
        else:
            incomplete_token += toks[t_idx][0]

    if len(tokens) == ngram:
        tokens.reverse()
        tokens = ' '.join(tokens)
        return tokens
    else:
        return None


def is_match(prefix, input):
    if prefix in input:
        return True
    else:
        return False


def get_likelihood_copybehavior(head0_exs, head1_exs, baseline_exs, ngram=2):
    ll0, ll1, llb = [], [], []
    for h0, h1, hb in zip(head0_exs, head1_exs, baseline_exs):
        assert h0['gold'] == h1['gold'] == hb['gold'], 'y wont you work'
        input = h0['input']
        gold = h0['gold']

        for idx, (t0, t1, tb) in enumerate(zip(h0['tokens'], h1['tokens'], hb['tokens'])):
            assert t0[0] == t1[0] == tb[0], 'tokens dont match, error in reading file maybe'
            tok = t0[0]
            if tok in ['<s>', '</s>']:
                continue

            prefix_ngram = get_prefix(h0['tokens'], idx, ngram)
            if prefix_ngram is None:
                continue
            copied = is_match(prefix_ngram, input)
            if copied:
                ll0.append(t0[1])
                ll1.append(t1[1])
                llb.append(tb[1])

    ll0 = np.mean(ll0)
    ll1 = np.mean(ll1)
    llb = np.mean(llb)

    return ll0, ll1, llb


if __name__=='__main__':

    folder = '../../data/newsroom/mixed/20k/model-bart-multheads-4/'
    print(folder)

    input_file = open(os.path.join(folder, 'head0', 'prob_outfinal.txt'))
    head0_exs = read_file(input_file)

    input_file = open(os.path.join(folder, 'head1', 'prob_outfinal.txt'))
    head1_exs = read_file(input_file)

    input_file = open(os.path.join(folder, 'prob_outfinal.txt'))
    baseline_exs = read_file(input_file)

    for ngram in range(2, 6):
        ll0, ll1, llb = get_likelihood_copybehavior(head0_exs, head1_exs, baseline_exs, ngram)
        print(f'likelihood w/ prefix of length {ngram} copied: baseline={llb:.2f}, head0={ll0:.2f}, head1={ll1:.2f}')