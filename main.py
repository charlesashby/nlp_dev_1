# TODO: ROADMAP
# 1. Fill unk with most occuring word
# 1.1 VIZ:
#       - Visualize accuracy vs length of the sentence
#       - Visualize total accuracy on the different test sets
#       - Visualize accuracy on the non-contiguous test sets
# 2. Fill contiguous unks with most occuring n-gram
#       - Visualize accuracy vs length of the sentence
#       - Visualize total accuracy on the different test sets
#       - Visualize accuracy on the non-contiguous test sets

import os, pickle
from nltk.util import ngrams
import collections

data_path = '/run/media/ashbylepoc/ff112aea-f91a-4fc7-a80b-4f8fa50d41f3/tmp/data/nlp_dev_1'
train_file_name = os.path.join(data_path, 'en/train-europarl-v7.fi-en.en')
# test_5 = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u5c.en')
# test_10 = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u10c.en')
# test_20 = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u20c.en')
# test_30 = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u30c.en')
# test_40 = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u40c.en')

# Benchmark
word_counts = {}
with open(train_file_name, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if i % 1000 == 0:
            print('tokenizing line: {}/1000000'.format(i))
        tokens = line.split(' ')
        for token in tokens:
            if token in word_counts:
                word_counts[token] += 1
            else:
                word_counts[token] = 1

# Sort tokens
sorted_tokens = sorted(word_counts.items(), key=lambda item: item[1])

# top words = ['the' - 1962491, ',' - 1508126, 'of' - 1002208, '.' - 961118, 'to' - 933167]
# let's try to replace all missing words with 'the' and compute our accuracy
# for different test sets
test_sets = [os.path.join(data_path,
                          'en/unk-europarl-v7.fi-en-u{}c.en'.format(i))
             for i in [5, 10, 20, 30, 40]]
test_sets_non_contiguous = [os.path.join(data_path,
                          'en/unk-europarl-v7.fi-en-u{}.en'.format(i))
             for i in [5, 10, 20, 30, 40]] + test_sets
with open('results_benchmark_clean.txt', 'a') as f:
    for test_set in test_sets_non_contiguous:
        print('Processing test set: {}'.format(test_sets_non_contiguous))
        lines_test_set = open(test_set, 'r').readlines()
        results = []
        for line in lines_test_set:
            tokens = line.split(' ')
            n_tokens = len(tokens)
            n_unk = 0
            n_true = 0

            # Merge unk's together
            tokens_clean = []
            for i, _ in enumerate(tokens):
                if tokens[i] == '<unk':
                    tokens_clean.append('{} {}'.format(tokens[i], tokens[i+1]))
                elif 'w="' == tokens[i][:3]:
                    pass
                else:
                    tokens_clean.append(tokens[i])

            for token in tokens_clean:
                if '<unk w="' in token:
                    n_unk += 1
                    if '<unk w="the"/>' == token:
                        n_true += 1
            if n_unk > 0:
                accuracy = float(n_true) / float(n_unk)
            else:
                accuracy = 'NA'
            results.append([n_tokens, n_unk, n_true, accuracy])

        # print results
        f.write('Results for test set: {}\n'.format(test_set))
        total_tokens = 0
        total_unk = 0
        total_true = 0
        for result in results:
            # f.write('-'.join(str(r) for r in result))
            # f.write('\n')
            total_tokens += result[0]
            total_unk += result[1]
            total_true += result[2]
            total_accuracy = float(total_true) / float(total_unk)
        f.write('total tokens: {} - total unk: {} - total true: {} - total acc: {}\n'.format(total_tokens,
                                                                           total_unk,
                                                                           total_true, total_accuracy))


# Build an n-gram dictionary
with open(train_file_name, 'r') as f:
    lines = f.readlines()[:10000]
    trigrams = [ngram for line in lines for ngram in ngrams(line.split(' '), 3, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='<s>')]

# with open('trigrams.pickle', 'wb') as f:
#     pickle.dump(trigrams, f, protocol=pickle.HIGHEST_PROTOCOL)
# with open('/run/media/ashbylepoc/ff112aea-f91a-4fc7-a80b-4f8fa50d41f3/tmp/data/nlp_dev_1trigrams.pickle', 'rb') as f:
#     trigrams = pickle.load(f)

lm = KneserNeyLM(3, trigrams, end_pad_symbol='<s>')
