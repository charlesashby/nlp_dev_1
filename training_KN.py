import os, pickle
from nltk.util import ngrams
import collections
from kneser_ney.kneser_ney import *

data_path = '/run/media/ashbylepoc/ff112aea-f91a-4fc7-a80b-4f8fa50d41f3/tmp/data/nlp_dev_1'
train_file_name = os.path.join(data_path, 'en/train-europarl-v7.fi-en.en')


if __name__ == '__main__':
    with open(train_file_name, 'r') as f:
        lines = f.readlines()[:10000]
        trigrams = [ngram for line in lines for ngram in
                    ngrams(line.split(' '), 3, pad_left=True, pad_right=True, left_pad_symbol='<s>',
                           right_pad_symbol='<s>')]

    # with open('trigrams.pickle', 'wb') as f:
    #     pickle.dump(trigrams, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('/run/media/ashbylepoc/ff112aea-f91a-4fc7-a80b-4f8fa50d41f3/tmp/data/nlp_dev_1trigrams.pickle', 'rb') as f:
    #     trigrams = pickle.load(f)

    lm = KneserNeyLM(3, trigrams, end_pad_symbol='<s>')