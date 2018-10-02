# NOTE: WHY NOT USE WORD2VEC?
# We chose not using word2vec or another pretrained embedding because the definition of
# "words" in our problem does not comprise punctuations.

import tensorflow as tf
import numpy as np
from gensim.models import KeyedVectors
import os, random, pickle


data_path = '/home/ashbylepoc/tmp/nlp_dev_1'
train_file_name = os.path.join(data_path, 'train-europarl-v7.fi-en.en')
word2vec_dir = '/home/ashbylepoc/tmp/word2vec/'
# model = KeyedVectors.load_word2vec_format(
#     '{}/GoogleNews-vectors-negative300.bin'.format(word2vec_dir),
#     binary=True)

def get_tokens_dict():
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
    return word_counts


def get_vector_representation(word, model):
    try:
        vector = model.get_vector(word)
        return vector
    except KeyError:
        return None

def get_max_n_tokens(lines):
    max_n_tokens = 0
    for line in lines:
        l = len(line.split(' '))
        if max_n_tokens < l:
            max_n_tokens = l
    return max_n_tokens




class DataReader(object):

    def __init__(self, file, token_dict):
        """

        :param file:
        :param token_dict:
            dictionary of all the tokens in
            our corpus; to create our final
            dict we add 2 "tokens": <UNKNOWN>,
            <UNKS>. Respectively, they are the
            len(token_dict)'th and len(token_dict)'th + 1
            elements of the dict
        """
        self.file = file
        self.token_dict = [t for t in token_dict] \
                          + ['<UNKNOWN>', '<UNK>']
        self.n_tokens_dict = len(self.token_dict)

        with open(self.file, 'r') as f:
            self.lines = f.readlines()
            self.n_lines = len(self.lines)

    def get_token_dict_pos(self, token):
        try:
            pos = self.token_dict.index(token)
        except ValueError:
            pos = -2
        return pos

    def make_mini_batch(self, lines, unk_p):
        """
        hide a % of the words, replace with special token,
        convert to one-hot vector and return the minibatch
        for X and Y. In our task:

        X = x1, ..., xn   Y = y1, ..., yc

        :param lines:
            lines to transform into mini batches
        :return:
            For x: Numpy array of shape (n_lines, max_n_words_per_sentence, n_tokens)
                    where n_tokens = # tokens in the whole dictionary
            For y: Numpy array of shape (n_lines, n_unks, n_
        """
        mini_batch_x = []
        mini_batch_y = []
        max_n_tokens = 0
        max_n_unks = 0
        n_lines = len(lines)
        for line in lines:

            tokens = line.split(' ')
            n_tokens = len(tokens)
            n_unks = int(n_tokens * unk_p)
            unks = [random.randint(0, n_tokens) for _ in range(n_unks)]
            sub_batch_x = np.zeros(shape=(n_tokens, self.n_tokens_dict))
            sub_batch_y = np.zeros(shape=(len(unks), self.n_tokens_dict))

            # j is an index for the unks that we have processed
            j = 0
            for i, token in enumerate(tokens):
                one_hot_x = np.zeros(shape=self.n_tokens_dict)
                if i in unks:
                    # Find the position of the token that we're
                    # removing and add the one hot vector to
                    # the mini batches
                    one_hot_y = np.zeros(shape=self.n_tokens_dict)
                    y_pos = self.get_token_dict_pos(token)

                    one_hot_y[y_pos] = 1
                    one_hot_x[-1] = 1
                    sub_batch_x[i, :] = one_hot_x
                    sub_batch_y[j, :] = one_hot_y
                    j += 1

                else:
                    x_pos = self.get_token_dict_pos(token)
                    one_hot_x[x_pos] = 1
                    sub_batch_x[i, :] = one_hot_x

            # Compute the maximum number of tokens because
            # at the end we want to pad the smaller sentences
            # (those with less tokens) with 0's
            if max_n_tokens < n_tokens:
                max_n_tokens = n_tokens
            if max_n_unks < n_unks:
                max_n_unks = n_unks

            # Add the sentence to the minibatch
            mini_batch_x.append(sub_batch_x)
            mini_batch_y.append(sub_batch_y)

        # Convert the mini_batch (list) to numpy arrays
        mini_batch_x_padded = np.zeros(shape=(n_lines,
                                            max_n_tokens,
                                            self.n_tokens_dict))
        mini_batch_y_padded = np.zeros(shape=(n_lines,
                                              max_n_unks,
                                              self.n_tokens_dict))
        # mini_batch_x_padded = np.pad(mini_batch_x, pad_width=  )






    def iterate_mini_batch(self, batch_size):
        n_samples = self.n_lines
        n_batch = int(n_samples / batch_size)

        for i in range(n_batch):



if __name__ == '__main__':
    # with open('tokens_dict.pickle', 'wb') as f:
    #     pickle.dump(word_counts, f)
    with open('tokens_dict.pickle', 'rb') as f:
        tokens_dict = pickle.load(f)