# NOTE: WHY NOT USE WORD2VEC?
# We chose not using word2vec or another pretrained embedding because the definition of
# "words" in our problem does not comprise punctuations.

from util.ops import conv2d, MLP
from tensorflow.contrib import rnn
import tensorflow as tf
import numpy as np
from gensim.models import KeyedVectors
import os, random, pickle, threading
import time

data_path = '/home/ashbylepoc/tmp/nlp_dev_1'
train_file_name = os.path.join(data_path, 'train-europarl-v7.fi-en.en')
# word2vec_dir = '/home/ashbylepoc/tmp/word2vec/'
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

    def worker(self):

        while self.batch_lines:
            line = self.batch_lines.pop()
            tokens = line.split(' ')
            n_tokens = len(tokens)
            n_unks = int(n_tokens * self.unk_p)
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
            self.mini_batch_x.append(sub_batch_x)
            self.mini_batch_y.append(sub_batch_y)
            if self.max_n_tokens < n_tokens:
                self.max_n_tokens = n_tokens
            if self.max_n_unks < n_unks:
                self.max_n_unks = n_unks

    def parallel_make_mini_batch(self, lines, unk_p, n_thread):
        self.unk_p = unk_p
        self.batch_lines = lines
        self.mini_batch_x = []
        self.mini_batch_y = []
        self.max_n_tokens = 0
        self.max_n_unks = 0
        n_lines = len(lines)

        print('3: {}'.format(time.time()))
        # for i in range(n_thread):
        #     t = threading.Thread(target=self.worker)
        #     t.start()
        t1 = threading.Thread(target=self.worker)
        t2 = threading.Thread(target=self.worker)
        t3 = threading.Thread(target=self.worker)

        # Start all threads
        t1.start()
        t2.start()
        t3.start()

        # Wait until all thread are dead
        t1.join()
        t2.join()
        t3.join()

        print('4: {}'.format(time.time()))
        # Convert the mini_batch (list) to numpy arrays
        mini_batch_x_padded = np.zeros(shape=(n_lines,
                                            self.max_n_tokens,
                                            self.n_tokens_dict))
        mini_batch_y_padded = np.zeros(shape=(n_lines,
                                              self.max_n_unks,
                                              self.n_tokens_dict))
        print('5: {}'.format(time.time()))
        for i in range(n_lines):
            mini_batch_x_padded[i, :len(self.mini_batch_x[i]), :] = self.mini_batch_x[i]
            mini_batch_y_padded[i, :len(self.mini_batch_y[i]), :] = self.mini_batch_y[i]
        print('6: {}'.format(time.time()))
        import pdb; pdb.set_trace()
        def numpy_fillna(data):
            # Get lengths of each row of data
            lens = np.array([len(i) for i in data])

            # Mask of valid places in each row
            mask = np.arange(lens.max()) < lens[:, None]

            # Setup output array and put elements from data into masked positions
            mask_shape = [s for s in mask.shape]
            mask_shape.append(self.n_tokens_dict)
            out = np.zeros(shape=mask_shape,
                           dtype='float32')

            out[mask] = np.concatenate(data)
            return out

        return mini_batch_x_padded, mini_batch_y_padded

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
            For y: Numpy array of shape (n_lines, n_unks, n_tokens)
        """
        mini_batch_x = []
        mini_batch_y = []
        mini_batch_unk_pos = []
        max_n_tokens = 0
        max_n_unks = 0
        n_lines = len(lines)

        for line in lines:

            tokens = line.split(' ')
            n_tokens = len(tokens)
            n_unks = int(n_tokens * unk_p)
            unks = [random.randint(0, n_tokens - 1) for _ in range(n_unks)]
            sub_batch_x = np.zeros(shape=(n_tokens, self.n_tokens_dict))
            sub_batch_y = np.zeros(shape=(n_tokens, self.n_tokens_dict))

            # Keep the position of the unks
            sub_batch_unk_pos = np.zeros(shape=n_tokens)
            sub_batch_unk_pos[unks] = 1.
            # j is an index for the unks that we have processed
            # j = 0
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
                    sub_batch_y[i, :] = one_hot_y
                    # j += 1

                else:
                    # one_hot_y = np.zeros(shape=self.n_tokens_dict)
                    # y_pos = self.get_token_dict_pos(token)
                    x_pos = self.get_token_dict_pos(token)
                    one_hot_x[x_pos] = 1
                    sub_batch_x[i, :] = one_hot_x
                    sub_batch_y[i, :] = one_hot_x


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
            mini_batch_unk_pos.append(sub_batch_unk_pos)
        # Convert the mini_batch (list) to numpy arrays
        mini_batch_x_padded = np.zeros(shape=(n_lines,
                                            max_n_tokens,
                                            self.n_tokens_dict))
        mini_batch_y_padded = np.zeros(shape=(n_lines,
                                              max_n_tokens,
                                              self.n_tokens_dict))
        mini_batch_unk_pos_padded = np.zeros(shape=(n_lines, max_n_tokens))

        for i in range(n_lines):
            mini_batch_x_padded[i, :len(mini_batch_x[i]), :] = mini_batch_x[i]
            mini_batch_y_padded[i, :len(mini_batch_y[i]), :] = mini_batch_y[i]
            mini_batch_unk_pos_padded[i, :len(mini_batch_unk_pos[i])] = mini_batch_unk_pos[i]

        return mini_batch_x_padded, mini_batch_y_padded, mini_batch_unk_pos_padded

    def iterate_mini_batch(self, batch_size, unk_p):
        n_samples = self.n_lines
        n_batch = int(n_samples / batch_size)

        for i in range(n_batch):
            inputs, targets, unk_pos = self.make_mini_batch(
                self.lines[i * batch_size: (i + 1) * batch_size], unk_p)
            yield inputs, targets



class LSTM(object):
    def __init__(self, tokens_dict, data_reader, batch_size):
        self.token_dict = [t for t in tokens_dict] \
                          + ['<UNKNOWN>', '<UNK>']
        self.n_tokens_dict = len(self.token_dict)
        self.data_reader = data_reader
        self.batch_size = batch_size
        self.rnn_size = 100
        self.x = tf.placeholder('float32', shape=[None, None, self.n_tokens_dict])
        self.y = tf.placeholder('float32', shape=[None, None, self.n_tokens_dict])
        self.unk_pos = tf.placeholder('float32', )

    def build(self):
        cell = rnn.LSTMCell(self.rnn_size, state_is_tuple=True, forget_bias=0.0, reuse=False)
        initial_rnn_state = cell.zero_state(self.batch_size, dtype='float32')
        outputs, final_rnn_state = tf.nn.dynamic_rnn(cell, self.x, initial_state=initial_rnn_state,
                                                     dtype='float32')






if __name__ == '__main__':
    # with open('tokens_dict.pickle', 'wb') as f:
    #     pickle.dump(word_counts, f)
    with open('tokens_dict.pickle', 'rb') as f:
        tokens_dict = pickle.load(f)

    data_reader = DataReader(train_file_name, tokens_dict)
    b_x, b_y, unk_pos = data_reader.make_mini_batch(data_reader.lines[:32], unk_p=0.1)
    batch_size = 32
    rnn_size = 100
    token_dict = [t for t in tokens_dict] + ['<UNKNOWN>', '<UNK>']
    n_tokens_dict = len(token_dict)
    u_pos = tf.placeholder('float32', shape=[None, None])
    y = tf.placeholder('float32', shape=[None, None, n_tokens_dict])
    x = tf.placeholder('float32', shape=[None, None, n_tokens_dict])
    cell = rnn.LSTMCell(rnn_size, state_is_tuple=True, forget_bias=0.0, reuse=False)
    initial_rnn_state = cell.zero_state(batch_size, dtype='float32')
    outputs, final_rnn_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_rnn_state,
                                                 dtype='float32')


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        o, f = sess.run([outputs, final_rnn_state, u_pos, y], feed_dict={x: b_x,
                                                                           u_pos: unk_pos,
                                                                           y: b_y})