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

# data_path = '/home/ashbylepoc/tmp/nlp_dev_1'
data_path = "/run/media/ashbylepoc/ff112aea-f91a-4fc7-a80b-4f8fa50d41f3/tmp/data/nlp_dev_1"
train_file_name = os.path.join(data_path, 'en/train-europarl-v7.fi-en.en')
word2vec_dir = '/home/ashbylepoc/tmp/word2vec/'
SAVE_PATH = data_path + 'checkpoints/lstm_1'



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


def tokenize_sentence(sentence):
    tokens = sentence.split(' ')
    tokens_clean = []
    for i, _ in enumerate(tokens):
        if tokens[i] == '<unk':
            tokens_clean.append('{} {}'.format(tokens[i], tokens[i + 1]))
        elif 'w="' == tokens[i][:3]:
            pass
        else:
            tokens_clean.append(tokens[i])
    return tokens_clean

def get_max_n_tokens(lines):
    max_n_tokens = 0
    for line in lines:
        l = len(line.split(' '))
        if max_n_tokens < l:
            max_n_tokens = l
    return max_n_tokens


class DataReader(object):

    def __init__(self, file, token_dict, max_n_tokens_dict, max_n_tokens_sentence):
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
        self.max_n_tokens_sentence = max_n_tokens_sentence
        sorted_tokens = sorted(token_dict.items(), key=lambda item: item[1])
        self.token_dict = [t[0] for t in sorted_tokens[-max_n_tokens_dict:]] + ['</s>','<UNKNOWN>', '<UNK>']
        # self.token_dict = [t for t in token_dict] + ['<UNKNOWN>', '<UNK>']
        self.n_tokens_dict = len(self.token_dict)

        with open(self.file, 'r') as f:
            self.lines = f.readlines()
            self.n_lines = len(self.lines)

    def get_token_dict_pos(self, token):
        if '<unk w="' in token:
            pos = -1
            return pos
        try:
            pos = self.token_dict.index(token)
        except ValueError:
            pos = -2
        return pos

    def line_to_one_hot_inference(self, line):
        """
        Return the one hot version of the sentence,

        :param line:
        :return:
        """
        tokens = tokenize_sentence(line)
        n_tokens = len(tokens)


    def make_mini_batch(self, lines):
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
        n_lines = len(lines)
        mini_batch_x = np.zeros(shape=[n_lines,
                                       self.max_n_tokens_sentence,
                                       self.n_tokens_dict])
        mini_batch_y = np.zeros(shape=[n_lines,
                                       self.max_n_tokens_sentence,
                                       self.n_tokens_dict])
        mini_batch_mask_end_sentence = np.zeros(shape=[n_lines,
                                                       self.max_n_tokens_sentence])

        for l, line in enumerate(lines):
            tokens = tokenize_sentence(line)
            tokens = tokens[:self.max_n_tokens_sentence]
            n_tokens = len(tokens)
            tokens += ['</s>' for _ in range(self.max_n_tokens_sentence - n_tokens - 1)]
            for i, token in enumerate(tokens[:-1]):
                token_pos = self.get_token_dict_pos(token)
                token_pos_target = self.get_token_dict_pos(tokens[i + 1])
                mini_batch_x[l][i][token_pos] = 1.
                mini_batch_y[l][i][token_pos_target] = 1.
                mini_batch_mask_end_sentence[l, i] = 1.
        return mini_batch_x, mini_batch_y, mini_batch_mask_end_sentence

    def iterate_mini_batch(self, batch_size, unk_p):
        n_samples = self.n_lines
        n_batch = int(n_samples / batch_size)

        for i in range(n_batch):
            inputs, targets, mask = self.make_mini_batch(
                self.lines[i * batch_size: (i + 1) * batch_size])
            yield inputs, targets, mask


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
#     model = KeyedVectors.load_word2vec_format(
#         '{}/GoogleNews-vectors-negative300.bin'.format(word2vec_dir),
#         binary=True)

    from lstm.lstm import *

    with open('tokens_dict.pickle', 'rb') as f:
        tokens_dict = pickle.load(f)
    print('building graph')
    # Visualize tokens distribution
    sorted_tokens = sorted(tokens_dict.items(), key=lambda item: item[1])

    # b_x, b_y, m = data_reader.make_mini_batch(data_reader.lines[:32])
    batch_size = 32
    rnn_size = 1000
    max_n_token_sentence = 100
    max_n_token_dict = 4000 + 3
    learning_rate = 0.01
    data_reader = DataReader(train_file_name, tokens_dict,
                             max_n_tokens_dict=4000,
                             max_n_tokens_sentence=max_n_token_sentence)
    token_dict = [t for t in tokens_dict][-4000:] + ['</s>', '<UNKNOWN>', '<UNK>']
    n_tokens_dict = len(token_dict)
    masks = tf.placeholder('float32', shape=[None, None])
    x = tf.placeholder('float32', shape=[None, None, max_n_token_dict])
    y = tf.placeholder('float32', shape=[None, None, max_n_token_dict])
    cell = rnn.LSTMCell(rnn_size, state_is_tuple=True, forget_bias=0.0, reuse=False)
    initial_rnn_state = cell.zero_state(batch_size, dtype='float32')
    outputs, final_rnn_state = tf.nn.dynamic_rnn(cell, x,
                                                 initial_state=initial_rnn_state,
                                                 dtype='float32')
    outputs_reshape = tf.reshape(outputs, shape=[-1, rnn_size])
    w = tf.get_variable("w", [rnn_size, max_n_token_dict], dtype='float32')
    b = tf.get_variable("b", [max_n_token_dict], dtype='float32')
    preds = tf.nn.softmax(tf.matmul(outputs_reshape, w) + b)
    # preds_reshaped = tf.reshape(preds, shape=[-1, max_n_token_sentence, max_n_token_dict])

    # compute loss
    y_reshaped = tf.reshape(y, shape=[-1, max_n_token_dict])
    # cross_entropy = y_reshaped * tf.log(tf.clip_by_value(preds, 1e-10, 1.0))

    # Remove EOS tokens from cross entropy loss
    # mask_reshaped = tf.reshape(masks, shape=[-1, max_n_token_dict])
    # cross_entropy_masked = - tf.reduce_sum(mask_reshaped * cross_entropy)
    # cost = cross_entropy_masked

    cost = - tf.reduce_sum(y_reshaped * tf.log(tf.clip_by_value(preds, 1e-10, 1.0)), axis=1)
    reshaped_mask = tf.reshape(masks, shape=(-1,))
    cost *= reshaped_mask
    cost = tf.reduce_sum(cost)
    # masks_reshaped = tf.reshape(masks, shape=[-1, max_n_token_dict])
    predictions = tf.cast(tf.equal(tf.argmax(preds, 1), tf.argmax(y_reshaped, 1)), dtype='float32')
    predictions *= reshaped_mask
    n_accurate_preds = tf.reduce_sum(predictions)
    # non_masked_predictions = tf.reshape(masks, shape=(-1,)) * tf.cast(predictions, dtype='float32')
    # predictions_reshaped = tf.reshape(predictions, shape=[-1, max_n_token_sentence])

    # predictions_sentence_wise = [tf.equal(tf.argmax(preds_reshaped[i]),
    #                                       tf.argmax(y[i])) for i in range(batch_size)]

    # acc = tf.reduce_mean(tf.cast(predictions, 'float32'))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    print('Done building graph ')
    print('training...')
    saver = tf.train.Saver()
    best_acc = 0.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        n_batch = int(700000 / 32)
        for j in range(100):
            # print('ok')
            for i in range(n_batch):
                b_x, b_y, m = data_reader.make_mini_batch(data_reader.lines[i * 32:(i + 1) * 32])

                _, c, n_accurate_preds_ = sess.run([optimizer, cost, n_accurate_preds],
                                                feed_dict={x: b_x, y: b_y, masks: m})

                print('TRAIN: iteration: {} - acc: {} - loss: {}'.format(i, n_accurate_preds_, c))
                if i % 100 == 0:
                    valid_acc = []
                    for k in range(10):
                        k += 1
                        # compute accuracy on validation set
                        bb_x, bb_y, mm = data_reader.make_mini_batch(data_reader.lines[-(k + 1) * 32:-k * 32])
                        cc, predictions__, n_accurate_preds__ = sess.run([cost, predictions, n_accurate_preds],
                                                    feed_dict={x: bb_x, y: bb_y, masks: mm})
                        n_pred_ok_ = 0
                        n_pred_ = 0
                        reshaped_predictions__ = predictions__.reshape(batch_size, max_n_token_sentence)
                        reshaped_maskk = mm.reshape((-1))
                        # reshaped_predictions__ = predictions__.reshape((-1,))

                        aa = n_accurate_preds__
                        valid_acc.append(aa)
                    mean_acc = np.mean(valid_acc)
                    if mean_acc > best_acc:
                        best_acc = mean_acc
                        save_path = saver.save(sess, SAVE_PATH)
                        print('saving model')
                    print('VALID: iteration: {} - acc: {} -- last_pred:'.format(i, mean_acc))


"""


    with tf.Session() as sess:
        saver.restore(sess, SAVE_PATH)
        b_x, b_y, m = data_reader.make_mini_batch(data_reader.lines[-32:])
        _, c, p, mm, predictions_, non_masked_predictions_ = sess.run([optimizer, cost,
         preds, masks, predictions, non_masked_predictions],
                                  feed_dict={x: b_x, y: b_y, masks: m})

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            # b_x, b_y = data_reader.make_mini_batch(data_reader.lines[i * 32:(i + 1) * 32])
            _, a, c, p, pp, m, preds_, mask_reshaped_ = sess.run([optimizer, acc, cost,
                                          predictions_sentence_wise,
                                          preds_reshaped, masks,
                                          preds, mask_reshaped],
                               feed_dict={x: b_x, y: b_y, masks: m})
            print('TRAIN: iteration: {} - acc: {} - loss: {}'.format(i, a, c))
"""


    # Test time inference
    # https://stackoverflow.com/questions/42440565/how-to-feed-back-rnn-output-to-input-in-tensorflow

