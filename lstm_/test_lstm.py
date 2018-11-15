import os, re, unicodedata
# from lstm_1 import *
import numpy as np
import os, csv, pickle, time
import tensorflow as tf
from tensorflow.contrib import rnn
# from lstm.lstm_1 import *
import sys

# data_path = '/home/ashbylepoc/tmp/nlp_dev_1'
data_path = "/run/media/ashbylepoc/ff112aea-f91a-4fc7-a80b-4f8fa50d41f3/tmp/data/nlp_dev_1/"
# raw_data = os.path.join(data_path, 'en/train-europarl-v7.fi-en.en')
# train_file_name = os.path.join(data_path, 'en/train_en_strutured.csv')
# test_file_name = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u10.en')
shuffled_p_train_set = os.path.join(data_path, 'en/trainp_en_shuffled.csv')
shuffled_p_valid_set = os.path.join(data_path, 'en/validp_en_shuffled.csv')
# word2vec_dir = '/home/ashbylepoc/tmp/word2vec/'
SAVE_PATH = data_path + 'checkpoints_hist_lstm'


# padded clean 8 - 4
shuffled_pc8_4_train_set = os.path.join(data_path, 'en/trainpc8-4_en_shuffled.csv')
shuffled_pc8_4_valid_set = os.path.join(data_path, 'en/validpc8-4_en_shuffled.csv')


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

def strip_accents(text):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def clean_line(line):
    cleaned_tokens = []
    for token in line.split(' '):
        try:
            if is_number(token):
                cleaned_tokens.append('<NUMBER>')
            elif '.\n' in token and token != '.\n':
                cleaned_tokens.append(token.replace('.\n', '').lower())
            else:
                cleaned_tokens.append(strip_accents(token).lower())
        except UnicodeDecodeError:
            pass
    cleaned_line = ' '.join([t for t in cleaned_tokens])
    return cleaned_line


def structure_file(file, suf=3, pre=5, clean=False):
    structured_unks = []
    lines = open(file, 'r').readlines()

    # clean lines...

    for j, line in enumerate(lines):
        if clean:
            cleaned_line = clean_line(line)
        else:
            cleaned_line = line
        tokens = tokenize_sentence(cleaned_line)
        n_tokens = len(tokens)
        for i, token in enumerate(tokens):
            if '<unk w="' in token:
                end = min(n_tokens, i + suf)
                start = max(0, i - pre)
                history = tokens[start:end + 1]
                if len(history) == pre + suf + 1:
                    structured_unk = history
                elif i + suf >= n_tokens:
                    structured_unk = ['<s>' for _ in range(pre + suf + 1)]
                    structured_unk[:len(history)] = history
                elif i - pre < 0:
                    structured_unk = ['<s>' for _ in range(pre + suf + 1)]
                    structured_unk[-len(history):] = history
                else:
                    print('hello')
                    # return NotImplementedError
                structured_unk[pre] = '<UNK>'
                y = re.search('<unk w="(.*)"/>', token).group(1)
                structured_unks.append([structured_unk, y])
    return structured_unks


def clean_structured_unks(structured_unks):
    structured_unks_clean = []
    for unk, unk_y in structured_unks:
        ok = True
        for t in unk:
            if '<unk w="' in t:
                ok = False
                break
        if ok:
            structured_unks_clean.append([unk, unk_y])
    return structured_unks_clean


def test_file(sess, perc_unk=10):
    test_file = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u{}.en'.format(perc_unk))
    structured_unks = structure_file(test_file, suf=3, pre=5, clean=True)
    # Remove sentences with unks...
    # structured_unks = clean_structured_unks(structured_unks)
    n_batch = len(structured_unks) / batch_size
    all_acc = []
    for i in range(n_batch):
        bb_x = []
        yy = []
        for unk in structured_unks[i * batch_size: (i + 1) * batch_size]:
            bb_x.append(unk[0])
            yy.append(unk[1])

        b_x = np.zeros(shape=[batch_size, pre + suf + 1, 10003])
        b_y = np.zeros(shape=[batch_size, 10003])
        for j, line in enumerate(bb_x):
            y_pos = get_token_dict_pos(token_dict, yy[j])
            b_y[j][y_pos] = 1.
            for k, token in enumerate(line):
                if '<unk w=' in token:
                    x_pos = -2
                else:
                    x_pos = get_token_dict_pos(token_dict, token)
                b_x[j][k][x_pos] = 1.

        # data_reader = DataReader(shuffled_p_train_set, shuffled_p_valid_set, 4, tokens_dict, 100, 10000)

        # it = enumerate(data_reader.iterate_mini_batch(32, dataset='valid', pre=pre, suf=suf))
        # ii, (bb_x, bb_y) = next(it)
        preds_, acc_ = sess.run([preds, acc], feed_dict={x: b_x, y: b_y})
        all_acc.append(acc_)
    print('Acc: {} file: {}'.format(np.mean(all_acc), test_file))
    return np.mean(all_acc)

def get_token_dict_pos(token_dict, token):
    if '<unk w="' in token:
        pos = -1
        return pos
    elif token == '<s>':
        pos = -3
        return pos
    try:
        pos = token_dict.index(token)
    except ValueError:
        # print('not found: {}'.format(token))
        pos = -2
    return pos


if __name__ == '__main__':
    # with open('tokens_dict.pickle', 'rb') as f:
    #     tokens_dict = pickle.load(f)
    with open('tokens_dict.pickle', 'rb') as f:
        tokens_dict = pickle.load(f)

    print('building graph')
    # b_x, b_y, m = data_reader.make_mini_batch(data_reader.lines[:32])

    batch_size = 32
    rnn_size = 1024
    learning_rate = 0.001
    sorted_tokens = sorted(tokens_dict.items(), key=lambda item: item[1])
    token_dict = [t[0] for t in sorted_tokens[-10000:]] + ['<s>', '<UNKNOWN>', '<UNK>']
    n_tokens_dict = len(token_dict)
    # tt = data_reader.iterate_mini_batch(batch_size)
    # t = next(tt)
    max_n_token_dict = 10000 + 3

    # GRAPH
    x = tf.placeholder('float32', shape=[None, None, max_n_token_dict])
    y = tf.placeholder('float32', shape=[None, max_n_token_dict])
    cell = rnn.LSTMCell(rnn_size, state_is_tuple=True, forget_bias=0.0, reuse=False)
    initial_rnn_state = cell.zero_state(batch_size, dtype='float32')
    outputs, final_rnn_state = tf.nn.dynamic_rnn(cell, x,
                                                 initial_state=initial_rnn_state,
                                                 dtype='float32')
    outputs = tf.transpose(outputs, [1, 0, 2])
    last = outputs[-1]
    outputs_reshape = tf.reshape(last, shape=[-1, rnn_size])
    w = tf.get_variable("w", [rnn_size, max_n_token_dict], dtype='float32')
    b = tf.get_variable("b", [max_n_token_dict], dtype='float32')
    preds = tf.nn.softmax(tf.matmul(outputs_reshape, w) + b)
    cost = - tf.reduce_sum(y * tf.log(tf.clip_by_value(preds, 1e-10, 1.0)), axis=1)
    cost = tf.reduce_mean(cost, axis=0)
    predictions = tf.cast(tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1)), dtype='float32')
    acc = tf.reduce_mean(predictions)

    saver = tf.train.Saver()
    pre = 8
    suf = 4
    with tf.Session() as sess:
        # saver.restore(sess, '{}/lstmp_{}_{}/lstm'.format(SAVE_PATH, 35000, 1))
        saver.restore(sess, '{}/lstmpc10kn8-4_{}_{}/lstm'.format(SAVE_PATH, 30000, 1))

        for m in [5, 10, 20, 30, 40]:
            test_file = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u{}.en'.format(m))
            structured_unks = structure_file(test_file, suf=4, pre=8, clean=True)
            # Remove sentences with unks...
            # structured_unks = clean_structured_unks(structured_unks)
            n_batch = len(structured_unks) / batch_size
            all_acc = []
            for i in range(n_batch):
                bb_x = []
                yy = []
                for unk in structured_unks[i * batch_size: (i + 1) * batch_size]:
                    bb_x.append(unk[0])
                    yy.append(unk[1])

                b_x = np.zeros(shape=[batch_size, pre + suf + 1, 10003])
                b_y = np.zeros(shape=[batch_size, 10003])
                for j, line in enumerate(bb_x):
                    y_pos = get_token_dict_pos(token_dict, yy[j])
                    b_y[j][y_pos] = 1.
                    for k, token in enumerate(line):
                        if '<unk w=' in token:
                            x_pos = -2
                        else:
                            x_pos = get_token_dict_pos(token_dict, token)
                        b_x[j][k][x_pos] = 1.

                # data_reader = DataReader(shuffled_p_train_set, shuffled_p_valid_set, 4, tokens_dict, 100, 10000)

                # it = enumerate(data_reader.iterate_mini_batch(32, dataset='valid', pre=pre, suf=suf))
                # ii, (bb_x, bb_y) = next(it)
                preds_, acc_ = sess.run([preds, acc], feed_dict={x: b_x, y: b_y})
                all_acc.append(acc_)
            print('Acc: {} file: {}'.format(np.mean(all_acc), test_file))