import os, re
# from lstm_1 import *
import numpy as np
import os, csv, pickle, time
import tensorflow as tf
from tensorflow.contrib import rnn
import sys

# data_path = '/home/ashbylepoc/tmp/nlp_dev_1'
data_path = "/run/media/ashbylepoc/ff112aea-f91a-4fc7-a80b-4f8fa50d41f3/tmp/data/nlp_dev_1/"
raw_data = os.path.join(data_path, 'en/train-europarl-v7.fi-en.en')
train_file_name = os.path.join(data_path, 'en/train_en_strutured.csv')
test_file_name = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u5.en')
shuffled_train_set = os.path.join(data_path, 'en/train_en_shuffled.csv')
shuffled_valid_set = os.path.join(data_path, 'en/valid_en_shuffled.csv')
word2vec_dir = '/home/ashbylepoc/tmp/word2vec/'
SAVE_PATH = data_path + 'checkpoints_hist_lstm'


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


# Structure test file
def structure_file(test_file_name, history_size):
    structured_unks = []
    lines = open(test_file_name, 'r').readlines()
    for j, line in enumerate(lines):
        tokens = tokenize_sentence(line)
        n_tokens = len(tokens)
        for i, token in enumerate(tokens):
            if '<unk w="' in token:
                if i >= history_size and n_tokens - i > history_size:
                    structured_unk = tokens[i - history_size: history_size + i + 1]
                    structured_unk[int((history_size * 2 + 1 ) /  2)] = '<unk>'
                    y = re.search('<unk w="(.*)"/>', token).group(1)
                    structured_unks.append([structured_unk, y])
                elif i < history_size and n_tokens - i > history_size:
                    start = max(i - history_size, 0)
                    end = min(history_size + i + 1, n_tokens)
                    middle = int((history_size * 2 + 1) / 2)
                    shift = middle - i
                    sub_tokens = tokens[start: end]
                    structured_unk = ['<s>' for _ in range(history_size * 2 + 1)]
                    for k, t in enumerate(sub_tokens):
                        structured_unk[k + shift] = t

                    structured_unk[middle] = '<unk>'
                    y = re.search('<unk w="(.*)"/>', token).group(1)
                    structured_unks.append([structured_unk, y])
                elif i >= history_size and n_tokens - i < history_size:
                    start = max(i - history_size, 0)
                    end = min(history_size + i + 1, n_tokens)
                    middle = int((history_size * 2 + 1) / 2)
                    shift = -(end - i)
                    sub_tokens = tokens[start: end]
                    structured_unk = ['<s>' for _ in range(history_size * 2 + 1)]
                    structured_unk[: end - i + history_size] = sub_tokens
                    structured_unk[middle] = '<unk>'
                    y = re.search('<unk w="(.*)"/>', token).group(1)
                    structured_unks.append([structured_unk, y])
    return structured_unks

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
    with open('tokens_dict.pickle', 'rb') as f:
        tokens_dict = pickle.load(f)
    print('building graph')
    # b_x, b_y, m = data_reader.make_mini_batch(data_reader.lines[:32])

    batch_size = 32
    # valid_batch_size = 1000
    rnn_size = 1000
    max_n_token_sentence = 100
    # max_n_token_dict = 10000 + 3
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
    # preds_reshaped = tf.reshape(preds, shape=[-1, max_n_token_sentence, max_n_token_dict])
    cost = - tf.reduce_sum(y * tf.log(tf.clip_by_value(preds, 1e-10, 1.0)), axis=1)
    cost = tf.reduce_mean(cost, axis=0)
    predictions = tf.cast(tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1)), dtype='float32')
    acc = tf.reduce_mean(predictions)

    history_size = 4
    structured_unks = structure_file(test_file_name, history_size)
    n_batch = len(structured_unks) / batch_size
    all_acc = []
    for i in range(n_batch):
        bb_x = []
        yy = []
        for unk in structured_unks[i * batch_size: (i + 1) * batch_size]:
            bb_x.append(unk[0])
            yy.append(unk[1])

        b_x = np.zeros(shape=[batch_size, history_size * 2 + 1, 10003])
        b_y = np.zeros(shape=[batch_size, 10003])
        for j, line in enumerate(bb_x):
            y_pos = get_token_dict_pos(token_dict, yy[j])
            b_y[j][y_pos] = 1.
            for k, token in enumerate(line):
                x_pos = get_token_dict_pos(token_dict, token)
                b_x[j][k][x_pos] = 1.

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, '{}/lstm_{}/lstm'.format(SAVE_PATH, 5000))
            preds_, acc_ = sess.run([preds, acc], feed_dict={x: b_x, y: b_y})
            all_acc.append(acc_)