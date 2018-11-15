import numpy as np
import os, csv, pickle, time
from util.utils import *
import tensorflow as tf
from tensorflow.contrib import rnn

data_path = "/run/media/ashbylepoc/b79b0a3e-a5b9-41ed-987f-8fa4bdb6b2e6/tmp/data/nlp_dev_1/"
SAVE_PATH = os.path.join(data_path, 'checkpoints_hist_lstm')
file_names = ['tps/tp1/tmp.2pvHHmkcyH/test/en/unk-healthcan-20.en',
              'tps/tp1/tmp.2pvHHmkcyH/test/en/unk-hans-20.en',
              'tps/tp1/tmp.2pvHHmkcyH/test/en/unk-europarl-v7.fi-en-u5.en',
              'tps/tp1/tmp.2pvHHmkcyH/test/en/unk-europarl-v7.fi-en-u10.en',
              'tps/tp1/tmp.2pvHHmkcyH/test/en/unk-europarl-v7.fi-en-u20.en',
              'tps/tp1/tmp.2pvHHmkcyH/test/en/unk-europarl-v7.fi-en-u30.en',
              'tps/tp1/tmp.2pvHHmkcyH/test/en/unk-europarl-v7.fi-en-u40.en',
              'tps/tp1/tmp.2pvHHmkcyH/test/en/unk-europarl-v7.fi-en-u50.en']



def structure_file(file, suf=3, pre=5, clean=False):
    structured_unks = []
    lines = open(file, 'r').readlines()

    # Store index of unks to reconstruct sentences
    unk_index = []
    for j, line in enumerate(lines):
        if clean:
            cleaned_line = clean_line(line)
        else:
            cleaned_line = line

        tokens = tokenize_sentence(cleaned_line)
        n_tokens = len(tokens)
        for i, token in enumerate(tokens):
            if '<unk' in token:
                unk_index.append([j, i])
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
                # y = re.search('<unk w="(.*)"/>', token).group(1)
                structured_unks.append([structured_unk])
    return structured_unks, unk_index


def get_lines(file_name):
    test_file = os.path.join(data_path, file_name)
    lines_clean = []
    lines = open(test_file, 'r').readlines()
    for line in lines:
        lines_clean.append(tokenize_sentence(line))
    return lines_clean


def build_file(file_name, file_out, token_dict, preds_indexes):
    lines = get_lines(file_name)
    for pp in preds_indexes:
        lines[pp[1][0]][pp[1][1]] = '<unk w="{}"/>'.format(token_dict[pp[2]])

    clean_lines = []
    for line in lines:
        clean_lines.append(' '.join(t for t in line))

    with open(file_out, 'a') as f:
        for line in clean_lines:
            if line[-1] == '\n':
                f.write('{}'.format(line))
            else:
                f.write('{}\n'.format(line))
    return True


def iterate_test_batch(token_dict, file_name, batch_size=32,
                       pre=12, suf=6, output_size=10000, clean=True):
    test_file = os.path.join(data_path, file_name)
    structured_unks, unk_indexes = structure_file(test_file, suf=suf, pre=pre, clean=clean)

    n_batch = len(structured_unks) / batch_size
    last_batch_size = int(batch_size * abs(n_batch - float(len(structured_unks) / float(batch_size))))
    for i in range(n_batch):
        bb_x = []
        # yy = []
        for unk in structured_unks[i * batch_size: (i + 1) * batch_size]:
            bb_x.append(unk[0])
            # yy.append(unk[1])

        b_x = np.zeros(shape=[batch_size, pre + suf + 1, output_size + 3])
        # b_y = np.zeros(shape=[batch_size, output_size + 3])
        for j, line in enumerate(bb_x):
            # y_pos = get_token_dict_pos(token_dict, yy[j])
            # b_y[j][y_pos] = 1.
            for k, token in enumerate(line):
                if '<unk w=' in token:
                    x_pos = -2
                else:
                    x_pos = get_token_dict_pos(token_dict, token)
                b_x[j][k][x_pos] = 1.
        yield b_x, unk_indexes[i * batch_size: (i + 1) * batch_size], batch_size
    if last_batch_size > 0:
        bb_x = []
        # yy = []
        for unk in structured_unks[-last_batch_size:]:
            bb_x.append(unk[0])
            # yy.append(unk[1])

        b_x = np.zeros(shape=[last_batch_size, pre + suf + 1, output_size + 3])
        # b_y = np.zeros(shape=[batch_size, output_size + 3])
        for j, line in enumerate(bb_x):
            # y_pos = get_token_dict_pos(token_dict, yy[j])
            # b_y[j][y_pos] = 1.
            for k, token in enumerate(line):
                if '<unk w=' in token:
                    x_pos = -2
                else:
                    x_pos = get_token_dict_pos(token_dict, token)
                b_x[j][k][x_pos] = 1.
        yield b_x, unk_indexes[-last_batch_size:], last_batch_size


if __name__ == '__main__':

    # word2vec:
    with open('clean_token_dict.pickle', 'rb') as f:
        tokens_dict = pickle.load(f)
    sorted_tokens = sorted(tokens_dict.items(), key=lambda item: item[1])
    token_dict = [t[0] for t in sorted_tokens[-10000:]] + ['<s>', '<UNKNOWN>', '<UNK>']
    # batch_size = 32
    rnn_size = 1024
    max_n_token_dict = 10000 + 3

    batch_size = tf.placeholder(tf.int32, [], name='batch_size')
    x = tf.placeholder('float32', shape=[None, None, max_n_token_dict])
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

    sess = tf.Session()
    saver = tf.train.Saver()
    model_path = '{}/lstmpc10k8-4n15_{}_{}/lstm'.format(SAVE_PATH, 60000, 2)
    saver.restore(sess, model_path)

    for file_name in file_names:
        preds_indexes = []
        print(file_name)
        for i, (b_x, unk_index, b_size) in enumerate(iterate_test_batch(token_dict, file_name=file_name,
                                                                 batch_size=32, pre=8, suf=4, output_size=10000,
                                                                 clean=True)):
            preds_ = sess.run([preds], feed_dict={x: b_x, batch_size: b_size})
            preds__ = np.argmax(preds_[0], axis=1)
            preds_indexes = preds_indexes + [[p, unk_index[i], preds__[i]]
                                             for i, p in enumerate(preds__)]

        build_file(file_name, 'results_/lstmpc10k8-4n15_{}.txt'.format(file_name.split('/')[-1]),
                   token_dict, preds_indexes)