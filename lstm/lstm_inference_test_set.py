import numpy as np
import os, csv, pickle, time
from util.utils import *
import tensorflow as tf
from tensorflow.contrib import rnn

# data_path = '/home/ashbylepoc/tmp/nlp_dev_1'
data_path = "/run/media/ashbylepoc/ff112aea-f91a-4fc7-a80b-4f8fa50d41f3/tmp/data/nlp_dev_1/"
SAVE_PATH = os.path.join(data_path, 'checkpoints_hist_lstm')


def iterate_test_batch(token_dict, perc_unk=10, c=False, batch_size=32,
                       pre=5, suf=3, output_size=10000, clean=True):
    if c:
        test_file = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u{}c.en'.format(perc_unk))
    else:
        test_file = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u{}.en'.format(perc_unk))

    structured_unks = structure_file(test_file, suf=suf, pre=pre, clean=clean)

    n_batch = len(structured_unks) / batch_size
    for i in range(n_batch):
        bb_x = []
        yy = []
        for unk in structured_unks[i * batch_size: (i + 1) * batch_size]:
            bb_x.append(unk[0])
            yy.append(unk[1])

        b_x = np.zeros(shape=[batch_size, pre + suf + 1, output_size + 3])
        b_y = np.zeros(shape=[batch_size, output_size + 3])
        for j, line in enumerate(bb_x):
            y_pos = get_token_dict_pos(token_dict, yy[j])
            b_y[j][y_pos] = 1.
            for k, token in enumerate(line):
                if '<unk w=' in token:
                    x_pos = -2
                else:
                    x_pos = get_token_dict_pos(token_dict, token)
                b_x[j][k][x_pos] = 1.
        yield b_x, b_y


if __name__ == '__main__':
    with open('clean_token_dict.pickle', 'rb') as f:
        tokens_dict = pickle.load(f)
    sorted_tokens = sorted(tokens_dict.items(), key=lambda item: item[1])
    token_dict = [t[0] for t in sorted_tokens[-10000:]] + ['<s>', '<UNKNOWN>', '<UNK>']

    batch_size = 32
    rnn_size = 1024
    max_n_token_dict = 10000 + 3

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

    sess = tf.Session()
    saver = tf.train.Saver()
    model_path = '{}/lstmpc10kn8-4_{}_{}/lstm'.format(SAVE_PATH, 30000, 3)
    saver.restore(sess, model_path)

    for c in [True, False]:
        for perc_unk in [5, 10, 20, 30, 40]:
            all_acc = []
            for i, (b_x, b_y) in enumerate(iterate_test_batch(token_dict, perc_unk=perc_unk, c=c,
                                                    batch_size=32, pre=8, suf=4, output_size=10000,
                                                    clean=True)):
                preds_, acc_ = sess.run([preds, acc], feed_dict={x: b_x, y: b_y})
                all_acc.append(acc_)
            print('Acc: {} - percentage truncated: {} - contiguous words: {}'.format(
                np.mean(all_acc), perc_unk, c))
            with open('results/lstmp10kn8-4_results.txt', 'a') as f:
                f.write('TEST: Acc: {} - percentage truncated: {} - contiguous words: {}\n'.format(
                np.mean(all_acc), perc_unk, c))