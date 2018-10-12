import numpy as np
import os, csv, pickle, time
from util.utils import *
import tensorflow as tf
from tensorflow.contrib import rnn

# data_path = '/home/ashbylepoc/tmp/nlp_dev_1'
data_path = "/run/media/ashbylepoc/ff112aea-f91a-4fc7-a80b-4f8fa50d41f3/tmp/data/nlp_dev_1/"
SAVE_PATH = os.path.join(data_path, 'checkpoints_hist_lstm')

# Accuracies lstmpc10kn12-6 1024 rnn size checkpoint: {}/lstmpc10kn12-6_35000_3/lstm
# Acc: 0.379237294197 - Top 3 Acc: 0.5 - percentage truncated: 5 - contiguous words: True
# Acc: 0.35130494833 - Top 3 Acc: 0.482486277819 - percentage truncated: 10 - contiguous words: True
# Acc: 0.337271332741 - Top 3 Acc: 0.467035055161 - percentage truncated: 20 - contiguous words: True
# Acc: 0.300570547581 - Top 3 Acc: 0.425311207771 - percentage truncated: 30 - contiguous words: True
# Acc: 0.266157120466 - Top 3 Acc: 0.391931116581 - percentage truncated: 40 - contiguous words: True
# Acc: 0.392613649368 - Top 3 Acc: 0.5215908885 - percentage truncated: 5 - contiguous words: False
# Acc: 0.38968372345 - Top 3 Acc: 0.51807230711 - percentage truncated: 10 - contiguous words: False
# Acc: 0.385116904974 - Top 3 Acc: 0.52360612154 - percentage truncated: 20 - contiguous words: False
# Acc: 0.361583769321 - Top 3 Acc: 0.501472532749 - percentage truncated: 30 - contiguous words: False
# Acc: 0.360232055187 - Top 3 Acc: 0.500659286976 - percentage truncated: 40 - contiguous words: False

# Accuracies lstmpc10kn8-4 1024 rnn size checkpoint: {}/lstmpc10kn8-4_30000_3/lstm
# Acc: 0.396716088057 - Top 3 Acc: 0.538665235043 - percentage truncated: 5 - contiguous words: True
# Acc: 0.365384608507 - Top 3 Acc: 0.509958803654 - percentage truncated: 10 - contiguous words: True
# Acc: 0.34222561121 - Top 3 Acc: 0.488376528025 - percentage truncated: 20 - contiguous words: True
# Acc: 0.306405603886 - Top 3 Acc: 0.441779047251 - percentage truncated: 30 - contiguous words: True
# Acc: 0.27699303627 - Top 3 Acc: 0.409539461136 - percentage truncated: 40 - contiguous words: True
# Acc: 0.378409087658 - Top 3 Acc: 0.523295462132 - percentage truncated: 5 - contiguous words: False
# Acc: 0.391942769289 - Top 3 Acc: 0.533885538578 - percentage truncated: 10 - contiguous words: False
# Acc: 0.407598912716 - Top 3 Acc: 0.540692448616 - percentage truncated: 20 - contiguous words: False
# Acc: 0.365674078465 - Top 3 Acc: 0.518488228321 - percentage truncated: 30 - contiguous words: False
# Acc: 0.372758448124 - Top 3 Acc: 0.519646644592 - percentage truncated: 40 - contiguous words: False

# Accuracies lstmp10k5-3 1024 rnn size checkpoint: {}/lstmp_25000_1/lstm
# Acc: 0.356991529465 - Top 3 Acc: 0.477754235268 - percentage truncated: 5 - contiguous words: True
# Acc: 0.324175834656 - Top 3 Acc: 0.448832422495 - percentage truncated: 10 - contiguous words: True
# Acc: 0.29382622242 - Top 3 Acc: 0.408727139235 - percentage truncated: 20 - contiguous words: True
# Acc: 0.242738589644 - Top 3 Acc: 0.360088169575 - percentage truncated: 30 - contiguous words: True
# Acc: 0.203076630831 - Top 3 Acc: 0.314434975386 - percentage truncated: 40 - contiguous words: True
# Acc: 0.366477280855 - Top 3 Acc: 0.491477280855 - percentage truncated: 5 - contiguous words: False
# Acc: 0.356551200151 - Top 3 Acc: 0.486445784569 - percentage truncated: 10 - contiguous words: False
# Acc: 0.35948741436 - Top 3 Acc: 0.486285984516 - percentage truncated: 20 - contiguous words: False
# Acc: 0.328534036875 - Top 3 Acc: 0.456806272268 - percentage truncated: 30 - contiguous words: False
# Acc: 0.324498951435 - Top 3 Acc: 0.456882923841 - percentage truncated: 40 - contiguous words: False



def iterate_test_batch(token_dict, perc_unk=10, c=False, batch_size=32,
                       pre=12, suf=6, output_size=10000, clean=True):
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
    with open('tokens_dict.pickle', 'rb') as f:
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
    top3_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(preds, tf.argmax(y, 1), k=3), dtype='float32'))
    sess = tf.Session()
    saver = tf.train.Saver()
    model_path = '{}/lstmp_{}_{}/lstm'.format(SAVE_PATH, 25000, 1)
    saver.restore(sess, model_path)

    for c in [True, False]:
        for perc_unk in [5, 10, 20, 30, 40]:
            all_acc = []
            all_top3_acc = []
            for i, (b_x, b_y) in enumerate(iterate_test_batch(token_dict, perc_unk=perc_unk, c=c,
                                                    batch_size=32, pre=5, suf=3, output_size=10000,
                                                    clean=False)):
                preds_, acc_, top3_acc_ = sess.run([preds, acc, top3_acc], feed_dict={x: b_x, y: b_y})
                all_acc.append(acc_)
                all_top3_acc.append(top3_acc_)
            print('Acc: {} - Top 3 Acc: {} - percentage truncated: {} - contiguous words: {}'.format(
                np.mean(all_acc), np.mean(all_top3_acc), perc_unk, c))
            with open('results/lstmp10k5-3_results.txt', 'a') as f:
                f.write('TEST: Acc: {} - Top 3 Acc: {} - percentage truncated: {} - contiguous words: {}\n'.format(
                np.mean(all_acc), np.mean(all_top3_acc), perc_unk, c))