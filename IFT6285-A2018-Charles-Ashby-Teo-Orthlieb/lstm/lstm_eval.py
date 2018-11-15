import numpy as np
import os, csv, pickle, time
from util.utils import *
import tensorflow as tf
from tensorflow.contrib import rnn
from gensim.models import KeyedVectors

# data_path = '/home/ashbylepoc/tmp/nlp_dev_1'
data_path = "/run/media/ashbylepoc/ff112aea-f91a-4fc7-a80b-4f8fa50d41f3/tmp/data/nlp_dev_1/"
SAVE_PATH = os.path.join(data_path, 'checkpoints_hist_lstm')
word2vec_dir = "/run/media/ashbylepoc/ff112aea-f91a-4fc" \
               "7-a80b-4f8fa50d41f3/tmp/data/word2vec"


# Lancaster Stemmer is used to remove tenses
# from words
# st = LancasterStemmer()
model = KeyedVectors.load_word2vec_format(
    '{}/GoogleNews-vectors-negative300.bin'.format(word2vec_dir),
    binary=True)


def get_word_distance(w1, w2):
    try:
        vector = model.distance(w1, w2)
        return vector
    except KeyError:
        return None


def get_lines(perc_unk, c=False):
    if c:
        test_file = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u{}c.en'.format(perc_unk))
    else:
        test_file = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u{}.en'.format(perc_unk))

    lines_clean = []
    lines = open(test_file, 'r').readlines()
    for line in lines:
        lines_clean.append(tokenize_sentence(line))
    return lines_clean


def build_file(perc_unk, c, file_out, token_dict, preds_indexes):
    lines = get_lines(perc_unk, c)
    for pp in preds_indexes:
        lines[pp[1][0]][pp[1][1]] = '<unk w="{}"/>'.format(token_dict[pp[2]])

    clean_lines = []
    for line in lines:
        clean_lines.append(' '.join(t for t in line))

    with open(file_out, 'a') as f:
        for line in clean_lines:
            f.write('{}\n'.format(line))
    return True


def iterate_test_batch(token_dict, perc_unk=10, c=False, batch_size=32,
                       pre=12, suf=6, output_size=10000, clean=True):
    if c:
        test_file = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u{}c.en'.format(perc_unk))
    else:
        test_file = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u{}.en'.format(perc_unk))

    structured_unks, unk_indexes = structure_file(test_file, suf=suf, pre=pre, clean=clean)

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
        yield b_x, b_y, unk_indexes[i * batch_size: (i + 1) * batch_size]


def eval(clean=True, batch_size=32, rnn_size=1024, max_n_token_dict=10003,
         file_out='lstmp10k5-3v1', pre=8, suf=4, iteration=35000, epoch=1):
    if clean:
        with open('tokens_dict.pickle', 'rb') as f:
            tokens_dict = pickle.load(f)
    else:
        with open('clean_token_dict.pickle', 'rb') as f:
            tokens_dict = pickle.load(f)

    sorted_tokens = sorted(tokens_dict.items(), key=lambda item: item[1])
    token_dict = [t[0] for t in sorted_tokens[-10000:]] + ['<s>', '<UNKNOWN>', '<UNK>']

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
    predictions = tf.cast(tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1)), dtype='float32')
    acc = tf.reduce_mean(predictions)
    top3_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(preds, tf.argmax(y, 1), k=3), dtype='float32'))

    sess = tf.Session()
    saver = tf.train.Saver()
    model_path = '{}/{}__{}_{}/lstm'.format(file_out, SAVE_PATH, iteration, epoch)
    saver.restore(sess, model_path)

    for c in [True, False]:
        for perc_unk in [5, 10, 20, 30, 40]:
            all_acc = []
            all_top3_acc = []
            preds_indexes = []
            for i, (b_x, b_y, unk_index) in enumerate(iterate_test_batch(token_dict, perc_unk=perc_unk, c=c,
                                                    batch_size=batch_size, pre=pre, suf=suf, output_size=max_n_token_dict - 3,
                                                    clean=clean)):
                preds_, acc_, top3_acc_, predictions_ = sess.run(
                    [preds, acc, top3_acc, predictions], feed_dict={x: b_x, y: b_y})
                preds__ = np.argmax(preds_, axis=1)
                b_y_ = np.argmax(b_y, axis=1)
                all_acc.append(acc_)
                all_top3_acc.append(top3_acc_)
                preds_indexes = preds_indexes + [[p, unk_index[i], preds__[i], b_y_[i]]
                                                 for i, p in enumerate(predictions_)]


            # Build file
            build_file(perc_unk, c, 'results/lstmp10k5-3v1_{}_{}.txt'.format(int(c), perc_unk)
                       , token_dict, preds_indexes)

            # Compute sentence accuracies
            acc_sentence = {}
            w2v_dists = []
            n_unknown = 0
            for i, u in enumerate(preds_indexes):
                # token predicted, target token
                t1, t2 = token_dict[preds_indexes[i][2]], token_dict[preds_indexes[i][3]]
                if t2 == '<UNKNOWN>':
                    n_unknown += 1.
                dist = get_word_distance(t1, t2)
                if dist is not None:
                    w2v_dists.append(dist)
                else:
                    if t1 == t2:
                        w2v_dists.append(0.)
                if u[1][0] in acc_sentence:
                    acc_sentence[u[1][0]][1] += 1.
                    if u[0] == 1.:
                        acc_sentence[u[1][0]][0] += 1.
                else:
                    acc_sentence[u[1][0]] = [0., 1.]
                    if u[0] == 1.:
                        acc_sentence[u[1][0]][0] += 1.
            sentence_acc = []
            perfect_sentence = 0

            perc_unknown = n_unknown / len(preds_indexes)

            for i, a in enumerate(acc_sentence):

                accuracy_sentence = acc_sentence[a][0] / acc_sentence[a][1]
                sentence_acc.append(accuracy_sentence)
                if accuracy_sentence == 1.:
                    perfect_sentence += 1.
            print('Acc: {} - Top 3 Acc: {} - % Perfect Sentence Prediction: {} '
                  '- Word2Vec Distance: {} - % unknown: {}'
                  '- percentage truncated: {} - contiguous words: {} '.format(
                np.mean(all_acc), np.mean(all_top3_acc), perfect_sentence / len(sentence_acc),
                np.mean(w2v_dists), perc_unknown, perc_unk, c))
            with open('results/{}_results.txt'.format(file_out), 'a') as f:
                f.write('Acc: {} - Top 3 Acc: {} - % Perfect Sentence Prediction: {} '
                          '- Word2Vec Distance: {} - % unknown: {}'
                          '- percentage truncated: {} - contiguous words: {} '.format(
                        np.mean(all_acc), np.mean(all_top3_acc), perfect_sentence / len(sentence_acc),
                        np.mean(w2v_dists), perc_unknown, perc_unk, c))

if __name__ == '__main__':
    eval()