import numpy as np
import os, csv, pickle, time
import tensorflow as tf
from tensorflow.contrib import rnn
import sys

csv.field_size_limit(sys.maxsize)

# data_path = '/home/ashbylepoc/tmp/nlp_dev_1'
data_path = "/run/media/ashbylepoc/ff112aea-f91a-4fc7-a80b-4f8fa50d41f3/tmp/data/nlp_dev_1/"
raw_data = os.path.join(data_path, 'en/train-europarl-v7.fi-en.en')
train_file_name = os.path.join(data_path, 'en/train_en_strutured.csv')
shuffled_train_set = os.path.join(data_path, 'en/train_en_shuffled.csv')
shuffled_valid_set = os.path.join(data_path, 'en/valid_en_shuffled.csv')
shuffled_p_train_set = os.path.join(data_path, 'en/trainp_en_shuffled.csv')
shuffled_p_valid_set = os.path.join(data_path, 'en/validp_en_shuffled.csv')
word2vec_dir = '/home/ashbylepoc/tmp/word2vec/'
SAVE_PATH = data_path + 'checkpoints_hist_lstm'
clean_dataset = os.path.join(data_path, 'clean_dataset.csv')
clean_dataset_padded = os.path.join(data_path, 'clean_dataset_padded.csv')

def get_lines(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        return lines

def build_dataset(n_pad=10):
    with open(train_file_name, 'a') as f:
        writer = csv.writer(f)
        lines = get_lines(raw_data)
        for j, line in enumerate(lines):
            if j % 1000 == 0:
                print('working on file {}/{}'.format(j, 1000000))
            padded_line = ' '.join('<s>' for _ in range(n_pad))
            padded_line = '{} {} {}'.format(padded_line, line, padded_line)
            tokens = padded_line.split(' ')
            n_tokens = len(tokens)
            for i in range(n_tokens - n_pad - 1):
                x = tokens[i: i + n_pad + 1]
                writer.writerow(x)

def build_dataset_clean(history_size=10):
    lines = open(raw_data, 'r').readlines()
    with open(clean_dataset_padded, 'a') as f:
        writer = csv.writer(f)
        for j, line in enumerate(lines):
            if j % 1000 == 0:
                print('working on file {}/{}'.format(j, 1000000))
            padded_line = ' '.join('<s>' for _ in range(history_size))
            padded_line = '{} {} {}'.format(padded_line, line, padded_line)
            tokens = padded_line.split(' ')
            n_tokens = len(tokens)
            for k, token in enumerate(tokens):
                if '.\n' in token and token != '.\n':
                    rep = token.replace('.\n', '')
                    tokens[k] = rep
            if n_tokens < history_size + 1:
                pass
            else:
                tt = []
                for i in range(n_tokens - 1 - history_size):
                    x = tokens[i: i + history_size * 2 + 1]
                    ok = True
                    for xx in x:
                        if xx == '':
                            ok = False
                    if ok:
                        if len(x) == history_size * 2 + 1:
                            writer.writerow(x)
                            tt.append(x)



def read_lines(reader, n_lines):
    lines = []
    for i, line in enumerate(reader):
        if i > n_lines:
            break
        else:
            lines.append(line)
    return lines, reader

def shuffle_dataset():
    reader = csv.reader(open(train_file_name, 'r'))
    for i in range(24):
        print('working on {}'.format(i))
        lines, reader = read_lines(reader, 1000000)
        np.random.shuffle(lines)
        with open(shuffled_p_valid_set, 'a') as f:
            writer = csv.writer(f)
            for row in lines:
                writer.writerow(row)

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



class DataReader(object):

    def __init__(self, train_file, valid_file, history_size,
                 tokens_dict, max_n_tokens_sentence, max_n_tokens_dict):
        self.reader_train = csv.reader(open(train_file, 'r'))
        self.reader_valid = csv.reader(open(valid_file, 'r'))
        self.history_size = history_size
        self.max_n_tokens_sentence = max_n_tokens_sentence
        sorted_tokens = sorted(tokens_dict.items(), key=lambda item: item[1])
        self.token_dict = [t[0] for t in sorted_tokens[-max_n_tokens_dict:]] + ['<s>','<UNKNOWN>', '<UNK>']
        self.n_tokens_dict = len(self.token_dict)

        # compute avrg % of unknown

    def compute_perc_unknown(self, x):
        unknowns = 0
        total = 0
        for xx in x:
            for xxx in xx:
                if xxx[-2] == 1.:
                    unknowns += 1
                    total += 1
                else:
                    total += 1
        return float(unknowns) / float(total)

    def get_token_dict_pos(self, token):
        if '<unk w="' in token:
            pos = -1
            return pos
        elif token == '<s>':
            pos = -3
            return pos
        try:
            pos = self.token_dict.index(token)
        except ValueError:
            # print('not found: {}'.format(token))
            pos = -2
        return pos

    def make_batch_test(self, lines):
        n_lines = len(lines)
        test_batch = []

        for l, line in enumerate(lines):

            tokens = tokenize_sentence(line)
            tokens = tokens[:self.max_n_tokens_sentence]
            n_tokens = len(tokens)
            line_x = np.zeros(shape=[1, n_tokens,
                                     self.n_tokens_dict])
            for i, token in enumerate(tokens):
                token_pos = self.get_token_dict_pos(token)
                line_x[0][i][token_pos] = 1.

            test_batch.append(line_x)
        return test_batch

    def make_mini_batch_train(self, lines, batch_size):
        n_lines = len(lines)
        act_batch_size = 0
        mini_batch_x = np.zeros(shape=[batch_size,
                                       self.max_n_tokens_sentence,
                                       self.n_tokens_dict])
        mini_batch_y = np.zeros(shape=[batch_size, self.n_tokens_dict])
        # We load twice the number of lines we need to make sure there are no
        # "unknown" y's (since we reduced the size of the word dict
        lines = enumerate(lines)
        while act_batch_size < batch_size:
            # import pdb; pdb.set_trace()
            j, line = next(lines)
            y = line[-1]
            y_pos = self.get_token_dict_pos(y)

            mini_batch_y[act_batch_size][y_pos] = 1.

            # print('actual batch size: {} {}'.format(act_batch_size < batch_size, act_batch_size))
            if y_pos != -2:
                x = line[- (self.max_n_tokens_sentence + 1):-1]
                # print('x tokens used for inference: {}'.format(x))
                # print('y token used for target: {}'.format(y))
                for i, token in enumerate(x):
                    token_pos = self.get_token_dict_pos(token)
                    mini_batch_x[act_batch_size][i][token_pos] = 1.
                act_batch_size += 1
            else:
                pass
        return mini_batch_x, mini_batch_y

    def mb_predict_middle_sentence(self, lines, batch_size, horizon=4):
        n_lines = len(lines)
        act_batch_size = 0
        mini_batch_x = np.zeros(shape=[batch_size,
                                       21,
                                       self.n_tokens_dict])
        mini_batch_y = np.zeros(shape=[batch_size, self.n_tokens_dict])
        # We load twice the number of lines we need to make sure there are no
        # "unknown" y's (since we reduced the size of the word dict
        lines = enumerate(lines)
        while act_batch_size < batch_size:
            # import pdb; pdb.set_trace()
            j, line = next(lines)

            x = line
            y = x[11]
            y_pos = self.get_token_dict_pos(y)

            mini_batch_y[act_batch_size][y_pos] = 1.

            # print('actual batch size: {} {}'.format(act_batch_size < batch_size, act_batch_size))
            if y_pos != -2:
                # x = line[-9:]

                # print('x tokens used for inference: {}'.format(x))
                # print('y token used for target: {}'.format(y))
                for i, token in enumerate(x):
                    if i == 11:
                        token_pos = -1
                        mini_batch_x[act_batch_size][i][token_pos] = 1.

                    else:
                        token_pos = self.get_token_dict_pos(token)
                        mini_batch_x[act_batch_size][i][token_pos] = 1.
                act_batch_size += 1
            else:
                pass
        return mini_batch_x, mini_batch_y

    def load_to_ram(self, batch_size, file):
        n_rows = batch_size
        self.data = []
        while n_rows > 0:
            self.data.append(next(file))
            n_rows -= 1
        if n_rows == 0:
            return True
        else:
            return False

    def iterate_mini_batch(self, batch_size, dataset='train'):
        if dataset == 'train':
            n_batch = int(2400000 / batch_size)
            for i in range(n_batch):
                # We load twice the number of lines we need to make sure there are no
                # "unknown" y's (since we reduced the size of the word dict
                if self.load_to_ram(batch_size * 2, self.reader_train):
                    # inputs, targets = self.make_mini_batch_train(self.data, batch_size)
                    inputs, targets = self.mb_predict_middle_sentence(self.data, batch_size)

                    yield inputs, targets
        else:
            # n_batch = int(1000. / batch_size)
            n_batch = 100
            for i in range(n_batch):
                # print('valid: {}'.format(i))
                # We load twice the number of lines we need to make sure there are no
                # "unknown" y's (since we reduced the size of the word dict
                if self.load_to_ram(batch_size * 2, self.reader_valid):
                    inputs, targets = self.mb_predict_middle_sentence(self.data, batch_size)
                    # inputs, targets = self.make_mini_batch_train(self.data, batch_size)
                    yield inputs, targets


if __name__ == '__main__':
    # from lstm.lstm_1 import *

    with open('../tokens_dict.pickle', 'rb') as f:
        tokens_dict = pickle.load(f)
    print('building graph')
    # b_x, b_y, m = data_reader.make_mini_batch(data_reader.lines[:32])
    batch_size = 32
    # valid_batch_size = 1000
    rnn_size = 2048
    max_n_token_sentence = 100
    max_n_token_dict = 10000 + 3
    learning_rate = 0.001
    # data_reader = DataReader(shuffled_train_set, shuffled_valid_set,
    #                          7, tokens_dict, 7, 10000)
    # tt = data_reader.iterate_mini_batch(batch_size)
    # t = next(tt)

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
    preds_argmax = tf.argmax(preds, axis=1)
    # preds_reshaped = tf.reshape(preds, shape=[-1, max_n_token_sentence, max_n_token_dict])
    cost = - tf.reduce_sum(y * tf.log(tf.clip_by_value(preds, 1e-10, 1.0)), axis=1)
    cost = tf.reduce_mean(cost, axis=0)
    predictions = tf.cast(tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1)), dtype='float32')
    acc = tf.reduce_mean(predictions)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    print('Done building graph ')
    print('training...')
    saver = tf.train.Saver()
    best_acc = 0.
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, SAVE_PATH)
        for epoch in range(100):
            data_reader = DataReader(shuffled_p_train_set, shuffled_p_valid_set,
                                     7, tokens_dict, 7, 10000)
            # print('ok')
            train_acc = []
            train_cost = []
            for i, batch in enumerate(data_reader.iterate_mini_batch(batch_size)):
                b_x, b_y = batch
                _, c, a, preds_ = sess.run([optimizer, cost, acc, preds],
                                            feed_dict={x: b_x, y: b_y})
                train_acc.append(a)
                train_cost.append(c)

                # Check last prediction:
                # argmax = np.argmax(preds_, axis=1)
                # print('-'.join(a for a in argmax[:10]))
                if i % 500 == 0:
                    print(np.argmax(preds_, axis=1))
                    print(np.argmax(b_y, axis=1))
                    print('TRAIN: iteration: {} - acc: {} - loss: {}'.format(i, np.mean(train_acc), np.mean(train_cost)))
                    with open('log_1.txt', 'a') as f:
                        f.write('TRAIN: iteration: {} - acc: {} - loss: {} \n'.format(i, np.mean(train_acc), np.mean(train_cost)))
                    train_acc = []
                    train_cost = []
                if i % 5000 == 0:
                    # print('validating')
                    valid_acc = []
                    for k, batch_valid in enumerate(data_reader.iterate_mini_batch(batch_size, dataset='valid')):
                        bb_x, bb_y = batch_valid
                        # compute accuracy on validation set
                        cc, aa, preds__ = sess.run([cost, acc, preds],
                                           feed_dict={x: bb_x, y: bb_y})
                        # reshaped_predictions__ = predictions__.reshape((-1,))
                        valid_acc.append(aa)
                    mean_acc = np.mean(valid_acc)
                    if mean_acc > best_acc:
                        best_acc = mean_acc
                        os.mkdir('{}/lstmp_{}_{}'.format(SAVE_PATH, i, epoch))
                        save_path = saver.save(sess, '{}/lstmp_{}_{}/lstm'.format(SAVE_PATH, i, epoch))
                        with open('log.txt', 'a') as f:
                            f.write('saving model: {}/lstmp_{}_{}/lstm'.format(SAVE_PATH, i, epoch))
                            print('saving model: {}/lstmp_{}_{}/lstm'.format(SAVE_PATH, i, epoch))
                    print(np.argmax(preds__, axis=1))
                    print(np.argmax(bb_y, axis=1))
                    print('VALID: iteration: {} - acc: {} -- last_pred:'.format(i, mean_acc))
                    with open('log.txt', 'a') as f:
                        f.write('VALID: iteration: {} - acc: {} \n'.format(i, mean_acc))


    """
        with tf.Session() as sess:
        # saver.restore(sess, SAVE_PATH)
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            _, c, a, predictions_ = sess.run([optimizer, cost, acc, predictions],
                               feed_dict={x: t[0], y: t[1]})
            print('acc: {} - cost: {}'.format(a, c))
    
    
    # Compute percentage unknown in a minibatch depending on the batch size
    
    for i in [6000, 7000, 8000, 9000, 10000, 15000, 20000, 30000]:
        data_reader = DataReader(shuffled_train_set, shuffled_valid_set,
                                 10, tokens_dict, 4, i)
        tt = data_reader.iterate_mini_batch(batch_size)
        t = next(tt)
        print(data_reader.compute_perc_unknown(t[0]))
    """
