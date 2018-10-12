import tensorflow as tf
import os, csv, pickle, time
from lstm.test_lstm import *
from tensorflow.contrib import rnn

data_path = "/run/media/ashbylepoc/ff112aea-f91a-4fc7-a80b-4f8fa50d41f3/tmp/data/nlp_dev_1/"
SAVE_PATH = data_path + 'checkpoints_hist_lstm_1'
shuffled_p_train_set = os.path.join(data_path, 'en/trainp_en_shuffled.csv')
shuffled_p_valid_set = os.path.join(data_path, 'en/validp_en_shuffled.csv')
shuffled_p8_4_train_set = os.path.join(data_path, 'en/trainp8-4_en_shuffled.csv')
shuffled_p8_4_valid_set = os.path.join(data_path, 'en/validp8-4_en_shuffled.csv')
shuffled_pc8_4_train_set = os.path.join(data_path, 'en/trainpc8-4_en_shuffled.csv')
shuffled_pc8_4_valid_set = os.path.join(data_path, 'en/validpc8-4_en_shuffled.csv')
shuffled_pc12_6_train_set = os.path.join(data_path, 'en/trainpc12-6_en_shuffled.csv')
shuffled_pc12_6_valid_set = os.path.join(data_path, 'en/validpc12-6_en_shuffled.csv')


with open('tokens_dict.pickle', 'rb') as f:
    tokens_dict = pickle.load(f)


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
pre = 5
suf = 3

sess = tf.Session()
saver.restore(sess, '{}/lstmpc10kn12-6_{}_{}/lstm'.format(SAVE_PATH, 35000, 3))

m = 5
test_file = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u{}.en'.format(m))
structured_unks = structure_file(test_file, suf=3, pre=5, clean=True)
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