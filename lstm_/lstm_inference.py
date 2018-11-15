import tensorflow as tf
import os, re
# from lstm_1 import *

# data_path = '/home/ashbylepoc/tmp/nlp_dev_1'
data_path = "/run/media/ashbylepoc/ff112aea-f91a-4fc7-a80b-4f8fa50d41f3/tmp/data/nlp_dev_1/"
raw_data = os.path.join(data_path, 'en/train-europarl-v7.fi-en.en')
train_file_name = os.path.join(data_path, 'en/train_en_strutured.csv')
test_file_name = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u10.en')
shuffled_train_set = os.path.join(data_path, 'en/train_en_shuffled.csv')
shuffled_valid_set = os.path.join(data_path, 'en/valid_en_shuffled.csv')
word2vec_dir = '/home/ashbylepoc/tmp/word2vec/'
SAVE_PATH = data_path + 'checkpoints/lstm_1'

import tensorflow as tf
import os, re
# from lstm_1 import *

# data_path = '/home/ashbylepoc/tmp/nlp_dev_1'
data_path = "/run/media/ashbylepoc/ff112aea-f91a-4fc7-a80b-4f8fa50d41f3/tmp/data/nlp_dev_1/"
raw_data = os.path.join(data_path, 'en/train-europarl-v7.fi-en.en')
train_file_name = os.path.join(data_path, 'en/train_en_strutured.csv')
test_file_name = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u10.en')
shuffled_train_set = os.path.join(data_path, 'en/train_en_shuffled.csv')
shuffled_valid_set = os.path.join(data_path, 'en/valid_en_shuffled.csv')
word2vec_dir = '/home/ashbylepoc/tmp/word2vec/'
SAVE_PATH = data_path + 'checkpoints/lstm_1'


def compute_unknown_test_set(file):
    for perc_unk in [5, 10, 20, 30, 40]:
        test_file = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u{}.en'.format(perc_unk))
        structured_unks = structure_file(test_file, suf=4, pre=8, clean=True)
        # Remove sentences with unks...
        # structured_unks = clean_structured_unks(structured_unks)
        n_batch = len(structured_unks) / batch_size
        total = 0
        unknowns = 0
        for i in range(n_batch):
            bb_x = []
            yy = []
            for unk in structured_unks[i * batch_size: (i + 1) * batch_size]:
                bb_x.append(unk[0])
                yy.append(unk[1])

            for j, line in enumerate(bb_x):
                y_pos = get_token_dict_pos(token_dict, yy[j])
                total += 1
                if y_pos == -2:
                    unknowns += 1
        print('{}/{}: {}'.format(unknowns, total, test_file))





if __name__ == '__main__':
    from lstm.lstm_1 import *
    with open('tokens_dict.pickle', 'rb') as f:
        tokens_dict = pickle.load(f)
    print('building graph')

    # b_x, b_y, m = data_reader.make_mini_batch(data_reader.lines[:32])
    # batch_size = 1
    # valid_batch_size = 1000
    rnn_size = 1000
    max_n_token_sentence = 100
    max_n_token_dict = 10000 + 3
    learning_rate = 0.001
    sorted_tokens = sorted(tokens_dict.items(), key=lambda item: item[1])
    token_dict = [t[0] for t in sorted_tokens[-max_n_token_dict:]] + ['<s>', '<UNKNOWN>', '<UNK>']
    data_reader = DataReader(shuffled_train_set, shuffled_valid_set,
                             10, tokens_dict, 400, 10000)

    # GRAPH
    batch_size = tf.placeholder(tf.int32, [], name='batch_size')
    y = tf.placeholder('float32', shape=[None, max_n_token_dict])
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
    # preds_reshaped = tf.reshape(preds, shape=[-1, max_n_token_sentence, max_n_token_dict])
    cost = - tf.reduce_sum(y * tf.log(tf.clip_by_value(preds, 1e-10, 1.0)), axis=1)
    cost = tf.reduce_mean(cost, axis=0)
    predictions = tf.cast(tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1)), dtype='float32')
    acc = tf.reduce_mean(predictions)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, '{}_{}'.format(SAVE_PATH, 200000))
        with open(test_file_name, 'r') as f:
            lines = f.readlines()
            tokens = [tokenize_sentence(line) for line in lines]
            batches = data_reader.make_batch_test(lines)

            for j, line in enumerate(batches):
                # line = line.reshape(1, 1, max_n_token_dict)
                # unks = []
                completed_sentence = []
                line = np.transpose(line, axes=[1, 0, 2])
                for i in range(10):
                    initial_input = line[:, i:i+4].reshape(1, 4, max_n_token_dict)
                    p, s = sess.run([preds, final_rnn_state],
                                    feed_dict={x: initial_input,
                                               batch_size: 1})
                    print(np.argmax(p))
                fetches = {'last_state': s,
                           'last_pred': p}
                accurate_pred = 0
                unks = []
                for i, token in enumerate(line[1:]):
                    i += 1

                    token = token.reshape(1, 1, max_n_token_dict)
                    if token[0][0][-1] == 1.:
                        # send previous pred
                        target_word = re.search('<unk w="(.*)"/>', tokens[j][i]).group(1)
                        feed_dict = {x: fetches['last_pred'].reshape(1, 1, max_n_token_dict),
                                     initial_rnn_state: fetches['last_state']}
                        p, s = sess.run([preds, final_rnn_state],
                                        feed_dict=feed_dict)
                        fetches['last_state'] = s
                        fetches['last_pred'] = p
                        pred_word_index = np.argmax(p.reshape(max_n_token_dict), axis=0)
                        predicted_word = token_dict[pred_word_index]
                        # completed_sentence.append(predicted_word[0])
                        unks.append(target_word)
                        if target_word == predicted_word:
                            accurate_pred += 1
                        print('PREDICTION: {} - ACTUAL: {}'.format(predicted_word, target_word))


                    else:
                        # send token
                        feed_dict = {x: token,
                                     initial_rnn_state: fetches['last_state']}
                        p, s = sess.run([preds, final_rnn_state],
                                        feed_dict=feed_dict)
                        fetches['last_state'] = s
                        fetches['last_pred'] = p
                        completed_sentence.append(tokens[j][i])
                        # print('ORIGINAL: {}'.format(completed_sentence))
                print("Accuracy: {}".format(float(accurate_pred) / float(len(unks))))

