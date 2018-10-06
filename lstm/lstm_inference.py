import tensorflow as tf
import os, re


# data_path = '/home/ashbylepoc/tmp/nlp_dev_1'
data_path = "/run/media/ashbylepoc/ff112aea-f91a-4fc7-a80b-4f8fa50d41f3/tmp/data/nlp_dev_1"
test_file_name = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u10.en')
word2vec_dir = '/home/ashbylepoc/tmp/word2vec/'
SAVE_PATH = data_path + 'checkpoints/lstm_1'




if __name__ == '__main__':
    from lstm.lstm import *

    with open('tokens_dict.pickle', 'rb') as f:
        tokens_dict = pickle.load(f)
    print('building graph')
    # Visualize tokens distribution
    sorted_tokens = sorted(tokens_dict.items(), key=lambda item: item[1])

    # b_x, b_y, m = data_reader.make_mini_batch(data_reader.lines[:32])
    batch_size = 1
    rnn_size = 1000
    max_n_token_sentence = 400
    max_n_token_dict = 1000 + 3
    learning_rate = 0.01
    data_reader = DataReader(train_file_name, tokens_dict,
                             max_n_tokens_dict=1000,
                             max_n_tokens_sentence=max_n_token_sentence)
    # data_reader = DataReader(train_file_name, tokens_dict,
    #                          max_n_tokens_dict=1000, max_n_tokens_sentence=max_n_token_sentence)
    token_dict = [t for t in tokens_dict][-1000:] + ['</s>', '<UNKNOWN>', '<UNK>']
    n_tokens_dict = len(token_dict)
    # masks = tf.placeholder('float32', shape=[None, max_n_token_sentence, max_n_token_dict])
    # y = tf.placeholder('float32', shape=[None, max_n_token_sentence, max_n_token_dict])
    # x = tf.placeholder('float32', shape=[None, max_n_token_sentence, max_n_token_dict])
    x = tf.placeholder('float32', shape=[None, None, max_n_token_dict])
    y = tf.placeholder('float32', shape=[None, None, max_n_token_dict])
    cell = rnn.LSTMCell(rnn_size, state_is_tuple=True, forget_bias=0.0, reuse=False)
    initial_rnn_state = cell.zero_state(batch_size, dtype='float32')
    outputs, final_rnn_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_rnn_state,
                                                 dtype='float32')
    outputs_reshape = tf.reshape(outputs, shape=[-1, rnn_size])
    w = tf.get_variable("w", [rnn_size, max_n_token_dict], dtype='float32')
    b = tf.get_variable("b", [max_n_token_dict], dtype='float32')
    preds = tf.nn.softmax(tf.matmul(outputs_reshape, w) + b)
    # preds_reshaped = tf.reshape(preds, shape=[-1, max_n_token_sentence, max_n_token_dict])

    # compute loss
    y_reshaped = tf.reshape(y, shape=[-1, max_n_token_dict])
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, SAVE_PATH)
        with open(test_file_name) as f:
            lines = f.readlines()
            tokens = [tokenize_sentence(line) for line in lines]
            b_x, b_y, m = data_reader.make_mini_batch(lines)

            for j, line in enumerate(b_x):
                # line = line.reshape(1, 1, 1003)
                # unks = []
                completed_sentence = []

                initial_input = line[0].reshape(1, 1, 1003)
                p, s = sess.run([preds, final_rnn_state],
                                feed_dict={x: initial_input})
                fetches = {'last_state': s,
                           'last_pred': p}
                accurate_pred = 0
                unks = []
                for i, token in enumerate(line[1:]):
                    i += 1
                    if token[-3] == 1.:
                        break
                    token = token.reshape(1, 1, 1003)
                    if token[0][0][-1] == 1.:
                        # send previous pred
                        target_word = re.search('<unk w="(.*)"/>', tokens[j][i]).group(1)
                        feed_dict = {x: fetches['last_pred'].reshape(1, 1, 1003),
                                     initial_rnn_state: fetches['last_state']}
                        p, s = sess.run([preds, final_rnn_state],
                                        feed_dict=feed_dict)
                        fetches['last_state'] = s
                        fetches['last_pred'] = p
                        pred_word_index = np.argmax(p.reshape(1003), axis=0)
                        predicted_word = tokens_dict[pred_word_index]
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

