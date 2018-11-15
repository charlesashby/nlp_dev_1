from lstm_1 import *

csv.field_size_limit(sys.maxsize)

# data_path = '/home/ashbylepoc/tmp/nlp_dev_1'
data_path = "/run/media/ashbylepoc/ff112aea-f91a-4fc7-a80b-4f8fa50d41f3/tmp/data/nlp_dev_1/"
raw_data = os.path.join(data_path, 'en/train-europarl-v7.fi-en.en')
train_file_name = os.path.join(data_path, 'en/train_en_strutured.csv')
shuffled_train_set = os.path.join(data_path, 'en/train_en_shuffled.csv')
shuffled_valid_set = os.path.join(data_path, 'en/valid_en_shuffled.csv')
word2vec_dir = '/home/ashbylepoc/tmp/word2vec/'
SAVE_PATH = data_path + 'checkpoints'
clean_dataset = os.path.join(data_path, 'clean_dataset.csv')



if __name__ == '__main__':
    with open('../tokens_dict.pickle', 'rb') as f:
        tokens_dict = pickle.load(f)
    print('building graph')
    # b_x, b_y, m = data_reader.make_mini_batch(data_reader.lines[:32])
    batch_size = 32
    # valid_batch_size = 1000
    rnn_size = 1000
    max_n_token_sentence = 100
    max_n_token_dict = 10000 + 3
    learning_rate = 0.001
    data_reader = DataReader(shuffled_train_set, shuffled_valid_set,
                             7, tokens_dict, 7, 10000)
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
    # preds_reshaped = tf.reshape(preds, shape=[-1, max_n_token_sentence, max_n_token_dict])
    cost = - tf.reduce_sum(y * tf.log(tf.clip_by_value(preds, 1e-10, 1.0)), axis=1)
    cost = tf.reduce_mean(cost, axis=0)
    predictions = tf.cast(tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1)), dtype='float32')
    acc = tf.reduce_mean(predictions)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

