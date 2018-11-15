import os
data_path = "/run/media/ashbylepoc/ff112aea-f91a-4fc7-a80b-4f8fa50d41f3/tmp/data/nlp_dev_1/"



shuffled_pc8_4_train_set = os.path.join(data_path, 'en/trainpc8-4_en_shuffled.csv')
shuffled_pc8_4_valid_set = os.path.join(data_path, 'en/validpc8-4_en_shuffled.csv')
data_reader = DataReader(shuffled_pc8_4_train_set, shuffled_pc8_4_valid_set,
                         7, tokens_dict, 7, 10000)
it = enumerate(data_reader.iterate_mini_batch(32, dataset='valid', pre=pre, suf=suf))
all_acc = []
for i in range(100):
    k, (b_x, b_y) = next(it)
    preds_, acc_ = sess.run([preds, acc], feed_dict={x: b_x, y: b_y})
    all_acc.append(acc_)

