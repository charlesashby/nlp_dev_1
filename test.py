from lstm_.lstm import DataReader
import os, pickle, time

data_path = '/home/ashbylepoc/tmp/nlp_dev_1'
train_file_name = os.path.join(data_path, 'train-europarl-v7.fi-en.en')


if __name__ == '__main__':
    print('1: {}'.format(time.time()))
    with open('tokens_dict.pickle', 'rb') as f:
        tokens_dict = pickle.load(f)
    data_reader = DataReader(train_file_name, tokens_dict)
    b_x, b_y = data_reader.make_mini_batch(data_reader.lines[:32], unk_p=0.1)
    print('2: {}'.format(time.time()))
    print(b_x.shape)
    print(b_y.shape)
    for b in b_x[0][0]:
        if b == 1:
            print('ok')