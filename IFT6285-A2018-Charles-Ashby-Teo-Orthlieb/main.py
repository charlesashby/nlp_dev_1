from lstm.lstm import train
from lstm.lstm_eval import eval

if __name__ == '__main__':
    # Train LSTM clean set 10k dict size 5 prefix 3 suffix
    train(clean=True, max_n_token_dict=10003, pre=5, suf=3, noise=None, file_out='lstmc10k5-3')

    # Train LSTM not clean set 10k dict size 5 prefix 3 suffix
    train(clean=False, max_n_token_dict=10003, pre=5, suf=3, noise=None, file_out='lstm10k5-3')

    # Train LSTM clean set 10k dict size 8 prefix 4 suffix
    train(clean=True, max_n_token_dict=10003, pre=8, suf=4, noise=None, file_out='lstmc10k8-3')

    # Train LSTM not clean set 10k dict size 8 prefix 4 suffix
    train(clean=False, max_n_token_dict=10003, pre=8, suf=4, noise=None, file_out='lstm10k8-4')

    # Train LSTM clean set 10k dict size 12 prefix 6 suffix 0.35 noise
    train(clean=True, max_n_token_dict=10003, pre=12, suf=6, noise=0.35, file_out='lstmc35n10k12-6')

    # Train LSTM clean set 10k dict size 8 prefix 4 suffix 0.35 noise
    train(clean=True, max_n_token_dict=10003, pre=8, suf=4, noise=0.35, file_out='lstmc35n10k8-4')

    # Train LSTM clean set 10k dict size 5 prefix 3 suffix 0.35 noise
    train(clean=True, max_n_token_dict=10003, pre=5, suf=3, noise=0.35, file_out='lstmc35n10k5-3')

    # Evaluate LSTM clean set 10k dict size 5 prefix 3 suffix
    eval(clean=True, file_out='lstmc10k5-3', pre=5, suf=3, epoch=1, iteration=35000)
