import matplotlib
import matplotlib.pyplot as plt
import os, pickle
import numpy as np

# Visualizing the # of words for each sentence in the
# training set

data_path = '/home/ashbylepoc/tmp/nlp_dev_1'
train_file_name = os.path.join(data_path, 'train-europarl-v7.fi-en.en')
lines = open(train_file_name, 'r').readlines()
tokens_dist = {i: 0 for i in range(400)}
max_tokens = 0
for line in lines:
    tokens_dist[len(line.split(' '))] += 1

bins = [i for i in range(400)]
fig, axs = plt.subplots(1, tight_layout=True)
y = np.array([tokens_dist[t] for t in tokens_dist.keys()])
axs.bar(bins, y)


# Visualizing tokens distribution
with open('clean_token_dict.pickle', 'rb') as f:
    tokens_dict = pickle.load(f)

# Visualize tokens distribution
sorted_tokens = sorted(tokens_dict.items(), key=lambda item: item[1])
bins = [i for i in range(len(sorted_tokens))]
y = np.array([t[1] for t in reversed(sorted_tokens)])
log_y = [np.log(t) for t in y]
fig_t, axs_t = plt.subplots(1, tight_layout=True)
axs_t.bar(bins, log_y)
axs_t.set_xlabel("token")
axs_t.set_ylabel("log(token occurences)")
axs_t.set_title("Number of occurences (logarithm) of each token in our cleaned dictionary")