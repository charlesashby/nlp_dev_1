import matplotlib
import matplotlib.pyplot as plt
import os
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
