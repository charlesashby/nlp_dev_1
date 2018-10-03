import tensorflow as tf
from lstm import DataReader


class DataReader(object):

    def __init__(self, file, token_dict, max_n_tokens_dict, max_n_tokens_sentence):
        """

        :param file:
        :param token_dict:
            dictionary of all the tokens in
            our corpus; to create our final
            dict we add 2 "tokens": <UNKNOWN>,
            <UNKS>. Respectively, they are the
            len(token_dict)'th and len(token_dict)'th + 1
            elements of the dict
        """
        self.file = file
        self.max_n_tokens_sentence = max_n_tokens_sentence
        sorted_tokens = sorted(token_dict.items(), key=lambda item: item[1])
        self.token_dict = [t[0] for t in sorted_tokens[-max_n_tokens_dict:]] + ['<UNKNOWN>', '<UNK>']
        # self.token_dict = [t for t in token_dict] + ['<UNKNOWN>', '<UNK>']
        self.n_tokens_dict = len(self.token_dict)

        with open(self.file, 'r') as f:
            self.lines = f.readlines()
            self.n_lines = len(self.lines)

    def get_token_dict_pos(self, token):
        try:
            pos = self.token_dict.index(token)
        except ValueError:
            pos = -2
        return pos

    def make_mini_batch(self, lines, unk_p):
        """
        hide a % of the words, replace with special token,
        convert to one-hot vector and return the minibatch
        for X and Y. In our task:

        X = x1, ..., xn   Y = y1, ..., yc

        :param lines:
            lines to transform into mini batches
        :return:
            For x: Numpy array of shape (n_lines, max_n_words_per_sentence, n_tokens)
                    where n_tokens = # tokens in the whole dictionary
            For y: Numpy array of shape (n_lines, n_unks, n_tokens)
        """
        # mini_batch_x = []
        # mini_batch_y = []
        # mini_batch_unk_pos = []
        # max_n_tokens = 0
        # max_n_unks = 0
        n_lines = len(lines)
        mini_batch_x = np.zeros(shape=[n_lines,
                                       self.max_n_tokens_sentence,
                                       self.n_tokens_dict])
        mini_batch_y = np.zeros(shape=[n_lines,
                                       self.max_n_tokens_sentence,
                                       self.n_tokens_dict])
        mini_batch_unk_pos = np.zeros(shape=[n_lines,
                                              self.max_n_tokens_sentence])
        for l, line in enumerate(lines):

            tokens = line.split(' ')[:self.max_n_tokens_sentence]
            n_tokens = len(tokens)
            n_unks = int(n_tokens * unk_p)
            unks = [random.randint(0, n_tokens - 1) for _ in range(n_unks)]
            # sub_batch_x = np.zeros(shape=(n_tokens, self.n_tokens_dict))
            # sub_batch_y = np.zeros(shape=(n_tokens, self.n_tokens_dict))
        #     sub_batch_y = np.zeros(shape=(self.max_n_tokens_sentence, self.n_tokens_dict))
        #     sub_batch_x = np.zeros(shape=(self.max_n_tokens_sentence, self.n_tokens_dict))
            # j is an index for the unks that we have processed
            # j = 0
            for i, token in enumerate(tokens):
                token_pos = self.get_token_dict_pos(token)
                # one_hot_x = np.zeros(shape=self.n_tokens_dict)
                if i in unks:
                    # Make sure token_pos is not -2 otherwise don't
                    # hide this token (useless)
                    if token_pos != -2:
                        # Find the position of the token that we're
                        # removing and add the one hot vector to
                        # the mini batches
                        # one_hot_y = np.zeros(shape=self.n_tokens_dict)
                        # one_hot_y[token_pos] = 1
                        # one_hot_x[-1] = 1
                        # sub_batch_x[i, :] = one_hot_x
                        # sub_batch_y[i, :] = one_hot_y
                        # j += 1
                        mini_batch_x[l][i][-1] = 1.
                        mini_batch_y[l][i][token_pos] = 1.
                    else:
                        # If the token was unknown remove it from
                        # the list of unk
                        del unks[unks.index(i)]

                else:
                    # one_hot_y = np.zeros(shape=self.n_tokens_dict)
                    # one_hot_x[token_pos] = 1
                    # sub_batch_x[i, :] = one_hot_x
                    # sub_batch_y[i, :] = one_hot_x
                    mini_batch_x[l][i][token_pos] = 1.
                    mini_batch_y[l][i][token_pos] = 1.

            # Keep the position of the unks
            # sub_batch_unk_pos = np.zeros(shape=n_tokens)
            # sub_batch_unk_pos[unks] = 1.
            mini_batch_unk_pos[l][unks] = 1.
            # Compute the maximum number of tokens because
            # at the end we want to pad the smaller sentences
            # (those with less tokens) with 0's
            # if max_n_tokens < n_tokens:
            #     max_n_tokens = n_tokens
            # if max_n_unks < n_unks:
            #     max_n_unks = n_unks

            # Add the sentence to the minibatch
            # mini_batch_x.append(sub_batch_x)
            # mini_batch_y.append(sub_batch_y)
            # mini_batch_unk_pos.append(sub_batch_unk_pos)
        # Convert the mini_batch (list) to numpy arrays
        # mini_batch_x_padded = np.zeros(shape=(n_lines,
        #                                     max_n_tokens,
        #                                     self.n_tokens_dict))
        # mini_batch_y_padded = np.zeros(shape=(n_lines,
        #                                       max_n_tokens,
        #                                       self.n_tokens_dict))
        # mini_batch_unk_pos_padded = np.zeros(shape=(n_lines, max_n_tokens))

        # for i in range(n_lines):
        #     mini_batch_x_padded[i, :len(mini_batch_x[i]), :] = mini_batch_x[i]
        #     mini_batch_y_padded[i, :len(mini_batch_y[i]), :] = mini_batch_y[i]
        #     mini_batch_unk_pos_padded[i, :len(mini_batch_unk_pos[i])] = mini_batch_unk_pos[i]

        # return mini_batch_x_padded, mini_batch_y_padded, mini_batch_unk_pos_padded
        return mini_batch_x, mini_batch_y, mini_batch_unk_pos

    def iterate_mini_batch(self, batch_size, unk_p):
        n_samples = self.n_lines
        n_batch = int(n_samples / batch_size)

        for i in range(n_batch):
            inputs, targets, unk_pos = self.make_mini_batch(
                self.lines[i * batch_size: (i + 1) * batch_size], unk_p)
            yield inputs, targets
