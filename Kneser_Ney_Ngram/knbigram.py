import os, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# data_path = '/run/media/ashbylepoc/ff112aea-f91a-4fc7-a80b-4f8fa50d41f3/tmp/data/nlp_dev_1'
data_path = '/home/ashbylepoc/tmp/nlp_dev_1/'
train_file_name = os.path.join(data_path, 'train-europarl-v7.fi-en.en')
test_sets = [os.path.join(data_path,
                          'unk-europarl-v7.fi-en-u{}c.en'.format(i))
             for i in [5, 10, 20, 30, 40]]
test_sets_non_contiguous = [os.path.join(data_path,
                          'unk-europarl-v7.fi-en-u{}.en'.format(i))
             for i in [5, 10, 20, 30, 40]] + test_sets
# Read the data and create count dictionaries
def getcountdic(dataFile):
    print 'please wait while it finishes reading the data ... '
    unigram={}
    bigram={}
    N_1 = {}
    N1_ = {}
    # open the file
    with open(dataFile) as inp:
        # loop over the lines
        for j, line in enumerate(inp):
            if j % 100000 == 0:
                print(j)
            line='<s> '+line.strip()+' </s>'
            allwords = line.split(' ')
            # Loop over the words in a line
            for i,aword in enumerate(allwords):

                # Puting data in unigram
                if aword in unigram:
                    unigram[aword]+=1
                else:
                    unigram[aword]=1
                # Puting data in bigram
                if i==0:
                    continue
                if i < len(allwords):
                    if (allwords[i-1],allwords[i]) in bigram:
                        bigram[allwords[i-1],allwords[i]]+=1
                    else:
                        bigram[allwords[i-1],allwords[i]]=1
        # Counting the wildcard dictionaries
        for akey in bigram.keys():
            if akey[0] in N1_:
                N1_[akey[0]]+=1
            else:
                N1_[akey[0]]=1 
            if akey[1] in N_1:
                N_1[akey[1]]+=1
            else:
                N_1[akey[1]]=1
    print 'done'
    return unigram, bigram, N_1, N1_


def KNsmoothing(c_wii_wi, cwii, cwi, nw1_ii, nw_1i, n__, N):
    if cwii > 0:
        return max(0.0,c_wii_wi - 0.5)/cwii + 0.5/cwii*nw1_ii*nw_1i/n__
    else:
        return cwi/float(N)


def testcount(testfile, unigram, bigram, n_1, n1_, dispSteps=False):
    n__ = sum(n_1.values())
    N = sum(unigram.values()) - unigram['<s>']
    # Open the test file
    with open(testfile) as test:
        pkn = 0.0
        count_n = 0
        mle_list = []
        smoothed_list = []
        for line in test:
            # add start and end tags to each sentence
            line='<s> '+line.strip()+' </s>'
            allwords = line.split(' ')
            # Iterate over the words in the line and compute pkn
            for i,aword in enumerate(allwords):
                if i==0:
                    continue
                # count bigrams
                bigCount = bigram.get((allwords[i-1],allwords[i]),0)
                if dispSteps:
                    print
                    print 'processing ('+allwords[i-1]+','+allwords[i]+\
                    ') :bigram count',bigCount
                # Smoothing 
                smoothed = 0
                if bigCount==0:
                    # Out of vocabulary case. Skip
                    if allwords[i] not in unigram:
                        if dispSteps:
                            print 'mle 0.0 (oov), skipping', allwords[i]
                        continue
                    # If the context is unseen
                    elif (allwords[i-1] not in unigram) and \
                        (allwords[i] in unigram):
                        smoothed = unigram[allwords[i]]/float(N)
                        if dispSteps:
                            print 'mle 0.0 (unseen context), unigram',smoothed
                    # If the bigram is unseen
                    elif (allwords[i-1] in unigram) and (allwords[i] in unigram):
                        smoothed = KNsmoothing(bigCount,unigram[allwords[i-1]],
                                               unigram[allwords[i]], n1_[allwords[i-1]], n_1[allwords[i]],
                                               n__, N)
                        if dispSteps:                        
                            print 'mle 0.0 (unseen bigram), smoothed',smoothed
                    mle = 0.0
                else:
                    # When the bigram has count > 0
                    smoothed = KNsmoothing(bigCount,
                                           unigram[allwords[i-1]], unigram[allwords[i]],
                                           n1_[allwords[i-1]], n_1[allwords[i]], n__,N)
                    # mle
                    mle = bigCount/float(unigram[allwords[i-1]])
                    
                    if dispSteps:
                        print 'mle',mle,',',
                        print 'smoothed',smoothed
                pkn = pkn + np.log(smoothed)
                count_n = count_n +1
                mle_list.append(mle)
                smoothed_list.append(smoothed)
        print 
        print 'total logp_kn', pkn
        print 'n =',count_n
        print 'Mean logp_kn',pkn/float(count_n)
        print 'Perplexity',np.exp(-1*pkn/float(count_n))
        
        if not dispSteps:
            mle_list = np.sort(mle_list,kind='heapsort')
            smoothed_list = np.sort(smoothed_list,kind='heapsort')
            plt.loglog(np.array(mle_list),'r-')
            plt.hold('on')
            plt.loglog(np.array(smoothed_list))
            plt.hold('off')
            plt.legend(['MLE','Kneser Ney'],loc='upper left')
            plt.xlabel('Various Bigrams (sorted by corresponding probabilities)')
            plt.ylabel('probability')
            plt.show()        
  
def main():
   print 'Running on sample dataset'
   uni,bi,n_1,n1_ =  getcountdic('/home/ashbylepoc/tmp/nlp_dev_1/train-europarl-v7.fi-en.en')
   print 'unigram:',uni
   print 'bigram:',bi
   print 'N1(wi-1,*):',n1_
   print 'N1(*,wi):',n_1  
   
   testcount('Kneser_Ney_Ngram/kn.test',uni,bi,n_1,n1_,dispSteps=True)
   
   print
   print 'Running on real dataset'
   uni,bi,n_1,n1_ = getcountdic('./training.eng')
   testcount('./test.eng',uni,bi,n_1,n1_,dispSteps=False)


def create_ngram_prob_table(uni, bi, n_1, n1_, file='Kneser_Ney_Ngram/bigram_prob.txt'):
    n__ = sum(n_1.values())
    N = sum(uni.values()) - uni['<s>']
    prob_list = []
    with open(file, 'a') as f:
        for bigram in bi.keys():
            f.write('{} {}\n'.format(bigram[0], bigram[1]))
    with open(file, 'r') as f:
        for bigram in f.readlines():
            bigrams = bigram[:-1].split(' ')
            bigCount = bi.get((bigrams[0], bigrams[1]), 0)
            smoothed = KNsmoothing(bigCount, uni[bigrams[0]],
                                   uni[bigrams[1]], n1_[bigrams[0]],
                                   n_1[bigrams[1]], n__, N)
            mle = bigCount/float(uni[bigrams[0]])
            pkn = np.log(smoothed)
            prob_list.append((bigrams[0], bigrams[1], pkn))
    with open('KN_prob_list', 'a') as f:
        for pp in prob_list:
            f.write('{} {} || {}\n'.format(pp[0], pp[1], pp[2]))


def knp_to_dict(file='KN_prob_list'):
    dd = {}
    with open(file, 'r') as f:
        for line in f.readlines():
            ll = line.split(' || ')
            lll = ll[0].split(' ')
            dd[(lll[0], lll[1])] = float(ll[1][:-1])
    return dd


def compute_probs(bigrams):
    unigrams = {}
    for bigram in bigrams:
        if bigram[0] in unigrams:
            unigrams[bigram[0]].append((bigram, bigrams[bigram]))
        else:
            unigrams[bigram[0]] = [(bigram, bigrams[bigram])]
    unigram_clean = {}
    for unigram in unigrams:
        sort = sorted(unigrams[unigram], key= lambda x: x[1])
        unigram_clean[unigram] = sort[-1]
        # unigram_clean[unigram] = sort

    return unigram_clean


def get_highest_prob_bigram(prefix, bigram_dict):
    probs = [(k[0], k[1], v) for k, v in bigram_dict.items() if prefix == k[0]]
    if len(probs) == 0:
        # return highest probability of unigram ('the')
        return 0, 'the', 0
    sorted_probs = sorted(probs, key=lambda tup: tup[2])
    return sorted_probs[-1]


def complete_sentence(sentence, bigrams_dict, prefix_dist_dict):

    tokens = '<s> {}'.format(sentence).split(' ')
    tokens_clean = []
    for i, _ in enumerate(tokens):
        if tokens[i] == '<unk':
            tokens_clean.append('{} {}'.format(tokens[i], tokens[i + 1]))
        elif 'w="' == tokens[i][:3]:
            pass
        else:
            tokens_clean.append(tokens[i])
    accuracy = 0.0
    completed_sentence = ['<s>']
    n_unk = 0.0
    n_true = 0.0
    highest_prob_bigrams = bigrams_dict
    for i, token in enumerate(tokens_clean):
        if i == len(tokens_clean) - 1:
            break
        if '<unk w="' in tokens_clean[i + 1]:
            # print('Processing unk: {}'.format(tokens_clean[i + 1]))
            n_unk += 1
            # highest_prob_bigram = get_highest_prob_bigram(tokens_clean[i], bigram_dict)

            try:
                if '<unk w="' in tokens_clean[i]:
                    prefix = re.search('<unk w="(.*)"/>', tokens_clean[i]).group(1)
                    highest_prob_bigram = highest_prob_bigrams[prefix][0]
                else:
                    prefix = tokens_clean[i]
                    highest_prob_bigram = highest_prob_bigrams[prefix][0]
            except KeyError:
                highest_prob_bigram = (0, 'the')
                prefix = tokens_clean[i]

            if '<unk w="{}"/>'.format(highest_prob_bigram[1]) == tokens_clean[i + 1]:
                n_true += 1
                good_suffix = True
            else:
                good_suffix = False

            if prefix in prefix_dist_dict:
                prefix_dist_dict[prefix]['unk'] += 1.0
                if good_suffix:
                    prefix_dist_dict[prefix]['true'] += 1.0
            else:
                if good_suffix:
                    prefix_dist_dict[prefix] = {'unk': 1.0, 'true': 1.0}
                else:
                    prefix_dist_dict[prefix] = {'unk': 1.0, 'true': 0.0}

            completed_sentence.append('<unk w="{}"/>'.format(highest_prob_bigram[1]))

        else:
            completed_sentence.append(tokens_clean[i + 1])
    return completed_sentence, n_unk, n_true, prefix_dist_dict


def evaluate_test_set(test_file, prefix_dist_dict):
    """
    test_file = '/home/ashbylepoc/tmp/nlp_dev_1/unk-europarl-v7.fi-en-u10.en'
    test_file = '/run/media/ashbylepoc/ff112aea-f91a-4fc7-a80b-4f8fa50d41f3/tmp/data/nlp_dev_1/en/unk-europarl-v7.fi-en-u5c.en'

    :param test_file:
    :param bigram_dict:
    :return:
    """
    lines = open(test_file, 'r').readlines()
    total_true = 0.0
    total_unk = 0.0
    bigram_dict = compute_probs(knp_to_dict())
    for i, line in enumerate(lines):
        completed_sentence, n_unk, n_true, prefix_dist_dict = complete_sentence(line, bigram_dict, prefix_dist_dict)
        print('Processing sentence {}/1000 - Accuracy: {}'.format(i, float(n_true) / float(n_unk)))
        total_true += n_true
        total_unk += n_unk
    return total_true, total_unk, prefix_dist_dict


def visualizing_failures(prefix_dist_dict):
    """
    To visualize failures, we neeed to do:
    1. Compute accuracy of the algo for each prefix
    2. Find the prefix that do well and those who do bad
    3. Compute the probability distribution of each prefix
    4. Compute metadata for each prefix in their respective
        sentence e.g. "Hello" might do well in short sentence
        but not in long sentences or with a sentence with
        many unks but not with a low level of unks
    5.1 For the prefix that did not work well <0.2 acc:
            compare their probability distribution on a graph
            where x: prefix, y: probability distribution in
            color
    :return:
    """
    bigrams = knp_to_dict()
    unigrams = {}
    for bigram in bigrams:
        if bigram[0] in unigrams:
            unigrams[bigram[0]].append((bigram, bigrams[bigram]))
        else:
            unigrams[bigram[0]] = [(bigram, bigrams[bigram])]
    unigram_clean = {}
    for unigram in unigrams:
        sort = sorted(unigrams[unigram], key= lambda x: x[1])
        # unigram_clean[unigram] = sort[-1]
        unigram_clean[unigram] = sort

    prefix_dist_dict = {}
    with open('results/KN_results.txt', 'a') as f:
        for test_set in test_sets:
            total_true, total_unk, prefix_dist_dict = evaluate_test_set(test_set, prefix_dist_dict)
            accuracy = float(total_true) / float(total_unk)
            f.write('processed file: {} -- accuracy: {}\n'.format(test_set, accuracy))
            print('processed file: {} -- accuracy: {}'.format(test_set, accuracy))

    accuracies = {}
    for prefix in prefix_dist_dict:
        prefix_unk, prefix_true = prefix_dist_dict[prefix]['unk'],\
                                  prefix_dist_dict[prefix]['true']
        accuracies[prefix] = {'acc': prefix_true / prefix_unk,
                              'n_unk': prefix_unk}
    i = 0
    top_prefix = []
    for accuracy in accuracies:
        if accuracies[accuracy]['acc'] >= 0.80 and accuracies[accuracy]['n_unk'] > 10:
            i += 1
            top_prefix.append((accuracy, [np.exp(t[1]) for t in reversed(unigram_clean[accuracy][-10:])]))
            print('prefix: {} -- n_unk: {} -- n_suffixes: {}'
                  ' -- difference 1st and 2nd suffix prob: {}'.format(accuracy, accuracies[accuracy]['n_unk'],
                                                                     len(unigram_clean[accuracy]),
                                                                  np.exp(unigram_clean[accuracy][-1][1]) - np.exp(unigram_clean[accuracy][-2][1])))
    print(i)
    worst_prefix = []
    i = 0
    for accuracy in accuracies:
        if accuracies[accuracy]['acc'] <= 0.20 and accuracies[accuracy]['n_unk'] > 10:
            i += 1
            worst_prefix.append((accuracy, [np.exp(t[1]) for t in reversed(unigram_clean[accuracy][-10:])]))

            print('prefix: {} -- n_unk: {} -- n_suffixes: {}'
                  ' -- difference 1st and 2nd suffix prob: {}'.format(accuracy, accuracies[accuracy]['n_unk'],
                                                                     len(unigram_clean[accuracy]),
                                                                  np.exp(unigram_clean[accuracy][-1][1]) - np.exp(unigram_clean[accuracy][-2][1])))
    print(i)

    # Build graph using the top 10 choices for the suffix using the best and worst performing prefix
    top_10_suffixes = [i + 1 for i in range(10)]
    top_10_prefix = [t[0] for t in top_prefix[:10]]
    worst_10_prefix = [t[0] for t in worst_prefix[:10]]
    probs_top = np.zeros(shape=(len(top_prefix), 10))
    for j, t in enumerate(top_prefix):
        pp = np.zeros(shape=10)
        # pp = []
        for i, p in enumerate(t[1]):
            pp[i] = p
        probs_top[j, :] = pp
        # probs_top.append(pp)

    fig, ax = plt.subplots()
    pp_tops = probs_top.transpose(1, 0)
    im = ax.imshow(pp_tops[:, :10])
    ax.set_xticks(np.arange(len(top_10_prefix)))
    ax.set_yticks(np.arange(len(top_10_suffixes)))
    ax.set_xticklabels(top_10_prefix)
    ax.set_yticklabels(top_10_suffixes)
    # ax.set_xticklabels(top_10_suffixes)
    # ax.set_yticklabels(top_10_prefix)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(len(top_10_suffixes)):
        for j in range(len(top_10_prefix)):
            ppp = str(pp_tops[i, j])[:3]
            ppp = float(ppp)
            text = ax.text(j, i, ppp,
                           ha="center", va="center", color="w")
    ax.set_title("")
    fig.tight_layout()
    plt.show()

    # Worse prefixes
    probs_worst = np.zeros(shape=(len(worst_prefix), 10))
    for j, t in enumerate(worst_prefix):
        pp = np.zeros(shape=10)
        for i, p in enumerate(t[1]):
            pp[i] = p
        probs_worst[j, :] = pp
    fig, ax = plt.subplots()
    pp_tops = probs_worst.transpose(1, 0)
    im = ax.imshow(pp_tops[:, :10])
    ax.set_xticks(np.arange(len(worst_10_prefix)))
    ax.set_yticks(np.arange(len(top_10_suffixes)))
    ax.set_xticklabels(worst_10_prefix)
    ax.set_yticklabels(top_10_suffixes)
    # ax.set_xticklabels(top_10_suffixes)
    # ax.set_yticklabels(top_10_prefix)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(len(top_10_suffixes)):
        for j in range(len(worst_10_prefix)):
            ppp = str(pp_tops[i, j])[:4]
            ppp = float(ppp)
            text = ax.text(j, i, ppp,
                           ha="center", va="center", color="w")
    ax.set_title("")
    fig.tight_layout()
    plt.show()


if __name__=='__main__':
    # Compute accuracies with KN on all the test sets
    # Here we can talk about the long time dependencies between the unks
    # the errors are compounded because the algo can only look at the last
    # token to make a prediction
    # When does the algo fail? We would expect it to fail when the prefix has many different
    # suffix possible (the algo's confidence is not very high since it has multiple different choices)
    # and to do good when it is confident (not many different choices and high probability for the first one)
    # => To verify this we will compute the algo's "probability distribution" vs the overall accuracy
    # it has for each prefix in the test set
    prefix_dist_dict = {}
    with open('results/KN_results.txt', 'a') as f:
        for test_set in test_sets:
            total_true, total_unk, prefix_dist_dict = evaluate_test_set(test_set, prefix_dist_dict)
            accuracy = float(total_true) / float(total_unk)
            f.write('processed file: {} -- accuracy: {}\n'.format(test_set, accuracy))
            print('processed file: {} -- accuracy: {}'.format(test_set, accuracy))
