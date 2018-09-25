# HW2: Plot rank vs. frequency on a log-log scale, 
# and fit parameters for the generalized power law. 
#
# Coded by Md Iftekhar Tanveer (itanveer@cs.rochester.edu)
#
import numpy as np
import matplotlib.pyplot as plt

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


def get_highest_prob_bigram(prefix, bigram_dict):
    probs = [(k[0], k[1], v) for k, v in bigram_dict.items() if prefix == k[0]]
    if len(probs) == 0:
        # return highest probability of unigram ('the')
    sorted_probs = sorted(probs, key=lambda tup: tup[2])
    return sorted_probs[-1]


def complete_sentence(sentence, bigram_dict):
    """
    test_sentence = 'It is important for me to take advantage of the 10th anniversary of this key event in European integration to pay homage , in my turn , to those men who created the <unk w="euro"/> , such as Pierre Werner , Helmut Kohl , François Mitterrand , Jacques <unk w="Delors"/> , Valéry Giscard d 'Estaing and others .'
    :param sentence:
    :param bigram_dict:
    :return:
    """
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
    for i, token in enumerate(tokens_clean):
        if i == len(tokens_clean) - 1:
            break
        if '<unk w="' in tokens_clean[i + 1]:
            # print('Processing unk: {}'.format(tokens_clean[i + 1]))
            n_unk += 1
            highest_prob_bigram = get_highest_prob_bigram(tokens_clean[i], bigram_dict)
            if '<unk w="{}"/>'.format(highest_prob_bigram[1]) == tokens_clean[i + 1]:
                n_true += 1
            completed_sentence.append('<unk w="{}"/>'.format(highest_prob_bigram[1]))
        else:
            completed_sentence.append(tokens_clean[i + 1])
    return completed_sentence, n_unk, n_true


def evaluate_test_set(test_file, bigram_dict):
    """
    test_file = '/home/ashbylepoc/tmp/nlp_dev_1/unk-europarl-v7.fi-en-u10.en'
    :param test_file:
    :param bigram_dict:
    :return:
    """
    lines = open(test_file, 'r').readlines()
    total_true = 0.0
    total_unk = 0.0
    for i, line in enumerate(lines):
        completed_sentence, n_unk, n_true = complete_sentence(line, bigram_dict)
        print('Processing sentence {}/1000 - Accuracy: {}'.format(i, float(n_true) / float(n_unk)))
        total_true += n_true
        total_unk += n_unk
    return total_true, total_unk

if __name__=='__main__':
    main()
