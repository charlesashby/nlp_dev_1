import numpy as np
import os, csv, pickle, time
from util.utils import *
import tensorflow as tf
from tensorflow.contrib import rnn
from gensim.models import KeyedVectors

# data_path = '/home/ashbylepoc/tmp/nlp_dev_1'
data_path = "/run/media/ashbylepoc/ff112aea-f91a-4fc7-a80b-4f8fa50d41f3/tmp/data/nlp_dev_1/"
SAVE_PATH = os.path.join(data_path, 'checkpoints_hist_lstm')
word2vec_dir = "/run/media/ashbylepoc/ff112aea-f91a-4fc" \
               "7-a80b-4f8fa50d41f3/tmp/data/word2vec"


# Lancaster Stemmer is used to remove tenses
# from words
# st = LancasterStemmer()
model = KeyedVectors.load_word2vec_format(
    '{}/GoogleNews-vectors-negative300.bin'.format(word2vec_dir),
    binary=True)


def get_word_distance(w1, w2):
    try:
        vector = model.distance(w1, w2)
        return vector
    except KeyError:
        return None

# Accuracies lstmpc10kn12-6 1024 rnn size checkpoint: {}/lstmpc10kn12-6_35000_3/lstm
# Acc: 0.379237294197 - Top 3 Acc: 0.5 - % Perfect Sentence Prediction %: 0.232862903226 - Word2Vec Distance: 0.418926745806- percentage truncated: 5 - contiguous words: True
# Acc: 0.35130494833 - Top 3 Acc: 0.482486277819 - % Perfect Sentence Prediction %: 0.13741223671 - Word2Vec Distance: 0.434668019139- percentage truncated: 10 - contiguous words: True
# Acc: 0.337271332741 - Top 3 Acc: 0.467035055161 - % Perfect Sentence Prediction %: 0.0451807228916 - Word2Vec Distance: 0.443663030218- percentage truncated: 20 - contiguous words: True
# Acc: 0.300570547581 - Top 3 Acc: 0.425311207771 - % Perfect Sentence Prediction %: 0.014014014014 - Word2Vec Distance: 0.47334796534- percentage truncated: 30 - contiguous words: True
# Acc: 0.266157120466 - Top 3 Acc: 0.391931116581 - % Perfect Sentence Prediction %: 0.003 - Word2Vec Distance: 0.49405699517- percentage truncated: 40 - contiguous words: True
# Acc: 0.392613649368 - Top 3 Acc: 0.5215908885 - % Perfect Sentence Prediction %: 0.251769464105 - Word2Vec Distance: 0.406568736984- percentage truncated: 5 - contiguous words: False
# Acc: 0.38968372345 - Top 3 Acc: 0.51807230711 - % Perfect Sentence Prediction %: 0.15625 - Word2Vec Distance: 0.40862522605- percentage truncated: 10 - contiguous words: False
# Acc: 0.385116904974 - Top 3 Acc: 0.52360612154 - % Perfect Sentence Prediction %: 0.065130260521 - Word2Vec Distance: 0.410183936527- percentage truncated: 20 - contiguous words: False
# Acc: 0.361583769321 - Top 3 Acc: 0.501472532749 - % Perfect Sentence Prediction %: 0.021021021021 - Word2Vec Distance: 0.429161041317- percentage truncated: 30 - contiguous words: False
# Acc: 0.360232055187 - Top 3 Acc: 0.500659286976 - % Perfect Sentence Prediction %: 0.00900900900901 - Word2Vec Distance: 0.428067671148- percentage truncated: 40 - contiguous words: False

# Accuracies lstmpc10k8-4n15 1024 rnn size checkpoint: {}/lstmpc10k8-4n15_60000_2/lstm
# Acc: 0.422669500113 - Top 3 Acc: 0.566737294197 - % Perfect Sentence Prediction %: 0.291330645161 - Word2Vec Distance: 0.374387629588- percentage truncated: 5 - contiguous words: True
# Acc: 0.393200546503 - Top 3 Acc: 0.540178596973 - % Perfect Sentence Prediction %: 0.175526579739 - Word2Vec Distance: 0.394630021447- percentage truncated: 10 - contiguous words: True
# Acc: 0.366806387901 - Top 3 Acc: 0.50304877758 - % Perfect Sentence Prediction %: 0.0532128514056 - Word2Vec Distance: 0.406598689428- percentage truncated: 20 - contiguous words: True
# Acc: 0.325337141752 - Top 3 Acc: 0.462136924267 - % Perfect Sentence Prediction %: 0.026026026026 - Word2Vec Distance: 0.438015774284- percentage truncated: 30 - contiguous words: True
# Acc: 0.290924936533 - Top 3 Acc: 0.430727541447 - % Perfect Sentence Prediction %: 0.004 - Word2Vec Distance: 0.453849910507- percentage truncated: 40 - contiguous words: True
# Acc: 0.419318169355 - Top 3 Acc: 0.563068211079 - % Perfect Sentence Prediction %: 0.279069767442 - Word2Vec Distance: 0.380643926809- percentage truncated: 5 - contiguous words: False
# Acc: 0.417921692133 - Top 3 Acc: 0.56588858366 - % Perfect Sentence Prediction %: 0.17439516129 - Word2Vec Distance: 0.386349441458- percentage truncated: 10 - contiguous words: False
# Acc: 0.428732007742 - Top 3 Acc: 0.567895710468 - % Perfect Sentence Prediction %: 0.0901803607214 - Word2Vec Distance: 0.368017484823- percentage truncated: 20 - contiguous words: False
# Acc: 0.403468579054 - Top 3 Acc: 0.549901843071 - % Perfect Sentence Prediction %: 0.04004004004 - Word2Vec Distance: 0.389278903439- percentage truncated: 30 - contiguous words: False
# Acc: 0.398074895144 - Top 3 Acc: 0.548523187637 - % Perfect Sentence Prediction %: 0.014014014014 - Word2Vec Distance: 0.392197981104- percentage truncated: 40 - contiguous words: False

# Accuracies lstmpc10kn8-4 1024 rnn size checkpoint: {}/lstmpc10kn8-4_30000_3/lstm
# Acc: 0.396716088057 - Top 3 Acc: 0.538665235043 - % Perfect Sentence Prediction %: 0.270161290323 - Word2Vec Distance: 0.402273442771- percentage truncated: 5 - contiguous words: True
# Acc: 0.365384608507 - Top 3 Acc: 0.509958803654 - % Perfect Sentence Prediction %: 0.154463390171 - Word2Vec Distance: 0.421796274642- percentage truncated: 10 - contiguous words: True
# Acc: 0.34222561121 - Top 3 Acc: 0.488376528025 - % Perfect Sentence Prediction %: 0.0542168674699 - Word2Vec Distance: 0.440924289492- percentage truncated: 20 - contiguous words: True
# Acc: 0.306405603886 - Top 3 Acc: 0.441779047251 - % Perfect Sentence Prediction %: 0.022022022022 - Word2Vec Distance: 0.462352109482- percentage truncated: 30 - contiguous words: True
# Acc: 0.27699303627 - Top 3 Acc: 0.409539461136 - % Perfect Sentence Prediction %: 0.004 - Word2Vec Distance: 0.479899442583- percentage truncated: 40 - contiguous words: True
# Acc: 0.378409087658 - Top 3 Acc: 0.523295462132 - % Perfect Sentence Prediction %: 0.23862487361 - Word2Vec Distance: 0.414210305983- percentage truncated: 5 - contiguous words: False
# Acc: 0.391942769289 - Top 3 Acc: 0.533885538578 - % Perfect Sentence Prediction %: 0.173387096774 - Word2Vec Distance: 0.413066745772- percentage truncated: 10 - contiguous words: False
# Acc: 0.407598912716 - Top 3 Acc: 0.540692448616 - % Perfect Sentence Prediction %: 0.0761523046092 - Word2Vec Distance: 0.392466574489- percentage truncated: 20 - contiguous words: False
# Acc: 0.365674078465 - Top 3 Acc: 0.518488228321 - % Perfect Sentence Prediction %: 0.025025025025 - Word2Vec Distance: 0.425198791012- percentage truncated: 30 - contiguous words: False
# Acc: 0.372758448124 - Top 3 Acc: 0.519646644592 - % Perfect Sentence Prediction %: 0.011011011011 - Word2Vec Distance: 0.420843147938- percentage truncated: 40 - contiguous words: False

# Accuracies lstmpc10k5-3 1024 rnn size checkpoint: {}/lstmpc10k5-3_65000_1
# Acc: 0.374470353127 - Top 3 Acc: 0.520656764507 - % Perfect Sentence Prediction %: 0.247983870968 - Word2Vec Distance: 0.399290031396- percentage truncated: 5 - contiguous words: True
# Acc: 0.344436824322 - Top 3 Acc: 0.495879113674 - % Perfect Sentence Prediction %: 0.140421263791 - Word2Vec Distance: 0.416855247452- percentage truncated: 10 - contiguous words: True
# Acc: 0.31707316637 - Top 3 Acc: 0.443978667259 - % Perfect Sentence Prediction %: 0.0491967871486 - Word2Vec Distance: 0.429121400069- percentage truncated: 20 - contiguous words: True
# Acc: 0.268801867962 - Top 3 Acc: 0.39120849967 - % Perfect Sentence Prediction %: 0.018018018018 - Word2Vec Distance: 0.458958742851- percentage truncated: 30 - contiguous words: True
# Acc: 0.228134676814 - Top 3 Acc: 0.347426474094 - % Perfect Sentence Prediction %: 0.004 - Word2Vec Distance: 0.483324783903- percentage truncated: 40 - contiguous words: True
# Acc: 0.392045468092 - Top 3 Acc: 0.536931812763 - % Perfect Sentence Prediction %: 0.247724974722 - Word2Vec Distance: 0.398565717702- percentage truncated: 5 - contiguous words: False
# Acc: 0.396837353706 - Top 3 Acc: 0.529367446899 - % Perfect Sentence Prediction %: 0.162298387097 - Word2Vec Distance: 0.400493932203- percentage truncated: 10 - contiguous words: False
# Acc: 0.392311155796 - Top 3 Acc: 0.536870479584 - % Perfect Sentence Prediction %: 0.0791583166333 - Word2Vec Distance: 0.392680842123- percentage truncated: 20 - contiguous words: False
# Acc: 0.362892657518 - Top 3 Acc: 0.509816765785 - % Perfect Sentence Prediction %: 0.023023023023 - Word2Vec Distance: 0.414618320169- percentage truncated: 30 - contiguous words: False
# Acc: 0.353507369757 - Top 3 Acc: 0.5 - % Perfect Sentence Prediction %: 0.00800800800801 - Word2Vec Distance: 0.417407519108- percentage truncated: 40 - contiguous words: False

# Accuracies lstmp10k5-3 1024 rnn size checkpoint: {}/lstmp_25000_1/lstm
# Acc: 0.356991529465 - Top 3 Acc: 0.477754235268 - % Perfect Sentence Prediction %: 0.243951612903 - Word2Vec Distance: 0.416720691505- percentage truncated: 5 - contiguous words: True
# Acc: 0.324175834656 - Top 3 Acc: 0.448832422495 - % Perfect Sentence Prediction %: 0.131394182548 - Word2Vec Distance: 0.430339792177- percentage truncated: 10 - contiguous words: True
# Acc: 0.29382622242 - Top 3 Acc: 0.408727139235 - % Perfect Sentence Prediction %: 0.039156626506 - Word2Vec Distance: 0.453974930073- percentage truncated: 20 - contiguous words: True
# Acc: 0.242738589644 - Top 3 Acc: 0.360088169575 - % Perfect Sentence Prediction %: 0.011011011011 - Word2Vec Distance: 0.483161478492- percentage truncated: 30 - contiguous words: True
# Acc: 0.203076630831 - Top 3 Acc: 0.314434975386 - % Perfect Sentence Prediction %: 0.003 - Word2Vec Distance: 0.506129550714- percentage truncated: 40 - contiguous words: True
# Acc: 0.366477280855 - Top 3 Acc: 0.491477280855 - % Perfect Sentence Prediction %: 0.234580384226 - Word2Vec Distance: 0.403012182537- percentage truncated: 5 - contiguous words: False
# Acc: 0.356551200151 - Top 3 Acc: 0.486445784569 - % Perfect Sentence Prediction %: 0.137096774194 - Word2Vec Distance: 0.42811878953- percentage truncated: 10 - contiguous words: False
# Acc: 0.35948741436 - Top 3 Acc: 0.486285984516 - % Perfect Sentence Prediction %: 0.063126252505 - Word2Vec Distance: 0.412663765897- percentage truncated: 20 - contiguous words: False
# Acc: 0.328534036875 - Top 3 Acc: 0.456806272268 - % Perfect Sentence Prediction %: 0.019019019019 - Word2Vec Distance: 0.441819779791- percentage truncated: 30 - contiguous words: False
# Acc: 0.324498951435 - Top 3 Acc: 0.456882923841 - % Perfect Sentence Prediction %: 0.004004004004 - Word2Vec Distance: 0.442030069032- percentage truncated: 40 - contiguous words: False

# TODO: Accuracies lstmp10k5-3v1 512 rnn size checkpoint: {}/lstmp10k5-3v1
# Acc: 0.300317794085 - Top 3 Acc: 0.427966088057 - % Perfect Sentence Prediction %: 0.195564516129 - Word2Vec Distance: 0.422023460347- percentage truncated: 5 - contiguous words: True
# Acc: 0.283310443163 - Top 3 Acc: 0.401785701513 - % Perfect Sentence Prediction %: 0.108324974925 - Word2Vec Distance: 0.416466967558- percentage truncated: 10 - contiguous words: True
# Acc: 0.25933688879 - Top 3 Acc: 0.369664639235 - % Perfect Sentence Prediction %: 0.035140562249 - Word2Vec Distance: 0.432371534192- percentage truncated: 20 - contiguous words: True
# Acc: 0.215767636895 - Top 3 Acc: 0.322095423937 - % Perfect Sentence Prediction %: 0.00900900900901 - Word2Vec Distance: 0.458571221689- percentage truncated: 30 - contiguous words: True
# Acc: 0.185952007771 - Top 3 Acc: 0.283862233162 - % Perfect Sentence Prediction %: 0 - Word2Vec Distance: 0.468477513726- percentage truncated: 40 - contiguous words: True
# Acc: 0.310795456171 - Top 3 Acc: 0.438068181276 - % Perfect Sentence Prediction %: 0.193124368049 - Word2Vec Distance: 0.425376431678- percentage truncated: 5 - contiguous words: False
# Acc: 0.300075292587 - Top 3 Acc: 0.430346399546 - % Perfect Sentence Prediction %: 0.10685483871 - Word2Vec Distance: 0.427096310288- percentage truncated: 10 - contiguous words: False
# Acc: 0.311375886202 - Top 3 Acc: 0.42985612154 - % Perfect Sentence Prediction %: 0.062124248497 - Word2Vec Distance: 0.406663685296- percentage truncated: 20 - contiguous words: False
# Acc: 0.283376961946 - Top 3 Acc: 0.409522265196 - % Perfect Sentence Prediction %: 0.014014014014 - Word2Vec Distance: 0.431293488649- percentage truncated: 30 - contiguous words: False
# Acc: 0.285996824503 - Top 3 Acc: 0.414820671082 - % Perfect Sentence Prediction %: 0.00500500500501 - Word2Vec Distance: 0.426785134087- percentage truncated: 40 - contiguous words: False

# Accuracies lstmpc10k8-4 1024 rnn size checkpoint: {}/lstmpc10k8-4_5000_3/lstm
# Acc: 0.387711852789 - Top 3 Acc: 0.525953412056 - % Perfect Sentence Prediction %: 0.259072580645 - Word2Vec Distance: 0.396370822778- percentage truncated: 5 - contiguous words: True
# Acc: 0.357486277819 - Top 3 Acc: 0.499656587839 - % Perfect Sentence Prediction %: 0.148445336008 - Word2Vec Distance: 0.423228611041- percentage truncated: 10 - contiguous words: True
# Acc: 0.319931387901 - Top 3 Acc: 0.44683688879 - % Perfect Sentence Prediction %: 0.0532128514056 - Word2Vec Distance: 0.442662197703- percentage truncated: 20 - contiguous words: True
# Acc: 0.270487546921 - Top 3 Acc: 0.390819489956 - % Perfect Sentence Prediction %: 0.017017017017 - Word2Vec Distance: 0.46778982119- percentage truncated: 30 - contiguous words: True
# Acc: 0.225619196892 - Top 3 Acc: 0.339686542749 - % Perfect Sentence Prediction %: 0.001 - Word2Vec Distance: 0.49024692466- percentage truncated: 40 - contiguous words: True
# Acc: 0.410227268934 - Top 3 Acc: 0.542045474052 - % Perfect Sentence Prediction %: 0.268958543984 - Word2Vec Distance: 0.388805008143- percentage truncated: 5 - contiguous words: False
# Acc: 0.396460831165 - Top 3 Acc: 0.538027107716 - % Perfect Sentence Prediction %: 0.163306451613 - Word2Vec Distance: 0.406462934473- percentage truncated: 10 - contiguous words: False
# Acc: 0.397706836462 - Top 3 Acc: 0.533722996712 - % Perfect Sentence Prediction %: 0.0731462925852 - Word2Vec Distance: 0.396978020995- percentage truncated: 20 - contiguous words: False
# Acc: 0.362238228321 - Top 3 Acc: 0.508671462536 - % Perfect Sentence Prediction %: 0.026026026026 - Word2Vec Distance: 0.420171889587- percentage truncated: 30 - contiguous words: False
# Acc: 0.355748951435 - Top 3 Acc: 0.50237339735 - % Perfect Sentence Prediction %: 0.00700700700701 - Word2Vec Distance: 0.426407816611- percentage truncated: 40 - contiguous words: False

# Accuracies lstmp5k8-4 1024 rnn size checkpoint: {}/lstmp8-4_35000_1/lstm
# Acc: 0.324682205915 - Top 3 Acc: 0.477224588394 - % Perfect Sentence Prediction: 0.210685483871 - Word2Vec Distance: 0.392288033616 - % unknown: 0.101694915254- percentage truncated: 5 - contiguous words: True
# Acc: 0.292925834656 - Top 3 Acc: 0.448145598173 - % Perfect Sentence Prediction: 0.119358074223 - Word2Vec Distance: 0.409599695082 - % unknown: 0.101304945055- percentage truncated: 10 - contiguous words: True
# Acc: 0.26867377758 - Top 3 Acc: 0.410632610321 - % Perfect Sentence Prediction: 0.0301204819277 - Word2Vec Distance: 0.425492357208 - % unknown: 0.0920350609756- percentage truncated: 20 - contiguous words: True
# Acc: 0.225103735924 - Top 3 Acc: 0.363848537207 - % Perfect Sentence Prediction: 0.00900900900901 - Word2Vec Distance: 0.454694026971 - % unknown: 0.0896006224066- percentage truncated: 30 - contiguous words: True
# Acc: 0.197368428111 - Top 3 Acc: 0.331656336784 - % Perfect Sentence Prediction: 0.001 - Word2Vec Distance: 0.471582435388 - % unknown: 0.0920085139319- percentage truncated: 40 - contiguous words: True
# Acc: 0.331250011921 - Top 3 Acc: 0.470454543829 - % Perfect Sentence Prediction: 0.202224469161 - Word2Vec Distance: 0.406257629914 - % unknown: 0.0960227272727- percentage truncated: 5 - contiguous words: False
# Acc: 0.318900614977 - Top 3 Acc: 0.475150614977 - % Perfect Sentence Prediction: 0.108870967742 - Word2Vec Distance: 0.406756629094 - % unknown: 0.0914909638554- percentage truncated: 10 - contiguous words: False
# Acc: 0.322392076254 - Top 3 Acc: 0.473246395588 - % Perfect Sentence Prediction: 0.0551102204409 - Word2Vec Distance: 0.39326681936 - % unknown: 0.0948741007194- percentage truncated: 20 - contiguous words: False
# Acc: 0.291230380535 - Top 3 Acc: 0.447480380535 - % Perfect Sentence Prediction: 0.017017017017 - Word2Vec Distance: 0.421207720439 - % unknown: 0.0983311518325- percentage truncated: 30 - contiguous words: False
# Acc: 0.291007369757 - Top 3 Acc: 0.449235230684 - % Perfect Sentence Prediction: 0.004004004004 - Word2Vec Distance: 0.421188221114 - % unknown: 0.0967827004219- percentage truncated: 40 - contiguous words: False

# TODO: Accuracies lstmp10k8-4 1024 rnn size checkpoint: {}/lstmp10k8-4_ _ /lstm
# Acc: 0.309851706028 - Top 3 Acc: 0.450211852789 - % Perfect Sentence Prediction: 0.197580645161 - Word2Vec Distance: 0.430822229725 - % unknown: 0.0688559322034- percentage truncated: 5 - contiguous words: True
# Acc: 0.274038463831 - Top 3 Acc: 0.421703308821 - % Perfect Sentence Prediction: 0.102306920762 - Word2Vec Distance: 0.453448112632 - % unknown: 0.0721153846154- percentage truncated: 10 - contiguous words: True
# Acc: 0.2578125 - Top 3 Acc: 0.385289639235 - % Perfect Sentence Prediction: 0.0301204819277 - Word2Vec Distance: 0.469057018414 - % unknown: 0.0611661585366- percentage truncated: 20 - contiguous words: True
# Acc: 0.220306023955 - Top 3 Acc: 0.337136924267 - % Perfect Sentence Prediction: 0.00800800800801 - Word2Vec Distance: 0.490926168959 - % unknown: 0.0619813278008- percentage truncated: 30 - contiguous words: True
# Acc: 0.183436527848 - Top 3 Acc: 0.29837462306 - % Perfect Sentence Prediction: 0.001 - Word2Vec Distance: 0.51173191878 - % unknown: 0.062790247678- percentage truncated: 40 - contiguous words: True
# Acc: 0.315340906382 - Top 3 Acc: 0.44829544425 - % Perfect Sentence Prediction: 0.19514661274 - Word2Vec Distance: 0.446305431693 - % unknown: 0.0647727272727- percentage truncated: 5 - contiguous words: False
# Acc: 0.31174698472 - Top 3 Acc: 0.440135538578 - % Perfect Sentence Prediction: 0.113911290323 - Word2Vec Distance: 0.446297832442 - % unknown: 0.0643825301205- percentage truncated: 10 - contiguous words: False
# Acc: 0.310926258564 - Top 3 Acc: 0.444244593382 - % Perfect Sentence Prediction: 0.0480961923848 - Word2Vec Distance: 0.436733404545 - % unknown: 0.0658723021583- percentage truncated: 20 - contiguous words: False
# Acc: 0.284522265196 - Top 3 Acc: 0.416884809732 - % Perfect Sentence Prediction: 0.011011011011 - Word2Vec Distance: 0.463293381694 - % unknown: 0.0696989528796- percentage truncated: 30 - contiguous words: False
# Acc: 0.277821719646 - Top 3 Acc: 0.417062222958 - % Perfect Sentence Prediction: 0.003003003003 - Word2Vec Distance: 0.462371457213 - % unknown: 0.0685654008439- percentage truncated: 40 - contiguous words: False

# Accuracies lstmpcn10k5-3 1024 rnn checkpoint: lstmpcn10k5-3_55000_1/lstm
# Acc: 0.399364411831 - Top 3 Acc: 0.536016941071 - % Perfect Sentence Prediction: 0.259072580645 - Word2Vec Distance: 0.390036910671 - % unknown: 0.03125- percentage truncated: 5 - contiguous words: True
# Acc: 0.37603020668 - Top 3 Acc: 0.512706041336 - % Perfect Sentence Prediction: 0.165496489468 - Word2Vec Distance: 0.399547556326 - % unknown: 0.0288461538462- percentage truncated: 10 - contiguous words: True
# Acc: 0.35003811121 - Top 3 Acc: 0.491806387901 - % Perfect Sentence Prediction: 0.0502008032129 - Word2Vec Distance: 0.419634016383 - % unknown: 0.0222942073171- percentage truncated: 20 - contiguous words: True
# Acc: 0.31846472621 - Top 3 Acc: 0.451244801283 - % Perfect Sentence Prediction: 0.021021021021 - Word2Vec Distance: 0.440405344641 - % unknown: 0.0238589211618- percentage truncated: 30 - contiguous words: True
# Acc: 0.287828952074 - Top 3 Acc: 0.42037537694 - % Perfect Sentence Prediction: 0.004 - Word2Vec Distance: 0.460738747355 - % unknown: 0.0240905572755- percentage truncated: 40 - contiguous words: True
# Acc: 0.408522725105 - Top 3 Acc: 0.537500023842 - % Perfect Sentence Prediction: 0.266936299292 - Word2Vec Distance: 0.38177542069 - % unknown: 0.03125- percentage truncated: 5 - contiguous words: False
# Acc: 0.402484953403 - Top 3 Acc: 0.541415691376 - % Perfect Sentence Prediction: 0.172379032258 - Word2Vec Distance: 0.392176892552 - % unknown: 0.0252259036145- percentage truncated: 10 - contiguous words: False
# Acc: 0.40265288949 - Top 3 Acc: 0.545638501644 - % Perfect Sentence Prediction: 0.0811623246493 - Word2Vec Distance: 0.386340634871 - % unknown: 0.0285521582734- percentage truncated: 20 - contiguous words: False
# Acc: 0.378435850143 - Top 3 Acc: 0.525196313858 - % Perfect Sentence Prediction: 0.029029029029 - Word2Vec Distance: 0.402234112265 - % unknown: 0.0287958115183- percentage truncated: 30 - contiguous words: False
# Acc: 0.377241551876 - Top 3 Acc: 0.523074865341 - % Perfect Sentence Prediction: 0.015015015015 - Word2Vec Distance: 0.401476402732 - % unknown: 0.0288765822785- percentage truncated: 40 - contiguous words: False


def get_lines(perc_unk, c=False):
    if c:
        test_file = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u{}c.en'.format(perc_unk))
    else:
        test_file = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u{}.en'.format(perc_unk))

    lines_clean = []
    lines = open(test_file, 'r').readlines()
    for line in lines:
        lines_clean.append(tokenize_sentence(line))
    return lines_clean


def build_file(perc_unk, c, file_out, token_dict):
    lines = get_lines(perc_unk, c)
    for pp in preds_indexes:
        lines[pp[1][0]][pp[1][1]] = '<unk w="{}"/>'.format(token_dict[pp[2]])

    clean_lines = []
    for line in lines:
        clean_lines.append(' '.join(t for t in line))

    with open(file_out, 'a') as f:
        for line in clean_lines:
            f.write('{}\n'.format(line))
    return True


def iterate_test_batch(token_dict, perc_unk=10, c=False, batch_size=32,
                       pre=12, suf=6, output_size=10000, clean=True):
    if c:
        test_file = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u{}c.en'.format(perc_unk))
    else:
        test_file = os.path.join(data_path, 'en/unk-europarl-v7.fi-en-u{}.en'.format(perc_unk))

    structured_unks, unk_indexes = structure_file(test_file, suf=suf, pre=pre, clean=clean)

    n_batch = len(structured_unks) / batch_size
    for i in range(n_batch):
        bb_x = []
        yy = []
        for unk in structured_unks[i * batch_size: (i + 1) * batch_size]:
            bb_x.append(unk[0])
            yy.append(unk[1])

        b_x = np.zeros(shape=[batch_size, pre + suf + 1, output_size + 3])
        b_y = np.zeros(shape=[batch_size, output_size + 3])
        for j, line in enumerate(bb_x):
            y_pos = get_token_dict_pos(token_dict, yy[j])
            b_y[j][y_pos] = 1.
            for k, token in enumerate(line):
                if '<unk w=' in token:
                    x_pos = -2
                else:
                    x_pos = get_token_dict_pos(token_dict, token)
                b_x[j][k][x_pos] = 1.
        yield b_x, b_y, unk_indexes[i * batch_size: (i + 1) * batch_size]


if __name__ == '__main__':

    # word2vec:
    with open('clean_token_dict.pickle', 'rb') as f:
        tokens_dict = pickle.load(f)
    sorted_tokens = sorted(tokens_dict.items(), key=lambda item: item[1])
    token_dict = [t[0] for t in sorted_tokens[-10000:]] + ['<s>', '<UNKNOWN>', '<UNK>']

    batch_size = 32
    rnn_size = 1024
    max_n_token_dict = 10000 + 3

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
    cost = - tf.reduce_sum(y * tf.log(tf.clip_by_value(preds, 1e-10, 1.0)), axis=1)
    cost = tf.reduce_mean(cost, axis=0)
    predictions = tf.cast(tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1)), dtype='float32')

    acc = tf.reduce_mean(predictions)

    top3_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(preds, tf.argmax(y, 1), k=3), dtype='float32'))
    sess = tf.Session()
    saver = tf.train.Saver()
    model_path = '{}/lstmpc10k8-4n15_{}_{}/lstm'.format(SAVE_PATH, 60000, 2)
    saver.restore(sess, model_path)

    for c in [True, False]:
        for perc_unk in [5, 10, 20, 30, 40]:
            all_acc = []
            all_top3_acc = []
            preds_indexes = []
            for i, (b_x, b_y, unk_index) in enumerate(iterate_test_batch(token_dict, perc_unk=perc_unk, c=c,
                                                    batch_size=32, pre=8, suf=4, output_size=10000,
                                                    clean=True)):
                preds_, acc_, top3_acc_, predictions_ = sess.run(
                    [preds, acc, top3_acc, predictions], feed_dict={x: b_x, y: b_y})
                preds__ = np.argmax(preds_, axis=1)
                b_y_ = np.argmax(b_y, axis=1)
                all_acc.append(acc_)
                all_top3_acc.append(top3_acc_)
                preds_indexes = preds_indexes + [[p, unk_index[i], preds__[i], b_y_[i]]
                                                 for i, p in enumerate(predictions_)]


            # Build file
            build_file(perc_unk, c, 'results/lstmpc10k8-4n15_{}c_{}u.txt'.format(int(c), perc_unk)
                       , token_dict)

            # Compute sentence accuracies
            acc_sentence = {}
            w2v_dists = []
            n_unknown = 0
            for i, u in enumerate(preds_indexes):
                # token predicted, target token
                t1, t2 = token_dict[preds_indexes[i][2]], token_dict[preds_indexes[i][3]]
                if t2 == '<UNKNOWN>':
                    n_unknown += 1.
                dist = get_word_distance(t1, t2)
                if dist is not None:
                    w2v_dists.append(dist)
                else:
                    if t1 == t2:
                        w2v_dists.append(0.)
                if u[1][0] in acc_sentence:
                    acc_sentence[u[1][0]][1] += 1.
                    if u[0] == 1.:
                        acc_sentence[u[1][0]][0] += 1.
                else:
                    acc_sentence[u[1][0]] = [0., 1.]
                    if u[0] == 1.:
                        acc_sentence[u[1][0]][0] += 1.
            sentence_acc = []
            perfect_sentence = 0

            perc_unknown = n_unknown / len(preds_indexes)

            for i, a in enumerate(acc_sentence):

                accuracy_sentence = acc_sentence[a][0] / acc_sentence[a][1]
                sentence_acc.append(accuracy_sentence)
                if accuracy_sentence == 1.:
                    perfect_sentence += 1.
            print('Acc: {} - Top 3 Acc: {} - % Perfect Sentence Prediction: {} '
                  '- Word2Vec Distance: {} - % unknown: {}'
                  '- percentage truncated: {} - contiguous words: {} '.format(
                np.mean(all_acc), np.mean(all_top3_acc), perfect_sentence / len(sentence_acc),
                np.mean(w2v_dists), perc_unknown, perc_unk, c))
            with open('results/lstmp10k5-3v1_results.txt', 'a') as f:
                f.write('Acc: {} - Top 3 Acc: {} - % Perfect Sentence Prediction: {} '
                          '- Word2Vec Distance: {} - % unknown: {}'
                          '- percentage truncated: {} - contiguous words: {} '.format(
                        np.mean(all_acc), np.mean(all_top3_acc), perfect_sentence / len(sentence_acc),
                        np.mean(w2v_dists), perc_unknown, perc_unk, c))