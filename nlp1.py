import numpy as np
import math
import re
from collections import OrderedDict
from scipy.optimize import fmin_l_bfgs_b
import pickle

# split line to list of word_tag
def split_line(line):
    return(re.split(' |\n',line))

# split split word_tag to list of word,tag
def split_word_tag(word_tag):
    return (re.split('_',word_tag))


def get_tags_list(file_path):
    """
        Extract out of text tags
        :param file_path: full path of the file to read
            return a list of all possible tags
    """

    tags_list=[]
    with open('train1.wtag') as f:
        for line in f:
            split_words = split_line(line)
            del split_words[-1]
            for word_idx in range(len(split_words)):
                cur_word, cur_tag = split_word_tag(split_words[word_idx])
                tags_list.append(cur_tag)
    return list(set(tags_list))


class feature_statistics_class():

    def __init__(self, file_path):
        self.n_total_features = 0  # Total number of features accumulated
        self.file_path = file_path

        # Init all features dictionaries
        # key: feature; value: number of appearances
        self.words_tags_count_dict = OrderedDict()
        self.spelling_prefix_count_dict = OrderedDict()
        self.spelling_suffix_count_dict = OrderedDict()
        self.trigram_tags_count_dict = OrderedDict()
        self.bigram_tags_count_dict = OrderedDict()
        self.unigram_tags_count_dict = OrderedDict()

    def get_word_tag_pair_count(self):
        """
            Extract out of text all word/tag pairs
                return all word/tag pairs with index of appearance
        """
        with open(self.file_path) as f:
            for line in f:
                split_words = split_line(line)
                del split_words[-1]
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_word_tag(split_words[word_idx])
                    if (cur_word, cur_tag) not in self.words_tags_count_dict:
                        self.words_tags_count_dict[(cur_word, cur_tag)] = 1
                    else:
                        self.words_tags_count_dict[(cur_word, cur_tag)] += 1
      
    def get_spelling_prefix_count(self):
        """
            Extract out of text all prefix/tag pairs
                return all prefix/tag pairs with index of appearance
        """
        with open(self.file_path) as f:
            for line in f:
                split_words = split_line(line)
                del split_words[-1]
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_word_tag(split_words[word_idx])
                    length = len(cur_word)
                    for i in range(min(4, length)):
                        if (cur_word[:i+1], cur_tag) not in self.spelling_prefix_count_dict:
                            self.spelling_prefix_count_dict[(cur_word[:i+1], cur_tag)] = 1
                        else:
                            self.spelling_prefix_count_dict[(cur_word[:i+1], cur_tag)] += 1
                                        
    def get_spelling_suffix_count(self):
        """
            Extract out of text all suffix/tag pairs
                return all suffix/tag pairs with index of appearance
        """
        with open(self.file_path) as f:
            for line in f:
                split_words = split_line(line)
                del split_words[-1]
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_word_tag(split_words[word_idx])
                    length=len(cur_word)
                    for i in range(min(4, length)):
                        if (cur_word[-i-1:], cur_tag) not in self.spelling_suffix_count_dict:
                            self.spelling_suffix_count_dict[(cur_word[-i-1:], cur_tag)] = 1
                        else:
                            self.spelling_suffix_count_dict[(cur_word[-i-1:], cur_tag)] += 1

    def get_trigram_tags_count(self):
        """
            Extract out of text all trigrams tags
                return all trigrams tags with index of appearance
        """
        with open(self.file_path) as f:
            for line in f:
                split_words = split_line(line)
                del split_words[-1]
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_word_tag(split_words[word_idx])
                    if word_idx > 1:
                        ptag = split_word_tag(split_words[word_idx-1])[1]
                        pptag = split_word_tag(split_words[word_idx-2])[1]
                    elif word_idx == 1:
                        ptag = split_word_tag(split_words[word_idx-1])[1]
                        pptag = '*'
                    else:
                        ptag = '*'
                        pptag = '*'
                    if (pptag, ptag, cur_tag) not in self.trigram_tags_count_dict:
                        self.trigram_tags_count_dict[(pptag,ptag,cur_tag)] = 1
                    else:
                        self.trigram_tags_count_dict[(pptag,ptag,cur_tag)] += 1

    def get_bigram_tags_count(self):
        """
            Extract out of text all bigram tags
                return all bigram tags with index of appearance
        """
        with open(self.file_path) as f:
            for line in f:
                split_words = split_line(line)
                del split_words[-1]
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_word_tag(split_words[word_idx])
                    if word_idx>0:
                        ptag =  split_word_tag(split_words[word_idx-1])[1]
                    else:
                        ptag = '*'
                    if (ptag,cur_tag) not in self.bigram_tags_count_dict:
                        self.bigram_tags_count_dict[(ptag,cur_tag)] = 1
                    else:
                        self.bigram_tags_count_dict[(ptag,cur_tag)] += 1

    def get_unigram_tags_count(self):
        """
            Extract out of text all tags
                return all tags with index of appearance
        """
        with open(self.file_path) as f:
            for line in f:
                split_words = split_line(line)
                del split_words[-1]
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_word_tag(split_words[word_idx])
                    if (cur_tag) not in self.unigram_tags_count_dict:
                        self.unigram_tags_count_dict[(cur_tag)] = 1
                    else:
                        self.unigram_tags_count_dict[(cur_tag)] += 1

    # --- ADD YOURE CODE BELOW --- #


"""### Indexing features 
After getting feature statistics, each feature is given an index to represent it. We include only features that appear more times in text than the lower bound - 'threshold'
"""


class feature2id_class():

    def __init__(self, feature_statistics, threshold, file_path):
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold                    # feature count threshold - empirical count must be higher than this
        self.file_path = file_path

        self.n_total_features = 0                     # Total number of features accumulated
        self.n_tag_pairs = 0                          # Number of Word\Tag pairs features
        self.n_prefix_tag = 0
        self.n_suffix_tag = 0
        self.n_trigram_tags = 0
        self.n_bigram_tags = 0
        self.n_unigram_tags = 0

        # Init all features dictionaries
        self.words_tags_dict = OrderedDict()
        self.prefix_tag_dict = OrderedDict()
        self.suffix_tag_dict = OrderedDict()
        self.trigram_tags_dict = OrderedDict()
        self.bigram_tags_dict = OrderedDict()
        self.unigram_tags_dict = OrderedDict()


    def get_word_tag_pairs(self):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(self.file_path) as f:
            for line in f:
                split_words = split_line(line)
                del split_words[-1]
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_word_tag(split_words[word_idx])
                    if ((cur_word, cur_tag) not in self.words_tags_dict) \
                        and (self.feature_statistics.words_tags_count_dict[(cur_word, cur_tag)] >= self.threshold):
                        self.words_tags_dict[(cur_word, cur_tag)] = self.n_tag_pairs
                        self.n_tag_pairs += 1
        self.n_total_features += self.n_tag_pairs
        

    def get_prefix_tag_pairs(self):
        """
            Extract out of text all prefix/tag pairs
                return all prefix/tag pairs with index of appearance
        """
        with open(self.file_path) as f:
            for line in f:
                split_words = split_line(line)
                del split_words[-1]
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_word_tag(split_words[word_idx])
                    length = len(cur_word)
                    for i in range (min(4,length)):
                        if ((cur_word[:i+1], cur_tag) not in self.prefix_tag_dict) \
                        and (self.feature_statistics.spelling_prefix_count_dict[(cur_word[:i+1], cur_tag)] >= self.threshold):
                            self.prefix_tag_dict[(cur_word[:i+1], cur_tag)] = self.n_total_features + self.n_prefix_tag
                            self.n_prefix_tag += 1
        self.n_total_features = self.n_total_features + self.n_prefix_tag    

    def get_suffix_tag_pairs(self):
        """
            Extract out of text all suffix/tag pairs
                return all suffix/tag pairs with index of appearance
        """
        with open(self.file_path) as f:
            for line in f:
                split_words = split_line(line)
                del split_words[-1]
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_word_tag(split_words[word_idx])
                    length = len(cur_word)
                    for i in range (min(4,length)):
                        if ((cur_word[-i-1:], cur_tag) not in self.suffix_tag_dict) \
                        and (self.feature_statistics.spelling_suffix_count_dict[(cur_word[-i-1:], cur_tag)] >= self.threshold):
                            self.suffix_tag_dict[(cur_word[-i-1:], cur_tag)] = self.n_total_features + self.n_suffix_tag
                            self.n_suffix_tag += 1
        self.n_total_features = self.n_total_features + self.n_suffix_tag       


    def get_trigram_tags_pairs(self):
        """
            Extract out of text all trigram tags
                return all trigram tags with index of appearance
        """
        with open(self.file_path) as f:
            for line in f:
                split_words = split_line(line)
                del split_words[-1]
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_word_tag(split_words[word_idx])
                    if word_idx>1:
                        ptag =  split_word_tag(split_words[word_idx-1])[1]
                        pptag =  split_word_tag(split_words[word_idx-2])[1]
                    elif word_idx == 1:
                        ptag =  split_word_tag(split_words[word_idx-1])[1]
                        pptag = '*'
                    else:
                        ptag = '*'
                        pptag = '*'
                    if ((pptag,ptag,cur_tag) not in self.trigram_tags_dict) \
                        and (self.feature_statistics.trigram_tags_count_dict[(pptag,ptag,cur_tag)] >= self.threshold):
                        self.trigram_tags_dict[(pptag,ptag,cur_tag)] = self.n_total_features + self.n_trigram_tags
                        self.n_trigram_tags += 1
        self.n_total_features = self.n_total_features + self.n_trigram_tags



    def get_bigram_tags_pairs(self):
        """
            Extract out of text all bigram tags
                return all bigram tags with index of appearance
        """
        with open(self.file_path) as f:
            for line in f:
                split_words = split_line(line)
                del split_words[-1]
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_word_tag(split_words[word_idx])
                    if word_idx>0:
                        ptag =  split_word_tag(split_words[word_idx-1])[1]
                    else:
                        ptag = '*'
                    if ((ptag,cur_tag) not in self.bigram_tags_dict) \
                        and (self.feature_statistics.bigram_tags_count_dict[(ptag,cur_tag)] >= self.threshold):
                        self.bigram_tags_dict[(ptag,cur_tag)] = self.n_total_features + self.n_bigram_tags
                        self.n_bigram_tags += 1
        self.n_total_features = self.n_total_features + self.n_bigram_tags


    def get_unigram_tags_pairs(self):
        """
            Extract out of text all tags
                return all tags with index of appearance
        """
        with open(self.file_path) as f:
            for line in f:
                split_words = split_line(line)
                del split_words[-1]
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_word_tag(split_words[word_idx])
                    if ((cur_tag) not in self.unigram_tags_dict) \
                        and (self.feature_statistics.unigram_tags_count_dict[(cur_tag)] >= self.threshold):
                        self.unigram_tags_dict[(cur_tag)] = self.n_total_features + self.n_unigram_tags
                        self.n_unigram_tags += 1
        self.n_total_features = self.n_total_features + self.n_unigram_tags


    # --- ADD YOURE CODE BELOW --- #


"""### Representing input data with features 
After deciding which features to use, we can represent input tokens as sparse feature vectors. This way, a token is represented with a vec with a dimension D, where D is the total amount of features. \
This is done at training step.

### History tuple
We define a tuple which hold all relevant knowledge about the current word, i.e. all that is relevant to extract features for this token.

$$History = (W_{cur}, T_{prev}, T_{next}, T_{cur}, W_{prev}, W_{next}) $$
"""


def represent_input_with_features(history, feature2id):
    """
        Extract feature vector in per a given history
        :param history: touple{word, pptag, ptag, ctag, nword, pword, pptag}
        :param word_tags_dict: word\tag dict
            Return a list with all features that are relevant to the given history
    """
    word = history[0]
    ptag = history[1]
    ntag = history[2]
    ctag = history[3]
    pword = history[4]
    nword = history[5]
    pptag = history[6]
    features = []

    # word-tag
    if (word, ctag) in feature2id.words_tags_dict:
        features.append(feature2id.words_tags_dict[(word, ctag)])

    # prefix-tag
    for i in range(min(4, len(word))):
        if (word[:i + 1], ctag) in feature2id.prefix_tag_dict:
            features.append(feature2id.prefix_tag_dict[(word[:i + 1], ctag)])

    # suffix-tag
    for i in range(min(4, len(word))):
        if (word[-i - 1:], ctag) in feature2id.suffix_tag_dict:
            features.append(feature2id.suffix_tag_dict[(word[-i - 1:], ctag)])

    # trigram tags
    if (pptag, ptag, ctag) in feature2id.trigram_tags_dict:
        features.append(feature2id.trigram_tags_dict[(pptag, ptag, ctag)])

    # bigram tags
    if (ptag, ctag) in feature2id.bigram_tags_dict:
        features.append(feature2id.bigram_tags_dict[(ptag, ctag)])

    # unigram tags
    if (ctag) in feature2id.unigram_tags_dict:
        features.append(feature2id.unigram_tags_dict[(ctag)])

    return features


"""find for each word in the data the relevant features"""
class word_feature_class():

    def __init__(self, feature2id, file_path, tags_list):
        #self.word_tags_dict = feature2id.word_tags_dict
        #self.prefix_tag_dict = feature2id.prefix_tag_dict
        self.feature2id = feature2id
        self.word_features_list = []          #hold for each word in the data the real tag anf the relevant features.
        self.word_tags_features_list = []     #hold for each word in the data the relevant features for each possible tag
        self.tags_list = tags_list          #list of all possible tags
        self.file_path = file_path

    def find_relevant_features(self,):
        """
            Extract for each word the relevant features
        """
        with open(self.file_path) as f:
            for line in f:
                split_words = split_line(line)
                del split_words[-1]
                pptag = '*'
                ptag = '*'
                pword = '*'
                length = len(split_words)
                for word_idx in range(length):
                    if word_idx > 1:
                        ptag = split_word_tag(split_words[word_idx-1])[1]
                        pword = split_word_tag(split_words[word_idx-1])[0]
                        pptag = split_word_tag(split_words[word_idx-2])[1]
                    elif word_idx == 1:
                        ptag = split_word_tag(split_words[word_idx-1])[1]
                        pword = split_word_tag(split_words[word_idx-1])[0]
                    word, ctag = split_word_tag(split_words[word_idx])
                    if word_idx == length-1:
                        ntag = '*'
                        nword = '*'
                    else:
                        ntag = split_word_tag(split_words[word_idx+1])[1]
                        nword = split_word_tag(split_words[word_idx+1])[0]
                    history = (word, ptag, ntag, ctag, pword, nword, pptag)
                    self.word_features_list.append((word, ctag, represent_input_with_features(history, self.feature2id)))
                    word_features_per_tag = []
                    for tag in self.tags_list:
                        history = (word, ptag, ntag, tag, pword, nword, pptag)
                        word_features_per_tag.append(represent_input_with_features(history, self.feature2id))
                        self.word_tags_features_list.append((word, word_features_per_tag))


### Part 2 - Optimization

def calc_objective_per_iter(w_i, word_features_list,word_tags_features_list,num_tags,num_words,num_total_features, lamda):
    """
        Calculate max entropy likelihood for an iterative optimization method
        :param w_i: weights vector in iteration i 
        :param arg_i: arguments passed to this function, such as lambda hyperparameter for regularization
        
            The function returns the Max Entropy likelihood (objective) and the objective gradient
    """

    ## Calculate the terms required for the likelihood and gradient calculations
    ## Try implementing it as efficient as possible, as this is repeated for each iteration of optimization.

    #linear term
    linear_term = 0
    for i in range (num_words):
        for feature in word_features_list[i][2]:
            linear_term += w_i[feature]

    #normalization term
    normalization_term = 0
    for i in range (num_words):
        sum_all_tags = 0
        for j in range (num_tags):
            sum_tag = 0
            for feature in word_tags_features_list[i][1][j]:
                sum_tag += w_i[feature]
            sum_all_tags += math.exp(sum_tag)
        normalization_term += math.log(sum_all_tags)

    #regularization
    regularization = 0
    for i in range(num_total_features):
        regularization += w_i[i]**2
    regularization = 0.5*regularization*lamda

    #empirical counts
    empirical_counts = np.zeros(num_total_features, dtype=np.float32)
    for i in range (num_words):
        for feature in word_features_list[i][2]:
            empirical_counts[feature] += 1

    #expected counts
    expected_counts = np.zeros(num_total_features, dtype=np.float32)
    for i in range (num_words):
        denominator = 0
        for k in range(num_tags):
            sum_tag = 0
            for feature in word_tags_features_list[i][1][k]:
                sum_tag += w_i[feature]
        denominator += math.exp(sum_tag)
        for j in range(num_tags):
            sum_tag = 0
            for feature in word_tags_features_list[i][1][j]:
                sum_tag += w_i[feature]
            numerator = math.exp(sum_tag)
            for feature in word_tags_features_list[i][1][j]:
                expected_counts[feature] += numerator/denominator

    #regularization grad
    regularization_grad = w_i*lamda

    likelihood = linear_term - normalization_term - regularization
    grad = empirical_counts - expected_counts - regularization_grad
    print("like = ",likelihood)
    return (-1)*likelihood, (-1)*grad

"""Now lets run the code untill we get the optimized weights"""

# Statistics
statistics = feature_statistics_class('train1.wtag')
statistics.get_word_tag_pair_count()
statistics.get_spelling_prefix_count()
statistics.get_spelling_suffix_count()
statistics.get_trigram_tags_count()
statistics.get_bigram_tags_count()
statistics.get_unigram_tags_count()

# feature2id
threshold=3
feature2id = feature2id_class(statistics, threshold,'train1.wtag')
feature2id.get_word_tag_pairs()
feature2id.get_prefix_tag_pairs()
feature2id.get_suffix_tag_pairs()
feature2id.get_trigram_tags_pairs()
feature2id.get_bigram_tags_pairs()
feature2id.get_unigram_tags_pairs()

tags_list = get_tags_list('train1.wtag')
word_features = word_feature_class(feature2id,'train1.wtag',tags_list)
word_features.find_relevant_features()
word_features_list=word_features.word_features_list #args1
word_tags_features_list = word_features.word_tags_features_list #args2
num_words = len(word_features_list) #args4
num_total_features = feature2id.n_total_features #args5
num_tags = len(tags_list) #args3
lamda = 100
args =(word_features_list,word_tags_features_list,num_tags,num_words,num_total_features, lamda)
w_0 = np.zeros(feature2id.n_total_features, dtype=np.float32)
optimal_params = fmin_l_bfgs_b(func=calc_objective_per_iter, x0=w_0, args=args, maxiter=1000, iprint=1)
weights = optimal_params[0]

# Now you can save weights using pickle.dump() - 'weights_path' specifies where the weight file will be saved.
# IMPORTANT - we expect to recieve weights in 'pickle' format, don't use any other format!!
weights_path = 'your_path_to_weights_dir/trained_weights_data_i.pkl' # i identifies which dataset this is trained on
with open(weights_path, 'wb') as f:
    pickle.dump(optimal_params, f)

#### In order to load pre-trained weights, just use the next code: ####
#                                                                     #
# with open(weights_path, 'rb') as f:                                 #
#   optimal_params = pickle.load(f)                                   #
# pre_trained_weights = optimal_params[0]                             #
#                                                                     #
#######################################################################
