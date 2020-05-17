import numpy as np
import nlp1
import re
import math
import sys
import tools
#from openpyxl import *
np.set_printoptions(threshold=sys.maxsize)

class Viterbi():
    """ Viterbi algorithm implementation

        Args: A trained OpTyTagger object (MEMM Tagger)

        Attributes:
            model (OpTyTagger Object):
            tags_list (list of strings):
            num_of_tags (int):
            tags_pairs (list of strings tuples):
            tags_pair_pos (dict):

    """

    def __init__(self, model):
        self.model = model
        self.tags_list = ['*'] + model.tags_list
        self.num_of_tags = len(self.tags_list)
        self.tags_pairs = [(x, y) for x in self.tags_list for y in self.tags_list]
        self.tags_pair_pos = {(pair[0], pair[1]): i for i, pair in enumerate(self.tags_pairs)}
        self.all_words=[]

    def get_history(self, v, t, u, sentence, k):
        """ Returns the history vector for the requested word in a certain sentence.

            Args:
                v (string): The tag of the Kth positioned word.
                t (string): The tag of the K-2th positioned word.
                u (string): The tag of the K-1th positioned word.
                sentence (list of strings): The current sentence.
                k (int): The position of the requested word.

            Returns:
                A history vector (list of strings): (word, ptag, ntag, ctag, pword, nword, pptag).
        """
        split_words = sentence + ['STOP'] + ['STOP'] + ['STOP']
        pptag = t
        ptag = u
        ctag = v
        ppword, pword, word, nword, nnword, nnnword = split_words[k-2:k+4]
        ntag = ' '  # TODO: deal with it - right now it's just a random word u chose.
        current_history = (word, ptag, ntag, ctag, pword, nword, pptag, ppword, nnword, nnnword)
        other_histories = []
        for tag in self.model.tags_list:
            other_histories.append((word, ptag, ntag, tag, pword, nword, pptag, ppword, nnword, nnnword))

        return current_history, other_histories

    def get_tri_probability(self, v, t, u, sentence, k):
        """ Returns the probability that the tag of the k word is v based on t as the k-2 tag,
            u as the k-1 tag and the current sentence.

            Args:
                v (string): The tag of the Kth positioned word.
                t (string): The tag of the K-2th positioned word.
                u (string): The tag of the K-1th positioned word.
                sentence (list of strings): The current sentence.
                k (int): The position of the requested word.

            Returns:
                numerator/denominator (float): Represent the probability of the requested event.
        """
        linear_term = 0
        split_words = ['*']+['*'] + sentence
        history, other_histories = self.get_history(v, t, u, split_words, k)
        word_features_list = nlp1.represent_input_with_features(history, self.model.feature2id)
        for feature in word_features_list:
            linear_term += self.model.weights[feature]
        numerator = math.exp(linear_term)

        denominator = 0
        for other_history in other_histories:
            linear_term = 0
            word_features_list = nlp1.represent_input_with_features(other_history, self.model.feature2id)
            for feature in word_features_list:
                linear_term += self.model.weights[feature]
            denominator = denominator + math.exp(linear_term)
        return float(numerator)/denominator

    def get_accuracy(self, real_tags, predicted):
        """
        Calculate and returns the accuracy measure of the current prediction.

        Arg:
            real_tags (list of strings): Contains the real tags.
            predicted (list of strings): Contains the predicted tags.

        Return:
            (float): The accuracy measure of the current prediction.
        """
        n = len(real_tags)
        num_of_correct = 0
        for i in range(n):
            num_of_correct = num_of_correct + (real_tags[i] == predicted[i])

        print("num of words:", n)
        print("num of correct tags:", num_of_correct)

        return float(num_of_correct)/n

    def run_viterbi(self, sentence, beam_size=5, active_beam=False):
        """
        Arg:
            sentence (string): The requested sentence to tag.
            beam_size (int): The beam size in case we use Beam Viterbi.
            active_beam (bool): Gets True if we want to run a Beam Viterbi algorithm instead of the regular one.

        Return:
            list of the predicted tags for the current sentence.
        """
        n = len(sentence) + 2
        pi = np.full((n, self.num_of_tags ** 2), -np.inf)
        bp = np.zeros((n, self.num_of_tags ** 2))
        pi[1, self.tags_pair_pos[('*', '*')]] = 1
        predicted_tags = ['*' for word in range(n)]

        for k in range(2, n):
            for u, v in self.tags_pairs:
                values = np.full(len(self.tags_list), -np.inf)
                for i, t in enumerate(self.tags_list):
                    if pi[k-1, self.tags_pair_pos[(t, u)]] == -np.inf or v == '*':
                        continue
                    trigram_prob = self.get_tri_probability(v, t, u, sentence, k)
                    values[i] = pi[k-1, self.tags_pair_pos[(t, u)]] * trigram_prob

                max_pos = np.argmax(values)
                pi[k, self.tags_pair_pos[(u, v)]] = values[max_pos]
                bp[k, self.tags_pair_pos[(u, v)]] = max_pos

            if active_beam:
                pi_k = pi[k, :]
                threshold = pi_k[np.argpartition(pi_k, len(pi_k) - beam_size)[len(pi_k) - beam_size]]
                pi[k, :] = np.where(pi_k >= threshold, pi_k, -np.inf)

        predicted_tags[n-2], predicted_tags[n-1] = self.tags_pairs[np.argmax(pi[n-1,:])]
        for k in range(n-3, 0, -1):
            predicted_tags[k] = self.tags_list[int(bp[k+2, self.tags_pair_pos[(predicted_tags[k+1], predicted_tags[k+2])]])]
        return predicted_tags

    def viterbi_that_file(self, file_path, with_tags=False):
        """
        Tag all the words in the requested file with the OpTy Model through Viterbi algorithm.

        Args:
            file_path (string): The path to the file that contains the words to be tagged.
            with_tags (bool): Gets True if the file contains words with tags.

        Return:
            A list of the predicted tags.
        """
        predictions = []
        real_tags = []
        sentence = []

        with open(file_path) as f:
            i = 0
            for line in f:
                print(i)
                #if i==100:
                   #break
                i+=1
                split_words = re.split(' |\n', line)
                del split_words[-1]
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = re.split('_', split_words[word_idx])
                    sentence.append(cur_word)
                    self.all_words.append(cur_word)
                    real_tags.append(cur_tag)

                pred_tags = self.run_viterbi(sentence, active_beam=True, beam_size=5)[2:]
                for prediction in pred_tags:
                    predictions.append(prediction)
                sentence = []

        pred_dict ={}
        for i in range (len(real_tags)):
            if real_tags[i] == predictions[i]:
                continue
            if (real_tags[i],predictions[i]) in pred_dict:
                pred_dict[(real_tags[i],predictions[i])] += 1
            else:
                pred_dict[(real_tags[i], predictions[i])] = 1
        print(pred_dict)
        print(real_tags)
        print(predictions)
        if with_tags:
            print("The Accuracy is:", self.get_accuracy(real_tags, predictions))
        tool = tools.SummeryTools(self.tags_list,real_tags,predictions,self.all_words)
        conf = tool.get_confusion_matrix()
        print(tool.get_most_common_mistakes_per_tag())
        print("-------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------")
        #tool.get_most_common_mistakes_per_words().to_excel("output.xlsx",sheet_name='Sheet_name_1')
        tool.get_most_common_mistakes_per_words().to_csv("tom3.csv",index=True)
        print(tool.get_most_common_mistakes_per_words())


if __name__ == '__main__':
    pass