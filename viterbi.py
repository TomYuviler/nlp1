import numpy as np
import nlp1
import random
import sys
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
        self.tags_list = ['*', 'DT', 'NNP', 'VB', 'RBZ']
        self.num_of_tags = len(self.tags_list) + 1
        self.tags_pairs = [(x, y) for x in self.tags_list for y in self.tags_list]
        self.tags_pair_pos = {(pair[0], pair[1]): i for i, pair in enumerate(self.tags_pairs)}

    def get_history(self, v, t, u, sentence, k):
        """ Return the history vector for the requested word in a certain sentence.

            Args:
                v (string): The tag of the Kth positioned word.
                t (string): The tag of the K-2th positioned word.
                u (string): The tag of the K-1th positioned word.
                sentence (list of strings): The current sentence.
                k (int): The position of the requested word.

            Returns:
                    A history vector (list of strings).
        """
        split_words = ['*'] + sentence.split(' ')
        del split_words[-1]
        split_words = split_words + ['STOP']
        pptag = t
        ptag = u
        ctag = v
        pword, word, nword = split_words[k-1:k+2]
        ntag = 'VBZ'
        history = (word, ptag, ntag, ctag, pword, nword, pptag)
        return len(split_words), history

    def get_tri_probability(self, v, t, u, sentence, k):
        linear_term = 0
        weights = [random.uniform(0, 1) for weight in self.model.weights]
        num_words, history = self.get_history(v, t, u, sentence, k)
        word_features_list = nlp1.represent_input_with_features(history, self.model.feature2id)
        for feature in word_features_list:
            linear_term += weights[feature]
        return linear_term

    def run_viterbi(self, sentence, beam_size=5):

        n = len(sentence.split(' '))
        pi = np.full((n, self.num_of_tags ** 2), -np.inf)
        bp = np.zeros((n, self.num_of_tags ** 2))
        pi[0, self.tags_pair_pos[('*', '*')]] = 1
        predicted_tags = ['*' for word in range(n)]

        for k in range(1, n):
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

            # Beam implementation:
            # pi_k = pi[k, :]
            # pi_k = pi_k[np.argpartition(pi_k, - beam_size)[- beam_size:]]
            # pi[k, :beam_size] = pi_k

        predicted_tags[-1], predicted_tags[-2] = self.tags_pairs[np.argmax(pi[n-1,:])]
        for k in range(n-3, -1, -1):

            predicted_tags[k] = self.tags_list[int(bp[k+2, self.tags_pair_pos[(predicted_tags[k+1], predicted_tags[k+2])]])]
        return predicted_tags


if __name__ == '__main__':
    pass