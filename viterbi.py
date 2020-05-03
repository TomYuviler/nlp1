import numpy as np
import nlp1

class Viterbi():
    """ Viterbi algorithm implementation"""

    def __init__(self, model):
        self.model = model
        self.tags_list = ['*', 'DT', 'NNP']
        self.num_of_tags = model.num_tags + 1
        self._idx_tag_dict = {idx: tag for idx, tag in enumerate(model.tags_list)}
        self.tags_pairs = [(x, y) for x in self.tags_list for y in self.tags_list]
        self.tags_pair_pos = {(pair[0], pair[1]): i for i, pair in enumerate(self.tags_pairs)}

    def get_history(self, v, t, u, sentence, k):
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
        num_words, history = self.get_history(v, t, u, sentence, k)
        word_features_list = nlp1.represent_input_with_features(history, self.model.feature2id)
        for feature in word_features_list:
            linear_term += self.model.weights[feature]
        print(linear_term)
        return linear_term

    def run_viterbi(self, sentence, beam_size=2):

        n = len(sentence.split(' '))
        pi = np.full((n, self.num_of_tags ** 2), -np.inf)
        bp = np.zeros((n, self.num_of_tags ** 2))
        pi[0, self.tags_pair_pos[('*', '*')]] = 1

        for k in range(1, n-2):
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

            pi_k = pi[k, :]
            pi_k = pi_k[np.argpartition(pi_k, - beam_size)[- beam_size:]]
            pi[k, :beam_size] = pi_k


        print("----------------")
        print(pi)



if __name__ == '__main__':
    pass