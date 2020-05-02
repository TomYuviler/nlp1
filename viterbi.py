import numpy as np

class Viterbi():
    """ Viterbi algorithm implementation"""

    def __init__(self, model):
        self.model = model
        self.num_of_tags = model.num_tags
        self.tags_list = model.tags_list
        self._idx_tag_dict = {idx: tag for idx, tag in enumerate(model.tags_list)}
        self.tags_pairs = [(x, y) for x in self.tags_list for y in self.tags_list]
        self.tags_pair_pos = {(x, y): i for i, x, y in enumerate(self.tags_pairs)}


    def run_viterbi(self, sentence, beam_size=2):

        n = len(sentence)
        pi = np.full((n, self.num_of_tags ** 2), -np.inf)
        bp = np.zeros((n, self.num_of_tags ** 2))

        for k in range(0, n - 1):
            for u, v in self.tags_pairs:
                values = np.full(len(self.tags_list), -np.inf)
                for i, t in enumerate(self.tags_list):
                    if pi[k, self.tags_pair_pos[(t, u)]] == -np.inf:
                        continue
                    trigram_prob = self.trigram_prob(v, t, u, sentence, k)
                    values[i] = pi[k, self.tags_pair_pos[(t, u)]] + trigram_prob

                max_pos = np.argmax(values)
                pi[k, self.tags_pair_pos[(u, v)]] = values[max_pos]
                bp[k, self.tags_pair_pos[(u, v)]] = max_pos

            pi_k = pi[k, :]
            pi_k = pi_k[np.argpartition(pi_k, - beam_size)[- beam_size:]]
            pi[k, :beam_size] = pi_k

        print(pi)


if __name__ == '__main__':
    pass