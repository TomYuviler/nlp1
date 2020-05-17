from nlp1 import *
from viterbi import get_num_of_sentences


def strip_tagged_file(file_path):
    """ Create a tags striped version of a tagged text file.

        Args:
            file_path (str): The path to the requested file to strip (the file need to have a blank.
    """

    with open('stripped_'+file_path, 'w') as f:
        f.write('')

    with open(file_path) as f:
        num_of_sentences = get_num_of_sentences(file_path)
        i = 0
        print("Stripping....")
        for line in f:
            i += 1
            print('Running on line ', i)
            split_words = re.split(' |\n', line)

            if i < num_of_sentences+1:
                del split_words[-1]
            with open('stripped_'+file_path, 'a') as q:
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = re.split('_', split_words[word_idx])
                    q.write(cur_word+' ')
                q.write('\n')

def generate_tagged_comp():
    """Generates the tagged competition files."""

    with open('OpTyTaggerModel1.pkl', 'rb') as pickle_file:
        model_a = pickle.load(pickle_file)
    print("Running Viterbi")
    viterbi_1 = viterbi.Viterbi(model_a)
    print("Tagging comp1...")
    viterbi_1.viterbi_that_file('comp1.words')

    with open('OpTyTaggerModel2.pkl', 'rb') as pickle_file:
        model_b = pickle.load(pickle_file)
    print("Running Viterbi")
    viterbi_1 = viterbi.Viterbi(model_b)
    print("Tagging comp2...")
    viterbi_1.viterbi_that_file('comp1.words')


if __name__ == '__main__':

    generate_tagged_comp()
    strip_tagged_file('tagged_comp1.words')
