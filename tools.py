import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


class SummeryTools:
    """
    Contains functions for analytics on the result of a OpTy MEMM POS tagger.
    """

    def __init__(self, tags_list, true_tags, predicted_tags, sentence):
        """
        Args:
             tags_list(list): A list of the available tags.
             true_tags(list of 'str'): The real tags for the requested list of words.
             predicted_tags(list of 'str'): The model's predicted tags for the requested list of words.
             sentence(list of 'str'): The requested sentence to predict.
        """
        self.num_of_tags = len(tags_list)
        self.tags_list = tags_list
        self.true_tags = true_tags
        self.predicted_tags = predicted_tags
        self.sentence = sentence
        self.confusion_matrix = pd.DataFrame(0, index=self.tags_list, columns=self.tags_list)

    def get_most_common_mistakes_per_tag(self):
        """Returns a table with all the tags and their most common mistake prediction"""

        confusion_matrix = pd.DataFrame.copy(self.confusion_matrix)

        for tag in self.tags_list:
            confusion_matrix.at[tag, tag] = -1

        common_mistake_per_tag = pd.concat([confusion_matrix.idxmax(axis=1), confusion_matrix.max(axis=1)], axis=1)
        common_mistake_per_tag.columns = ["wrong predicted tag", "num of occurrences"]
        common_mistake_per_tag = common_mistake_per_tag.sort_values(by=["num of occurrences"], ascending=False)

        return common_mistake_per_tag

    def get_most_common_mistakes_per_words(self):
        """Returns a table with all the real (word, tag) pairs and their most common mistake prediction"""

        word_true_tag_list = [(word, true_tag) for word, true_tag in zip(self.sentence, self.true_tags)]
        confusion_matrix = pd.DataFrame(0, index=set(word_true_tag_list), columns=self.tags_list)

        for word_true_tag, predicted_tag in zip(word_true_tag_list, self.predicted_tags):
            confusion_matrix.at[word_true_tag, predicted_tag] = confusion_matrix.at[word_true_tag, predicted_tag] + 1

        for word_true_tag in word_true_tag_list:
            confusion_matrix.at[word_true_tag, word_true_tag[1]] = -1

        common_mistake_per_word = pd.concat([confusion_matrix.idxmax(axis=1), confusion_matrix.max(axis=1)], axis=1)
        common_mistake_per_word.columns = ["wrong predicted tag", "num of occurrences"]
        common_mistake_per_word = common_mistake_per_word.sort_values(by=["num of occurrences"], ascending=False)

        return common_mistake_per_word

    def get_confusion_matrix(self):
        """Returns a full confusion matrix of the true and the predicted tags."""

        for true_tag, predicted_tag in zip(self.true_tags, self.predicted_tags):
            self.confusion_matrix.at[true_tag, predicted_tag] = self.confusion_matrix.at[true_tag, predicted_tag] + 1

        return pd.DataFrame.copy(self.confusion_matrix)

    def get_top10_confusion_matrix(self):
        """ Return a confusion matrix of the top 10 most mistakable true tags."""

        confusion_matrix = pd.DataFrame.copy(self.confusion_matrix)

        for tag in self.tags_list:
            confusion_matrix.at[tag, tag] = 0

        top10_tags = confusion_matrix.sum(axis=1).sort_values(ascending=False).head(10)
        confusion_matrix = confusion_matrix.loc[top10_tags.index]

        plt.figure(figsize=(20, 20))
        ax = sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 20})
        ax.set_xlabel("Predicted Tags", fontsize=25)
        ax.set_ylabel("Real Tags", fontsize=25)
        ax.set_title("Top 10 most mistakable tags confusion matrix", fontsize=25)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 1, top - 1)
        plt.savefig('confusion_matrix_1')

        return confusion_matrix
