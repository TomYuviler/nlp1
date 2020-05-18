
def get_num_of_sentences(file_path):
    """Returns the number of sentences in the file."""

    with open(file_path) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def k_fold(file_path, k):
    """ Split the data into k-1 blocks to use as train data and 1 block use as test data.
        At each iteration split the data into a different permutation of the blocks and save a train and test files.

        Args:
            file_path (str): The path to the text file.
            k (int): The number of blocks.
    """

    num_of_sentences = get_num_of_sentences(file_path)
    block_size = int(num_of_sentences/k)
    with open(file_path) as f:
        data = f.readlines()
        i = block_size
        while i <= num_of_sentences:
            with open('train.wtag', 'w') as train_file:
                if i == block_size:
                    train_file.writelines(data[i:])
                else:
                    if i == num_of_sentences:
                        train_file.writelines(data[:-block_size])
                    else:
                        train_file.writelines(data[:i-block_size] + data[i + block_size:])
            with open('test.wtag', 'w') as test_file:
                test_file.writelines(data[i - block_size:i])

            yield
            i = i + block_size



