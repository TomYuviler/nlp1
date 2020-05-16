
def get_num_of_sentences(file_path):
    with open(file_path) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def k_fold(file_path, k):

    num_of_sentences = get_num_of_sentences(file_path)
    block_size = int(num_of_sentences/k)
    print(block_size)
    with open(file_path) as f:
        data = f.readlines()
        i = block_size
        while i <= num_of_sentences:
            print("33333")
            with open('train.wtag', 'w') as train_file:
                if i == block_size:
                    train_file.writelines(data[i:])
                else:
                    if i == num_of_sentences:
                        train_file.writelines(data[:i])
                    else:
                        train_file.writelines(data[:i]+data[i + block_size:])
            with open('test.wtag', 'w') as test_file:
                test_file.writelines(data[i - block_size:i])
                print(i)
            yield
            i = i + block_size



