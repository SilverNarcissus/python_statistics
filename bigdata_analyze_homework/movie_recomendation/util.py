import numpy as np
import os
from gensim.models import Word2Vec
from sklearn import preprocessing

# file path constant
MATRIX_ZIP_FILE = 'data/matrix'
USER_FILE = 'data/users.txt'
TRAIN_FILE = 'data/netflix_train.txt'
TEST_FILE = 'data/netflix_test.txt'
MOVIE_FILE = 'data/movie_titles.txt'


# fill score matrix by record
def fill_score(matrix, record, user_id_index_map):
    value = record.split(' ')
    user_index = user_id_index_map[value[0]]
    movie_index = int(value[1]) - 1
    movie_score = int(value[2])
    matrix[user_index][movie_index] = movie_score


def build_data_matrix_and_dump():
    with open(USER_FILE, encoding='utf-8') as user_file:
        with open(TRAIN_FILE, encoding='utf-8') as train_file:
            with open(TEST_FILE, encoding='utf-8') as test_file:
                # build user_id_index_map
                user_id_index_map = {}
                i = 0
                line = user_file.readline()
                while line:
                    user_id = line.strip()
                    user_id_index_map[user_id] = i
                    i += 1
                    line = user_file.readline()

                # build train matrix and test matrix
                train = np.zeros((10000, 10000))
                test = np.zeros((10000, 10000))

                line = train_file.readline()
                while line:
                    print(line)
                    fill_score(train, line.strip(), user_id_index_map)
                    line = train_file.readline()

                line = test_file.readline()
                while line:
                    print(line)
                    fill_score(test, line.strip(), user_id_index_map)
                    line = test_file.readline()

                np.savez(MATRIX_ZIP_FILE, train=train, test=test)
                return train, test


# build movie matrix by title
def build_movie_matrix():
    model = Word2Vec.load("word2vec.model")
    size = 100
    res = []
    # for analyze
    miss = 0
    total = 0
    count = 0
    with open(MOVIE_FILE, encoding='latin-1') as movie_file:
        line = movie_file.readline()
        while line:
            count += 1
            record = line.strip().split(',')
            movie_name = record[2]
            row = np.zeros(size)
            for word in movie_name.split(' '):
                total += 1
                if word in model.wv:
                    row = row + model.wv[word]
                else:
                    miss += 1
            res.append(row)
            # only need 10000 movies
            if count >= 10000:
                break
            line = movie_file.readline()

    res = np.array(res)
    print(res.shape)
    # 结果进行归一化
    res = preprocessing.MinMaxScaler().fit_transform(res.T)
    print(res.T)
    print('miss:', miss, 'total:', total)
    return res.T


def generate_data():
    if os.path.exists(MATRIX_ZIP_FILE + ".npz"):
        zip_file = np.load(MATRIX_ZIP_FILE + ".npz")
        return zip_file['train'], zip_file['test']

    print("Can not find dump matrix, build from txt, please wait...")
    return build_data_matrix_and_dump()


def eval_rmse(test, final_score):
    # indicate which movie was scored
    non_zero_index_array = np.nonzero(test)
    test_data = test[non_zero_index_array]
    score_data = final_score[non_zero_index_array]

    return np.sqrt(np.average((score_data - test_data) ** 2))
