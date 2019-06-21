import pickle
import math

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split


class DataSet(object):
    def load_pickle(self, name):
        with open(name, 'rb') as f:
            return pickle.load(f, encoding='latin1')

    def save_pickle(self, obj, name, protocol=3):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, protocol=protocol)

    def generate_inverse_mapping(self, data_list):
        inverse_mapping = dict()
        for inner_id, true_id in enumerate(data_list):
            inverse_mapping[true_id] = inner_id
        return inverse_mapping

    def convert_to_inner_index(self, user_records, user_mapping, item_mapping):
        inner_user_records = []
        user_inverse_mapping = self.generate_inverse_mapping(user_mapping)
        item_inverse_mapping = self.generate_inverse_mapping(item_mapping)

        for user_id in range(len(user_mapping)):
            real_user_id = user_mapping[user_id]
            item_list = list(user_records[real_user_id])
            for index, real_item_id in enumerate(item_list):
                item_list[index] = item_inverse_mapping[real_item_id]
            inner_user_records.append(item_list)

        return inner_user_records, user_inverse_mapping, item_inverse_mapping

    def split_data_randomly(self, user_records, seed=0):
        # randomly hold part of the data as the test set
        test_ratio = 0.2
        train_set = []
        test_set = []
        for user_id, item_list in enumerate(user_records):
            tmp_train_sample, tmp_test_sample = train_test_split(item_list, test_size=test_ratio, random_state=seed)

            train_sample = []
            for place in item_list:
                if place not in tmp_test_sample:
                    train_sample.append(place)

            test_sample = []
            for place in tmp_test_sample:
                test_sample.append(place)

            train_set.append(train_sample)
            test_set.append(test_sample)
        return train_set, test_set

    def split_data_sequentially(self, user_records, test_radio=0.2):
        train_set = []
        test_set = []

        for item_list in user_records:
            len_list = len(item_list)
            num_test_samples = int(math.ceil(len_list * test_radio))
            train_sample = []
            test_sample = []
            for i in range(len_list - num_test_samples, len_list):
                test_sample.append(item_list[i])

            for place in item_list:
                if place not in set(test_sample):
                    train_sample.append(place)

            train_set.append(train_sample)
            test_set.append(test_sample)

        return train_set, test_set

    def generate_rating_matrix(self, train_set, num_users, num_items):
        # three lists are used to construct sparse matrix
        row = []
        col = []
        data = []
        for user_id, article_list in enumerate(train_set):
            for article in article_list:
                row.append(user_id)
                col.append(article)
                data.append(1)

        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

        return rating_matrix

    def load_item_content(self, f_in, D=8000):
        fp = open(f_in)
        lines = fp.readlines()
        X = np.zeros((len(lines), D))
        for i, line in enumerate(lines):
            strs = line.strip().split(' ')[2:]
            for strr in strs:
                segs = strr.split(':')
                X[i, int(segs[0])] = float(segs[1])

        return csr_matrix(X)

    def data_index_shift(self, lists, increase_by=2):
        """
        Increase the item index to contain the pad_index
        :param lists:
        :param increase_by:
        :return:
        """
        for seq in lists:
            for i, item_id in enumerate(seq):
                seq[i] = item_id + increase_by

        return lists