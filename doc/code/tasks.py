# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import abc
import csv
import numpy as np


def get_task(params, is_training):
    name = params.task.lower()

    if name == "amafull":
        return AMAFull(params.data_path, is_training)
    elif name == "amapolar":
        return AMAPolar(params.data_path, is_training)
    elif name == "yahoo":
        return YaHoo(params.data_path, is_training)
    elif name == "yelpfull":
        return YelpFull(params.data_path, is_training)
    elif name == "yelppolar":
        return YelpPolar(params.data_path, is_training)
    else:
        raise NotImplementedError("Not Supported: {}".format(name))


class Task(object):
    def __init__(self, data_path, is_training=False):
        self.data_path = data_path
        self.is_training = is_training

        self.trainset = []
        self.devset = []
        self.testset = []

        if self.is_training:
            self._read_all_train_dev_data()
        self._read_all_test_data()

    def _clean_text(self, text_in):
        return text_in.replace('\\"', '"').replace('\\n', ' ')

    def _read_all_train_dev_data(self):
        train_data_path = os.path.join(self.data_path, "train.csv")

        dataset = []
        with open(train_data_path) as tfile:
            reader = csv.reader(tfile, delimiter=",")

            for sample in reader:
                dataset.append(sample)

        np.random.shuffle(dataset)

        # split the dataset with 90% and 10%
        dev_size = int(len(dataset) * 0.1)

        self.devset = dataset[:dev_size]
        self.trainset = dataset[dev_size:]

    def _read_all_test_data(self):
        test_data_path = os.path.join(self.data_path, "test.csv")

        self.testset = []
        with open(test_data_path) as tfile:
            reader = csv.reader(tfile, delimiter=",")

            for sample in reader:
                self.testset.append(sample)

    def _data_iter(self, iterator):
        for sample in iterator:
            label = int(sample[0]) - 1
            document = ' '.join(sample[1:])

            document = self._clean_text(document)

            yield (label, document)

    def get_train_data(self):
        np.random.shuffle(self.trainset)
        for sample in self._data_iter(self.trainset):
            yield sample

    def get_dev_data(self):
        for sample in self._data_iter(self.devset):
            yield sample

    def get_test_data(self):
        for sample in self._data_iter(self.testset):
            yield sample

    @abc.abstractmethod
    def get_label_size(self):
        raise NotImplementedError("Not Supported")


# amazon_review_full_csv
class AMAFull(Task):
    def __init__(self, data_path, is_training=False):
        data_path = os.path.join(data_path, "amazon_review_full_csv")
        super(AMAFull, self).__init__(data_path, is_training)

    def get_label_size(self):
        return 5


# amazon_review_polarity_csv
class AMAPolar(Task):
    def __init__(self, data_path, is_training=False):
        data_path = os.path.join(data_path, "amazon_review_polarity_csv")
        super(AMAPolar, self).__init__(data_path, is_training)

    def get_label_size(self):
        return 2


# yahoo_answers_csv
class YaHoo(Task):
    def __init__(self, data_path, is_training=False):
        data_path = os.path.join(data_path, "yahoo_answers_csv")
        super(YaHoo, self).__init__(data_path, is_training)

    def get_label_size(self):
        return 10


# yelp_review_full_csv
class YelpFull(Task):
    def __init__(self, data_path, is_training=False):
        data_path = os.path.join(data_path, "yelp_review_full_csv")
        super(YelpFull, self).__init__(data_path, is_training)

    def get_label_size(self):
        return 5


# yelp_review_polarity_csv
class YelpPolar(Task):
    def __init__(self, data_path, is_training=False):
        data_path = os.path.join(data_path, "yelp_review_polarity_csv")
        super(YelpPolar, self).__init__(data_path, is_training)

    def get_label_size(self):
        return 2
