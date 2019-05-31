# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np
import tensorflow as tf

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from bert.tokenization import BasicTokenizer as Tokenizer


class Vocab(object):
    def __init__(self, lower=False, vocab_file=None):
        self.lower = lower

        self.word2id = {}
        self.id2word = {}
        self.word2count = {}

        self.pad_sym = "<pad>"
        self.eos_sym = "<eos>"
        self.unk_sym = "<unk>"

        self.clean()

        self.pretrained_embedding = None

        if vocab_file is not None:
            self.load_vocab(vocab_file)

            if os.path.exists(vocab_file + ".npz"):
                pretrain_embedding = np.load(vocab_file + ".npz")['data']
                self.pretrained_embedding = pretrain_embedding

    def clean(self):
        self.word2id = {}
        self.id2word = {}
        self.word2count = {}

        self.insert(self.pad_sym)
        self.insert(self.unk_sym)
        self.insert(self.eos_sym)

    def insert(self, token):
        token = token if not self.lower else token.lower()
        if token not in self.word2id:
            index = len(self.word2id)
            self.word2id[token] = index
            self.id2word[index] = token

            self.word2count[token] = 0
        self.word2count[token] += 1

    def size(self):
        return len(self.word2id)

    def load_vocab(self, vocab_file):
        with open(vocab_file, 'r') as reader:
            for token in reader:
                self.insert(token.strip())

    def get_token(self, id):
        if id in self.id2word:
            return self.id2word[id]
        return self.unk_sym

    def get_id(self, token):
        token = token if not self.lower else token.lower()
        if token in self.word2id:
            return self.word2id[token]
        return self.word2id[self.unk_sym]

    def sort_vocab(self, least_freq=-1):
        sorted_word2count = sorted(
            self.word2count.items(), key=lambda x: - x[1])
        self.clean()
        for word, freq in sorted_word2count:
            if least_freq > 0:
                if freq <= least_freq:
                    continue
            self.insert(word)

    def save_vocab(self, vocab_file):
        with open(vocab_file, 'w') as writer:
            for id in range(self.size()):
                writer.write(self.id2word[id].encode("utf-8") + "\n")

        np.savez(vocab_file + ".npz", data=self.pretrained_embedding)

    def to_id(self, tokens, append_eos=True):
        if not append_eos:
            return [self.get_id(token) for token in tokens]
        else:
            return [self.get_id(token) for token in
                    tokens + [self.eos_sym]]

    def to_tokens(self, ids):
        return [self.get_token(id) for id in ids]

    def eos(self):
        return self.get_id(self.eos_sym)

    def pad(self):
        return self.get_id(self.pad_sym)

    def make_vocab(self, data_set, use_char=False, embedding_path=None):
        tf.logging.info("Starting Reading Data in {} Manner".format(use_char))
        tokenizer = Tokenizer(do_lower_case=False)

        for data_iter in [data_set.get_train_data(),
                          data_set.get_dev_data(),
                          data_set.get_test_data()]:
            for sample in data_iter:
                label, document = sample

                tokens = tokenizer.tokenize(document)
                for token in tokens:
                    if not use_char:
                        self.insert(token)
                    else:
                        for char in list(token):
                            self.insert(char)

        tf.logging.info("Data Loading Over, Starting Sorted")
        self.sort_vocab(least_freq=3 if use_char else -1)

        # process the vocabulary with pretrained-embeddings
        if embedding_path is not None:
            tf.logging.info("Pretrained Word Embedding Loading")
            embed_tokens = {}
            embed_size = None
            with open(embedding_path, 'r') as reader:
                for line in reader:
                    segs = line.strip().split(' ')

                    token = segs[0]
                    # Not used in our training data, pass
                    if token not in self.word2id:
                        continue
                    embed_tokens[token] = list(map(float, segs[1:]))

                    if embed_size is None:
                        embed_size = len(segs) - 1

            self.clean()
            for token in embed_tokens:
                self.insert(token)

            # load embeddings
            embeddings = np.zeros([len(embed_tokens), embed_size])
            for token in embed_tokens:
                # 3: the special symbols
                embeddings[self.get_id(token) - 3] = embed_tokens[token]

            self.pretrained_embedding = embeddings

        tf.logging.info("Vocabulary Loading Finished")
