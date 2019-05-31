# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Vocab(object):
    def __init__(self, vocab_file=None):
        self.word2id = {}
        self.id2word = {}
        self.word2count = {}

        self.pad_sym = "[PAD]"
        self.cls_sym = "[CLS]"
        self.sep_sym = "[SEP]"
        self.unk_sym = "[UNK]"

        if vocab_file is not None:
            self.load_vocab(vocab_file)

    def insert(self, token):
        if token not in self.word2id:
            index = len(self.word2id)
            self.word2id[token] = index
            self.id2word[index] = token

            self.word2count[token] = 0
        self.word2count[token] += 1

    @property
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
        if token in self.word2id:
            return self.word2id[token]
        return self.word2id[self.unk_sym]

    def save_vocab(self, vocab_file):
        with open(vocab_file, 'w') as writer:
            for id in range(self.size):
                writer.write(self.id2word[id] + "\n")

    def to_id(self, tokens):
        return [self.get_id(token) for token in tokens]

    def to_tokens(self, ids):
        return [self.get_token(id) for id in ids]

    @property
    def pad(self):
        return self.get_id(self.pad_sym)

    @property
    def cls(self):
        return self.get_id(self.cls_sym)

    @property
    def sep(self):
        return self.get_id(self.sep_sym)
