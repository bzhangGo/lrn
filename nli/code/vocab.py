# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import argparse
import numpy as np


class Vocab(object):
    def __init__(self, lower=False, vocab_file=None):
        self.word2id = {}
        self.id2word = {}
        self.word2count = {}

        self.pad_sym = "<pad>"
        self.eos_sym = "<eos>"
        self.unk_sym = "<unk>"

        self.lower = lower

        self.insert(self.pad_sym)
        self.insert(self.unk_sym)
        self.insert(self.eos_sym)

        if vocab_file is not None:
            self.load_vocab(vocab_file)

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
        self.word2id, self.id2word, self.word2count = {}, {}, {}
        self.insert(self.pad_sym)
        self.insert(self.unk_sym)
        self.insert(self.eos_sym)
        for word, freq in sorted_word2count:
            if least_freq > 0:
                if freq <= least_freq:
                    continue
            self.insert(word)

    def save_vocab(self, vocab_file):
        with open(vocab_file, 'w') as writer:
            for id in range(self.size()):
                writer.write(self.id2word[id].encode("utf-8") + "\n")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Vocabulary Preparation')
    parser.add_argument('--char', action='store_true', help='build char-level vocabulary')
    parser.add_argument('--lower', action='store_true', help='lower-case datasets')
    parser.add_argument('--embeddings', type=str, default='no', help='pre-trained word embedding path')
    parser.add_argument('inputs', type=str, help='the input file path, separate with comma')
    parser.add_argument('output', type=str, help='the output file name')

    args = parser.parse_args()

    vocab = Vocab(lower=args.lower)
    for data_file in args.inputs.split(','):
        with open(data_file, 'r') as reader:
            for text in reader:
                tokens = text.strip().split()

                for token in tokens:
                    if not args.char:
                        vocab.insert(token)
                    else:
                        for char in list(token):
                            vocab.insert(char)

    vocab.sort_vocab(least_freq=3 if args.char else -1)

    # process the vocabulary with pretrained-embeddings
    if args.embeddings != "no":
        embed_tokens = {}
        embed_size = None
        with open(args.embeddings, 'r') as reader:
            for line in reader:
                segs = line.strip().split(' ')

                token = segs[0]
                # Not used in our training data, pass
                if token not in vocab.word2id:
                    continue
                embed_tokens[token] = list(map(float, segs[1:]))

                if embed_size is None:
                    embed_size = len(segs) - 1

        vocab = Vocab(lower=args.lower)
        for token in embed_tokens:
            vocab.insert(token)

        # load embeddings
        embeddings = np.zeros([len(embed_tokens), embed_size])
        for token in embed_tokens:
            # 3: the special symbols
            embeddings[vocab.get_id(token) - 3] = embed_tokens[token]
        np.savez(args.output + ".npz", data=embeddings)

    vocab.save_vocab(args.output)

    print("Loading {} tokens from {}".format(vocab.size(), args.inputs))
