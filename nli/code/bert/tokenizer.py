# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import argparse

from . import tokenization


def load_tokenizer(params):
    tokenization.validate_case_matches_checkpoint(
        params.lower,
        os.path.join(params.bert_dir, 'bert_model.ckpt')
    )
    tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(params.bert_dir, 'vocab.txt'),
        do_lower_case=params.bert_lower
    )
    return tokenizer


def tokenize(params):
    tokenizer = load_tokenizer(params)

    with open(params.output, 'w') as writer:
        with open(params.input, 'r') as reader:
            for line in reader:
                writer.write(' '.join(tokenizer.tokenize(line.strip())).encode('utf8') + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Vocabulary Preparison')
    parser.add_argument('--lower', action='store_true', help='whether lowercase the model')
    parser.add_argument('--bert_dir', type=str, help='the pre-trained model directory')
    parser.add_argument('input', type=str, help='the input un-tokenized file')
    parser.add_argument('output', type=str, help='the output tokenized file')

    args = parser.parse_args()

    tokenize(args)

    print("Finishing!")
