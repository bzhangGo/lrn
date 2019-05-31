# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from utils.util import batch_indexer, token_indexer
from utils.thread import threadsafe_generator


class Dataset(object):
    def __init__(self,
                 data_file,
                 max_len,
                 max_w_len,
                 word_vocab,
                 bert_vocab,
                 tokenizer,
                 char_vocab=None,
                 enable_char=True,
                 batch_or_token='batch'):
        self.p_file = data_file[0]
        self.h_file = data_file[1]
        self.l_file = data_file[2]
        self.max_len = max_len
        self.max_w_len = max_w_len
        self.enable_char = enable_char
        self.batch_or_token = batch_or_token

        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        if self.enable_char:
            assert self.char_vocab, 'Character vocabulary must be provided!'

        self.bert_vocab = bert_vocab
        self.tokenizer = tokenizer
        self.enable_bert = not (tokenizer is None or bert_vocab is None)

        self.leak_buffer = []

    # split word-based tokens into sub-word based tokens
    def _tokenize(self, tokens):
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_sub_tokens = []
        for (i, token) in enumerate(tokens):
            orig_to_tok_index.append(len(all_sub_tokens))
            sub_tokens = self.tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_sub_tokens.append(sub_token)
        return all_sub_tokens, tok_to_orig_index, orig_to_tok_index

    def load_data(self, train=True):
        with open(self.p_file, 'r') as p_reader, \
                open(self.h_file, 'r') as h_reader, \
                open(self.l_file, 'r') as l_reader:
            while True:
                p = p_reader.readline()
                h = h_reader.readline()
                l = l_reader.readline()

                if p == "":
                    break

                p_tokens = p.strip().split()
                h_tokens = h.strip().split()

                p_token_ids = self.word_vocab.to_id(p_tokens, append_eos=False)
                h_token_ids = self.word_vocab.to_id(h_tokens, append_eos=False)
                p_char_ids = [self.char_vocab.to_id(list(token), append_eos=False) for token in p_tokens]
                h_char_ids = [self.char_vocab.to_id(list(token), append_eos=False) for token in h_tokens]

                l_id = int(l.strip())

                if self.enable_bert:
                    max_len = int(1e6)
                    if train:
                        max_len = self.max_len

                    p_sub_info = self._tokenize(p_tokens[:max_len])
                    p_sub_tokens = p_sub_info[0]
                    p_token_to_subword_index = p_sub_info[2]
                    p_subword_ids = [self.bert_vocab.cls] + self.bert_vocab.to_id(p_sub_tokens) + [self.bert_vocab.sep]

                    h_sub_info = self._tokenize(h_tokens[:max_len])
                    h_sub_tokens = h_sub_info[0]
                    h_token_to_subword_index = h_sub_info[2]
                    h_subword_ids = self.bert_vocab.to_id(h_sub_tokens) + [self.bert_vocab.sep]

                    yield (l_id, p_token_ids, h_token_ids, p_char_ids, h_char_ids,
                           p_subword_ids, h_subword_ids, p_token_to_subword_index, h_token_to_subword_index)
                else:
                    yield (l_id, p_token_ids, h_token_ids, p_char_ids, h_char_ids)

    def to_matrix(self, batch, train=True):
        max_p_len = max([len(sample[1]) for _, sample in batch])
        max_h_len = max([len(sample[2]) for _, sample in batch])

        max_p_sub_len = max_p_len
        max_h_sub_len = max_h_len
        if self.enable_bert:
            max_p_sub_len = max([len(sample[5]) for _, sample in batch])
            max_h_sub_len = max([len(sample[6]) for _, sample in batch])

        if train:
            max_p_len = min(max_p_len, self.max_len)
            max_h_len = min(max_h_len, self.max_len)

        batch_size = len(batch)

        samples = {'sample_idx': np.zeros([batch_size], dtype=np.int32),
                   'p_token_ids': np.zeros([batch_size, max_p_len], dtype=np.int32),
                   'h_token_ids': np.zeros([batch_size, max_h_len], dtype=np.int32),
                   'l_id': np.zeros([batch_size], dtype=np.int32),
                   'raw': batch}
        if self.enable_char:
            samples['p_char_ids'] = np.zeros(
                [batch_size, max_p_len, self.max_w_len], dtype=np.int32)
            samples['h_char_ids'] = np.zeros(
                [batch_size, max_h_len, self.max_w_len], dtype=np.int32)
        if self.enable_bert:
            samples['p_subword_ids'] = np.zeros([batch_size, max_p_sub_len], dtype=np.int32)
            samples['h_subword_ids'] = np.zeros([batch_size, max_h_sub_len], dtype=np.int32)
            samples['p_subword_back'] = np.zeros([batch_size, max_p_len], dtype=np.int32)
            samples['h_subword_back'] = np.zeros([batch_size, max_h_len], dtype=np.int32)

        for eidx, (sidx, sample) in enumerate(batch):
            samples['sample_idx'][eidx] = sidx

            l_id, p_token_ids, h_token_ids, p_char_ids, h_char_ids = sample[:5]

            if self.enable_bert:
                p_subword_ids, h_subword_ids, p_subword_back, h_subword_back = sample[5:]

                # deal with subwords
                cur_p_sub_w_len = min(max_p_sub_len, len(p_subword_ids))
                samples['p_subword_ids'][eidx, :cur_p_sub_w_len] = p_subword_ids[:max_p_sub_len]
                cur_h_sub_w_len = min(max_h_sub_len, len(h_subword_ids))
                samples['h_subword_ids'][eidx, :cur_h_sub_w_len] = h_subword_ids[:max_h_sub_len]

                cur_p_sub_b_len = min(max_p_len, len(p_subword_back))
                samples['p_subword_back'][eidx, :cur_p_sub_b_len] = p_subword_back[:max_p_len]
                cur_h_sub_b_len = min(max_h_len, len(h_subword_back))
                samples['h_subword_back'][eidx, :cur_h_sub_b_len] = h_subword_back[:max_h_len]

            # deal with premise tokens and chars
            cur_p_t_len = min(max_p_len, len(p_token_ids))
            samples['p_token_ids'][eidx, :cur_p_t_len] = p_token_ids[:max_p_len]

            if self.enable_char:
                for tidx, p_c_ids in enumerate(p_char_ids):
                    if tidx >= max_p_len:
                        break
                    cur_p_c_len = min(self.max_w_len, len(p_c_ids))
                    samples['p_char_ids'][eidx, tidx, :cur_p_c_len] = p_c_ids[:self.max_w_len]

            # deal with passage tokens and chars
            cur_h_t_len = min(max_h_len, len(h_token_ids))
            samples['h_token_ids'][eidx, :cur_h_t_len] = h_token_ids[:max_h_len]

            if self.enable_char:
                for tidx, h_c_ids in enumerate(h_char_ids):
                    if tidx >= max_h_len:
                        break
                    cur_h_c_len = min(self.max_w_len, len(h_c_ids))
                    samples['h_char_ids'][eidx, tidx, :cur_h_c_len] = h_c_ids[:self.max_w_len]

            samples['l_id'][eidx] = l_id

        return samples

    @threadsafe_generator
    def batcher(self, size, buffer_size=1000, shuffle=True, train=True):
        def _handle_buffer(_buffer):
            sorted_buffer = sorted(buffer,
                                   key=lambda xx: max(len(xx[1][1]), len(xx[1][2])))
            if self.batch_or_token == 'batch':
                buffer_index = batch_indexer(len(sorted_buffer), size)
            else:
                buffer_index = token_indexer(
                    [[len(data[1][1]), len(data[1][2])] for data in sorted_buffer], size)
            index_over_index = batch_indexer(len(buffer_index), 1)
            if shuffle:
                np.random.shuffle(index_over_index)

            for ioi in index_over_index:
                index = buffer_index[ioi[0]]
                batch = [_buffer[ii] for ii in index]
                yield self.to_matrix(batch, train=train)

        buffer = self.leak_buffer
        self.leak_buffer = []
        for i, sample in enumerate(self.load_data(train=train)):
            buffer.append((i, sample))
            if len(buffer) >= buffer_size:
                for data in _handle_buffer(buffer):
                    # check whether the data is tailed
                    # tokens are counted on 'p'
                    batch_size = len(data['raw']) if self.batch_or_token == 'batch' \
                        else np.sum(data['p_token_ids'] > 0)
                    if batch_size < size * 0.8:
                        self.leak_buffer += data['raw']
                    else:
                        yield data
                buffer = self.leak_buffer
                self.leak_buffer = []

        # deal with data in the buffer
        if len(buffer) > 0:
            for data in _handle_buffer(buffer):
                # check whether the data is tailed
                batch_size = len(data['raw']) if self.batch_or_token == 'batch' \
                    else np.sum(data['p_token_ids'] > 0)
                if train and batch_size < size * 0.8:
                    self.leak_buffer += data['raw']
                else:
                    yield data
