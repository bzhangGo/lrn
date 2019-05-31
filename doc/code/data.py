# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import spacy
import numpy as np
from utils.util import batch_indexer, token_indexer
from utils.thread import threadsafe_generator

from bert.tokenization import BasicTokenizer as Tokenizer


class Dataset(object):
    def __init__(self,
                 task_data,
                 max_len,
                 max_w_len,
                 max_p_num,
                 word_vocab,
                 bert_vocab,
                 tokenizer,
                 enable_hierarchy=True,
                 char_vocab=None,
                 enable_char=True,
                 batch_or_token='batch'):
        self.data_set = task_data
        self.max_len = max_len
        self.max_w_len = max_w_len
        self.max_p_num = max_p_num
        self.enable_char = enable_char
        self.batch_or_token = batch_or_token

        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        if self.enable_char:
            assert self.char_vocab, 'Character vocabulary must be provided!'

        self.bert_vocab = bert_vocab
        self.bert_bpe_tokenizer = tokenizer
        self.bert_word_tokenizer = Tokenizer(do_lower_case=False)
        self.enable_bert = not (tokenizer is None or bert_vocab is None)

        self.enable_hierarchy = enable_hierarchy

        self.nlp = None
        self._create_nlp()

        self.leak_buffer = []

    def _create_nlp(self):
        cls = spacy.util.get_lang_class('en')  # 1. get Language instance, e.g. English()
        nlp = cls()

        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        self.nlp = nlp

    # split word-based tokens into sub-word based tokens
    def _tokenize(self, tokens):
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_sub_tokens = []
        for (i, token) in enumerate(tokens):
            orig_to_tok_index.append(len(all_sub_tokens))
            sub_tokens = self.bert_bpe_tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_sub_tokens.append(sub_token)
        return all_sub_tokens, tok_to_orig_index, orig_to_tok_index

    def load_data(self, train="train"):
        if train == "train":
            data_iter = self.data_set.get_train_data()
        elif train == "dev":
            data_iter = self.data_set.get_dev_data()
        else:
            assert train == "test"
            data_iter = self.data_set.get_test_data()

        for sample in data_iter:
            label, document = sample

            sentences = []
            if self.enable_hierarchy:
                parsed_document = self.nlp(document.decode('utf-8'))
                for sentence in parsed_document.sents:
                    tokened_sentence = self.bert_word_tokenizer.tokenize(sentence.string.encode('utf-8'))

                    sentences.append(tokened_sentence)
            else:
                sentences.append(self.bert_word_tokenizer.tokenize(document.decode("utf-8")))

            yield label, sentences

    def _process_one_sample(self, sample):
        sample['token_ids'] = [self.word_vocab.to_id(sentence, append_eos=False) for sentence in sample['tokens']]

        if self.enable_char:
            sample['char_ids'] = []
            for sentence in sample['tokens']:
                sample['char_ids'].append(
                    [self.char_vocab.to_id(list(token), append_eos=False) for token in sentence])

        if self.enable_bert:
            sample['subword_ids'] = []
            sample['token_to_subword_index'] = []
            sample['subword_to_token_index'] = []

            for sentence in sample['tokens']:
                sub_info = self._tokenize(sentence)
                sub_tokens = sub_info[0]
                subword_to_token_index = sub_info[1]
                token_to_subword_index = sub_info[2]
                subword_ids = [self.bert_vocab.cls] + self.bert_vocab.to_id(sub_tokens)

                sample['subword_ids'].append(subword_ids)
                sample['subword_to_token_index'].append(subword_to_token_index)
                sample['token_to_subword_index'].append([idx if idx < 512 else 0 for idx in token_to_subword_index])

        return sample

    def to_matrix(self, _batch, train="train"):
        # pre-tokenize the dataset
        batch = []
        for bidx, _sample in _batch:
            sample = {
                "label_id": _sample[0],
                "tokens": _sample[1],
            }

            batch.append((bidx, self._process_one_sample(sample)))

        # extract maximum numpy statistics
        max_p_num = max([len(sample['token_ids']) for _, sample in batch])
        max_len = max([len(sentence) for _, sample in batch for sentence in sample['token_ids']])

        if train == "train":
            max_p_num = min(max_p_num, self.max_p_num)
            max_len = min(max_len, self.max_len)

        max_sub_len = max_len
        if self.enable_bert:
            max_sub_len = max([len(sub_sentence) for _, sample in batch for sub_sentence in sample['subword_ids']])
            max_sub_len = min(max_sub_len, 512)

        batch_size = len(batch)

        samples = {'sample_idx': np.zeros([batch_size], dtype=np.int32),
                   'token_ids': np.zeros([batch_size * max_p_num, max_len], dtype=np.int32),
                   'l_id': np.zeros([batch_size], dtype=np.int32),
                   'raw': batch}
        if self.enable_char:
            samples['char_ids'] = np.zeros(
                [batch_size * max_p_num, max_len, self.max_w_len], dtype=np.int32)
        if self.enable_bert:
            samples['subword_ids'] = np.zeros([batch_size * max_p_num, max_sub_len], dtype=np.int32)
            samples['subword_back'] = np.zeros([batch_size * max_p_num, max_len], dtype=np.int32)

        for eidx, (sidx, sample) in enumerate(batch):
            samples['sample_idx'][eidx] = sidx

            for pidx, _ in enumerate(sample['token_ids']):
                if pidx >= max_p_num:
                    break
                f_pidx = eidx * max_p_num + pidx

                # deal with tokens
                token_ids = sample['token_ids'][pidx]
                samples['token_ids'][f_pidx, :min(max_len, len(token_ids))] = token_ids[:max_len]

                # deal with chars
                if self.enable_char:
                    for tidx, c_ids in enumerate(sample['char_ids'][pidx]):
                        if tidx >= max_len:
                            break
                        samples['char_ids'][f_pidx, tidx, :min(self.max_w_len, len(c_ids))] = c_ids[:self.max_w_len]

                # deal with bert
                if self.enable_bert:
                    subword_ids = sample['subword_ids'][pidx]
                    samples['subword_ids'][f_pidx, :min(max_sub_len, len(subword_ids))] = subword_ids[:max_sub_len]
                    subword_back = sample['token_to_subword_index'][pidx]
                    samples['subword_back'][f_pidx, :min(max_len, len(subword_back))] = subword_back[:max_len]

            samples['l_id'][eidx] = sample['label_id']

        return samples

    @threadsafe_generator
    def batcher(self, size, buffer_size=1000, shuffle=True, train="train"):
        # free up the instance length limitation
        if train != "train":
            self.max_len = int(1e6)
            self.batch_or_token = 'batch'

        def _handle_buffer(_buffer):
            sorted_buffer = sorted(_buffer, key=lambda xx: max([len(v) for v in xx[1][1]]))

            if self.batch_or_token == 'batch':
                buffer_index = batch_indexer(len(sorted_buffer), size)
            else:
                buffer_index = token_indexer(
                    [[len(v) for v in data[1][1]] for data in sorted_buffer], size)

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
                        else np.sum(data['token_ids'] > 0)
                    if batch_size < size * 0.1:
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
                    else np.sum(data['token_ids'] > 0)
                if train == 'train' and batch_size < size * 0.1:
                    self.leak_buffer += data['raw']
                else:
                    yield data
