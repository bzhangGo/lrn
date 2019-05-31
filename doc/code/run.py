# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import random

import copy
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

import bert
import main as graph
from tasks import get_task
from vocab import Vocab
from utils.recorder import Recorder

# define global initial parameters
global_params = tc.training.HParams(
    # lrate decay
    # number of shards
    nstable=4,
    # start, end: learning rate decay parameters used in GNMT+
    lrdecay_start=600000,
    lrdecay_end=1200000,
    # warmup steps: start point for learning rate stop increaing
    warmup_steps=400,
    # select strategy: noam, gnmt+, epoch, score and vanilla
    lrate_strategy="gnmt+",
    # learning decay rate
    lrate_decay=0.5,
    # when using score, patience number of bad score obtained for one decay
    lrate_patience=1,
    # weight decay for L2 loss
    weight_decay=3e-4,

    # early stopping
    estop_patience=100,

    # initialization
    # type of initializer
    initializer="uniform",
    # initializer range control
    initializer_gain=0.08,

    # parameters for rnnsearch
    # encoder and decoder hidden size
    hidden_size=150,
    # source and target embedding size
    embed_size=300,
    # character embedding size
    char_embed_size=30,
    # dropout value
    dropout=0.1,
    # word random dropout
    word_dropout=0.1,
    # label smoothing value
    label_smooth=0.1,
    # gru, lstm, sru or atr
    cell="atr",
    # whether use layer normalization, it will be slow
    layer_norm=False,
    # notice that when opening the swap memory switch
    # you can train reasonably larger batch on condition
    # that your system will use much more cpu memory
    swap_memory=True,

    # whether use character embedding
    use_char=True,
    # whether lowercase word
    lower=False,

    # task name
    task="snli",

    model_name="InferNet",

    # constant batch size at 'batch' mode for batch-based batching
    batch_size=80,
    token_size=2000,
    batch_or_token='token',
    # batch size for decoding, i.e. number of source sentences decoded at the same time
    eval_batch_size=32,
    # whether shuffle batches during training
    shuffle_batch=True,
    # whether use multiprocessing deal with data reading, default true
    data_multiprocessing=True,

    # word vocabulary
    word_vocab_file="",
    # char vocabulary
    char_vocab_file="",
    # pretrained word embedding
    pretrain_word_embedding_file="/path/to/glove",
    # dataset path file
    data_path="",
    # output directory
    output_dir="",
    # output during testing
    test_output="",

    # adam optimizer hyperparameters
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-9,
    # gradient clipping value
    clip_grad_norm=5.0,
    # initial learning rate
    lrate=1e-5,

    # allowed maximum sentence length
    max_len=100,
    # maximum word length
    max_w_len=25,
    # maximum sentence number
    max_p_num=8,
    # hierarchy neural network
    enable_hierarchy=False,

    # maximum epochs
    epoches=10,
    # the effective batch size is: batch/token size * update_cycle
    # sequential update cycle
    update_cycle=1,
    # the number of gpus
    gpus=[0],
    # whether enable ema
    ema_decay=0.9999,

    # print information every disp_freq training steps
    disp_freq=100,
    # evaluate on the development file every eval_freq steps
    eval_freq=10000,
    # save the model parameters every save_freq steps
    save_freq=5000,
    # saved checkpoint number
    checkpoints=5,
    # the maximum training steps, program with stop if epoches or max_training_steps is metted
    max_training_steps=1000,

    # bert configuration
    bert=None,
    bert_dir="",
    tune_bert=False,
    enable_bert=True,
    use_bert_single=True,

    # number of threads for threaded reading, seems useless
    nthreads=6,
    # buffer size controls the number of sentences readed in one time,
    buffer_size=100,
    # a unique queue in multi-thread reading process
    max_queue_size=100,
    # random control, not so well for tensorflow.
    random_seed=1234,
    # whether or not train from checkpoint
    train_continue=True,
)

flags = tf.flags
flags.DEFINE_string("config", "", "Additional Mergable Parameters")
flags.DEFINE_string("parameters", "", "Command Line Refinable Parameters")
flags.DEFINE_string("name", "model", "Description of the training process for distinguishing")
flags.DEFINE_string("mode", "train", "train or test")


# saving model configuration
def save_parameters(params, output_dir):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MkDir(output_dir)

    param_name = os.path.join(output_dir, "param.json")
    with tf.gfile.Open(param_name, "w") as writer:
        tf.logging.info("Saving parameters into {}"
                        .format(param_name))
        writer.write(params.to_json())

    params.word_vocab.save_vocab(os.path.join(output_dir, "vocab.word"))
    if params.use_char:
        params.char_vocab.save_vocab(os.path.join(output_dir, "vocab.char"))


# load model configuration
def load_parameters(params, output_dir):
    param_name = os.path.join(output_dir, "param.json")
    param_name = os.path.abspath(param_name)

    if tf.gfile.Exists(param_name):
        tf.logging.info("Loading parameters from {}"
                        .format(param_name))
        with tf.gfile.Open(param_name, 'r') as reader:
            json_str = reader.readline()
            params.parse_json(json_str)
    return params


# build training process recorder
def setup_recorder(params):
    recorder = Recorder()
    # This is for early stopping, currectly I did not use it
    recorder.bad_counter = 0    # start from 0
    recorder.estop = False

    recorder.lidx = -1      # local data index
    recorder.step = 0       # global step, start from 0
    recorder.epoch = 1      # epoch number, start from 1
    recorder.history_scores = []
    recorder.valid_script_scores = []

    # trying to load saved recorder
    record_path = os.path.join(params.output_dir, "record.json")
    record_path = os.path.abspath(record_path)
    if tf.gfile.Exists(record_path):
        recorder.load_from_json(record_path)

    params.add_hparam('recorder', recorder)
    return params


# print model configuration
def print_parameters(params):
    tf.logging.info("The Used Configuration:")
    for k, v in params.values().items():
        tf.logging.info("%s\t%s", k.ljust(20), str(v).ljust(20))
    tf.logging.info("")


def main(_):
    # set up logger
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info("Welcome Using Zero :)")

    params = global_params

    # try loading parameters
    # priority: command line > saver > default
    # 1. load latest path to load parameters
    if os.path.exists(flags.FLAGS.config):
        params.override_from_dict(eval(open(flags.FLAGS.config).read()))
    params = load_parameters(params, params.output_dir)
    # 2. refine with command line parameters
    if os.path.exists(flags.FLAGS.config):
        params.override_from_dict(eval(open(flags.FLAGS.config).read()))
    params.parse(flags.FLAGS.parameters)

    # set up random seed
    random.seed(params.random_seed)
    np.random.seed(params.random_seed)
    tf.set_random_seed(params.random_seed)

    # loading vocabulary
    tf.logging.info("Begin Loading Vocabulary")
    start_time = time.time()
    full_task = get_task(params, True)
    if not os.path.exists(params.word_vocab_file):
        params.word_vocab = Vocab(lower=params.lower)
        params.word_vocab.make_vocab(full_task,
                                     use_char=False, embedding_path=params.pretrain_word_embedding_file)
    else:
        params.word_vocab = Vocab(lower=params.lower, vocab_file=params.word_vocab_file)
    if params.use_char:
        if not os.path.exists(params.char_vocab_file):
            params.char_vocab = Vocab(lower=False)
            params.char_vocab.make_vocab(full_task,
                                         use_char=True, embedding_path=None)
        else:
            params.char_vocab = Vocab(lower=False, vocab_file=params.char_vocab_file)

    tf.logging.info("End Loading Vocabulary, Word Vocab Size {}, "
                    "Char Vocab Size {}, within {} seconds"
                    .format(params.word_vocab.size(),
                            params.char_vocab.size() if params.use_char else 0,
                            time.time() - start_time))

    if flags.FLAGS.mode == "vocab":
        save_parameters(params, params.output_dir)
        return

    # save parameters
    if flags.FLAGS.mode == "train":
        save_parameters(params, params.output_dir)

    # loading bert config
    if params.enable_bert:
        bert_config = bert.load_config(params.bert_dir)
        params.bert = tc.training.HParams(**bert_config)

        # loading vocabulary
        tf.logging.info("Begin Loading Vocabulary")
        start_time = time.time()
        params.bert.vocab = bert.load_vocab(params.bert_dir)
        tf.logging.info("End Loading Vocabulary, Vocab Size {}, within {} seconds"
                        .format(params.bert.vocab.size,
                                time.time() - start_time))

    # loading task label information
    params.label_size = full_task.get_label_size()

    # print parameters
    print_parameters(params)

    # print the used datasets
    tf.logging.info("Task {} is performed with data {}".format(params.task, full_task.data_path))

    mode = flags.FLAGS.mode
    if mode == "train":
        # load the recorder
        params = setup_recorder(params)

        graph.train(params)
    elif mode == "test":
        graph.evaluate(params)
    else:
        tf.logging.error("Invalid mode: {}".format(mode))


if __name__ == '__main__':
    tf.app.run()
