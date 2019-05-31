# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import tensorflow as tf

from .vocab import Vocab


def load_vocab(model_dir):
    vocab = Vocab(
        vocab_file=os.path.join(model_dir, 'vocab.txt')
    )
    return vocab


def load_config(model_dir):
    with tf.gfile.GFile(
            os.path.join(model_dir, 'bert_config.json'),
            "r"
    ) as reader:
      text = reader.read()
    return json.loads(text)


def load_model(session, model_dir):
    tf.logging.warn("Starting Loading BERT Pre-trained Model")
    ops = []
    reader = tf.train.load_checkpoint(
        os.path.join(model_dir, "bert_model.ckpt")
    )

    for var in tf.global_variables():
        name = var.op.name
        name = name[name.find('/bert/')+1:]

        if reader.has_tensor(name) and 'Adam' not in name:
            tf.logging.info('{} **Good**'.format(name))
            ops.append(
                tf.assign(var, reader.get_tensor(name)))
        else:
            tf.logging.warn("{} --Bad--".format(name))
    restore_op = tf.group(*ops, name="restore_global_vars")
    session.run(restore_op)
