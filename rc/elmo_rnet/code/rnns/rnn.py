# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from rnns import get_cell


def rnn(cell_name, x, d, mask=None, ln=False, init_state=None, sm=True):
    """Self implemented RNN procedure, supporting mask trick"""
    # cell_name: gru, lstm or atr
    # x: input sequence embedding matrix, [batch, seq_len, dim]
    # d: hidden dimension for rnn
    # mask: mask matrix, [batch, seq_len]
    # ln: whether use layer normalization
    # init_state: the initial hidden states, for cache purpose
    # sm: whether apply swap memory during rnn scan
    # dp: variational dropout

    in_shape = tf.shape(x)
    batch_size, time_steps = in_shape[0], in_shape[1]

    cell = get_cell(cell_name, d, ln=ln)

    if init_state is None:
        init_state = cell.get_init_state(shape=[batch_size])
    if mask is None:
        mask = tf.ones([batch_size, time_steps], tf.float32)

    # prepare projected input
    cache_inputs = cell.fetch_states(x)
    cache_inputs = [tf.transpose(v, [1, 0, 2])
                    for v in list(cache_inputs)]
    mask_ta = tf.transpose(tf.expand_dims(mask, -1), [1, 0, 2])

    def _step_fn(prev, x):
        t, h_ = prev
        m = x[-1]
        v = x[:-1]

        h = cell(h_, v)
        h = m * h + (1. - m) * h_

        return t + 1, h

    time = tf.constant(0, dtype=tf.int32, name="time")
    step_states = (time, init_state)
    step_vars = cache_inputs + [mask_ta]

    outputs = tf.scan(_step_fn,
                      step_vars,
                      initializer=step_states,
                      parallel_iterations=32,
                      swap_memory=sm)

    output_ta = outputs[1]
    output_state = outputs[1][-1]

    outputs = tf.transpose(output_ta, [1, 0, 2])

    return (outputs, output_state), \
           (cell.get_hidden(outputs), cell.get_hidden(output_state))
