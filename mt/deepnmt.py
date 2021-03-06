# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf

import func
from utils import util
from rnns import rnn


def encoder(source, params):
    mask = tf.to_float(tf.cast(source, tf.bool))
    hidden_size = params.hidden_size

    source, mask = util.remove_invalid_seq(source, mask)

    embed_name = "embedding" if params.shared_source_target_embedding \
        else "src_embedding"
    src_emb = tf.get_variable(embed_name,
                              [params.src_vocab.size(), params.embed_size])
    src_bias = tf.get_variable("bias", [params.embed_size])

    inputs = tf.gather(src_emb, source)
    inputs = tf.nn.bias_add(inputs, src_bias)

    if util.valid_dropout(params.dropout):
        inputs = tf.nn.dropout(inputs, 1. - params.dropout)

    with tf.variable_scope("encoder"):
        x = inputs

        for layer in range(params.num_encoder_layer):
            with tf.variable_scope("layer_{}".format(layer)):
                # forward rnn
                with tf.variable_scope('forward'):
                    outputs = rnn.rnn(params.cell, x, hidden_size, mask=mask,
                                      ln=params.layer_norm, sm=params.swap_memory,
                                      dp=params.dropout)
                    output_fw, state_fw = outputs[1]
                if layer == 0:
                    # backward rnn
                    with tf.variable_scope('backward'):
                        if not params.caencoder:
                            outputs = rnn.rnn(params.cell, tf.reverse(x, [1]),
                                              hidden_size, mask=tf.reverse(mask, [1]),
                                              ln=params.layer_norm, sm=params.swap_memory,
                                              dp=params.dropout)
                            output_bw, state_bw = outputs[1]
                        else:
                            outputs = rnn.cond_rnn(params.cell, tf.reverse(x, [1]),
                                                   tf.reverse(output_fw, [1]), hidden_size,
                                                   mask=tf.reverse(mask, [1]),
                                                   ln=params.layer_norm,
                                                   sm=params.swap_memory,
                                                   num_heads=params.num_heads,
                                                   one2one=True)
                            output_bw, state_bw = outputs[1]

                        output_bw = tf.reverse(output_bw, [1])

                    if not params.caencoder:
                        y = tf.concat([output_fw, output_bw], -1)
                        z = tf.concat([state_fw, state_bw], -1)
                    else:
                        y = output_bw
                        z = state_bw
                else:
                    y = output_fw
                    z = state_fw

                y = func.linear(y, hidden_size, ln=False, scope="ff")

                # short cut via residual connection
                if x.get_shape()[-1].value == y.get_shape()[-1].value:
                    x = func.residual_fn(x, y, dropout=params.dropout)
                else:
                    x = y
                if params.layer_norm:
                    x = func.layer_norm(x, scope="ln")

    with tf.variable_scope("decoder_initializer"):
        decoder_cell = rnn.get_cell(
            params.cell, hidden_size, ln=params.layer_norm
        )

    return {
        "encodes": x,
        "decoder_initializer": {
            "layer_{}".format(l):
                decoder_cell.get_init_state(
                    x=z, scope="layer_{}".format(l))
            for l in range(params.num_decoder_layer)
        },
        "mask": mask
    }


def decoder(target, state, params):
    mask = tf.to_float(tf.cast(target, tf.bool))
    hidden_size = params.hidden_size

    if 'decoder' not in state:
        target, mask = util.remove_invalid_seq(target, mask)

    embed_name = "embedding" if params.shared_source_target_embedding \
        else "tgt_embedding"
    tgt_emb = tf.get_variable(embed_name,
                              [params.tgt_vocab.size(), params.embed_size])
    tgt_bias = tf.get_variable("bias", [params.embed_size])

    inputs = tf.gather(tgt_emb, target)
    inputs = tf.nn.bias_add(inputs, tgt_bias)

    # shift
    if 'decoder' not in state:
        inputs = tf.pad(inputs, [[0, 0], [1, 0], [0, 0]])
        inputs = inputs[:, :-1, :]
    else:
        inputs = tf.cond(tf.reduce_all(tf.equal(target, params.tgt_vocab.pad())),
                         lambda: tf.zeros_like(inputs),
                         lambda: inputs)
        mask = tf.ones_like(mask)

    if util.valid_dropout(params.dropout):
        inputs = tf.nn.dropout(inputs, 1. - params.dropout)

    with tf.variable_scope("decoder"):
        x = inputs
        for layer in range(params.num_decoder_layer):
            with tf.variable_scope("layer_{}".format(layer)):
                init_state = state["decoder_initializer"]["layer_{}".format(layer)]
                if 'decoder' in state:
                    init_state = state["decoder"]["state"]["layer_{}".format(layer)]
                if layer == 0 or params.use_deep_att:
                    returns = rnn.cond_rnn(params.cell, x, state["encodes"], hidden_size,
                                           init_state=init_state, mask=mask,
                                           num_heads=params.num_heads,
                                           mem_mask=state["mask"], ln=params.layer_norm,
                                           sm=params.swap_memory, one2one=False,
                                           dp=params.dropout)
                    (_, hidden_state), (outputs, _), contexts, attentions = returns
                    c = contexts
                else:
                    if params.caencoder:
                        returns = rnn.cond_rnn(params.cell, x, c,
                                               hidden_size, init_state=init_state,
                                               mask=mask, mem_mask=mask,
                                               ln=params.layer_norm,
                                               sm=params.swap_memory,
                                               num_heads=params.num_heads,
                                               one2one=True, dp=params.dropout)
                        (_, hidden_state), (outputs, _), contexts, attentions = returns
                    else:
                        outputs = rnn.rnn(params.cell, tf.concat([x, c], -1),
                                          hidden_size, mask=mask, init_state=init_state,
                                          ln=params.layer_norm, sm=params.swap_memory,
                                          dp=params.dropout)
                        outputs, hidden_state = outputs[1]
                if 'decoder' in state:
                    state['decoder']['state']['layer_{}'.format(layer)] = hidden_state

                y = func.linear(outputs, hidden_size, ln=False, scope="ff")

                # short cut via residual connection
                if x.get_shape()[-1].value == y.get_shape()[-1].value:
                    x = func.residual_fn(x, y, dropout=params.dropout)
                else:
                    x = y
                if params.layer_norm:
                    x = func.layer_norm(x, scope="ln")

    feature = func.linear(tf.concat([x, c], -1), params.embed_size, ln=params.layer_norm, scope="ff")
    feature = tf.nn.tanh(feature)

    if util.valid_dropout(params.dropout):
        feature = tf.nn.dropout(feature, 1. - params.dropout)

    if 'dev_decode' in state:
        feature = x[:, -1, :]

    embed_name = "tgt_embedding" if params.shared_target_softmax_embedding \
        else "softmax_embedding"
    embed_name = "embedding" if params.shared_source_target_embedding \
        else embed_name
    softmax_emb = tf.get_variable(embed_name,
                                  [params.tgt_vocab.size(), params.embed_size])
    feature = tf.reshape(feature, [-1, params.embed_size])
    logits = tf.matmul(feature, softmax_emb, False, True)

    soft_label, normalizer = util.label_smooth(
        target,
        util.shape_list(logits)[-1],
        factor=params.label_smooth)
    centropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits,
        labels=soft_label
    )
    centropy -= normalizer
    centropy = tf.reshape(centropy, tf.shape(target))

    loss = tf.reduce_sum(centropy * mask, -1) / tf.reduce_sum(mask, -1)
    loss = tf.reduce_mean(loss)

    # these mask tricks mainly used to deal with zero shapes, such as [0, 1]
    loss = tf.cond(tf.equal(tf.shape(target)[0], 0),
                   lambda: tf.constant(0, dtype=tf.float32),
                   lambda: loss)

    return loss, logits, state


def train_fn(features, params, initializer=None):
    with tf.variable_scope(params.model_name or "model",
                           initializer=initializer,
                           reuse=tf.AUTO_REUSE):
        state = encoder(features['source'], params)
        loss, logits, state = decoder(features['target'], state, params)

        return {
            "loss": loss
        }


def infer_fn(params):
    params = copy.copy(params)
    params = util.closing_dropout(params)

    def encoding_fn(source):
        with tf.variable_scope(params.model_name or "model",
                               reuse=tf.AUTO_REUSE):
            state = encoder(source, params)
            state["decoder"] = {
                "state": state["decoder_initializer"]
            }
            return state

    def decoding_fn(target, state, time):
        with tf.variable_scope(params.model_name or "model",
                               reuse=tf.AUTO_REUSE):
            if params.search_mode == "cache":
                step_loss, step_logits, step_state = decoder(
                    target, state, params)
            else:
                estate = encoder(state, params)
                estate['dev_decode'] = True
                _, step_logits, _ = decoder(target, estate, params)
                step_state = state

            return step_logits, step_state

    return encoding_fn, decoding_fn
