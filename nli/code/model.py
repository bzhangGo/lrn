# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import bert
import numpy as np
import tensorflow as tf

import func
from rnns import rnn
from utils import util


def wrap_rnn(x, cell_type, nlayers, hidden_size, mask=None, bidir=True,
             use_ln=True, concat=True, dropout=0.0, scope=None):
    outputs = [x]
    states = []

    if mask is None:
        xshp = util.shape_list(x)
        mask = tf.ones([xshp[0], xshp[1]], tf.float32)

    for layer in range(nlayers):
        with tf.variable_scope("{}_layer_{}".format(scope or 'rnn', layer)):
            with tf.variable_scope("fw_rnn"):
                _, (o_fw, o_fw_s) = rnn.rnn(
                    cell_type, outputs[-1], hidden_size,
                    mask=mask, ln=use_ln)
            if bidir:
                with tf.variable_scope("bw_rnn"):
                    _, (o_bw, o_bw_s) = rnn.rnn(
                        cell_type, tf.reverse(outputs[-1], [1]), hidden_size,
                        mask=tf.reverse(mask, [1]), ln=use_ln)
                    o_bw = tf.reverse(o_bw, [1])

            if layer != nlayers - 1:
                o_fw = util.valid_apply_dropout(o_fw, dropout)
                o_fw_s = util.valid_apply_dropout(o_fw_s, dropout)

                if bidir:
                    o_bw = util.valid_apply_dropout(o_bw, dropout)
                    o_bw_s = util.valid_apply_dropout(o_bw_s, dropout)

            if not bidir:
                outputs.append(o_fw)
                states.append(o_fw_s)
            else:
                outputs.append(tf.concat([o_fw, o_bw], -1))
                states.append(tf.concat([o_fw_s, o_bw_s], -1))

    if concat:
        return tf.concat(outputs[1:], -1), tf.concat(states, -1)
    else:
        return outputs[-1], states[-1]


def tensor2vector(tensor, hidden_size, mask=None, init=None,
                  use_ln=False, dropout=0.1, scope="vecatt"):
    with tf.variable_scope(scope):
        if util.valid_dropout(dropout):
            tensor = tf.nn.dropout(tensor, 1. - dropout)

        if init is None:
            m = tf.nn.tanh(func.linear(tensor, hidden_size, ln=use_ln, scope="m_tensor"))
        else:
            init = util.expand_tile_dims(init, tf.shape(tensor)[1], 1)
            if util.valid_dropout(dropout):
                init = tf.nn.dropout(init, 1. - dropout)

            m = tf.nn.tanh(func.linear(tensor, hidden_size, ln=use_ln, scope="m_tensor") +
                           func.linear(init, hidden_size, scope="m_init"))
        s = func.linear(m, 1, bias=False, scope="sore")

        if mask is None:
            mask = tf.ones([tf.shape(tensor)[0], tf.shape(tensor)[1]], tf.float32)
        s = tf.squeeze(s, -1) + (1. - mask) * (-1e9)
        w = tf.nn.softmax(s)

        return tf.reduce_sum(tf.expand_dims(w, 2) * tensor, axis=1), s


def embedding_layer(features, params):
    p = features['p']
    h = features['h']

    p_mask = tf.to_float(tf.cast(p, tf.bool))
    h_mask = tf.to_float(tf.cast(h, tf.bool))

    with tf.device('/cpu:0'):
        symbol_embeddings = tf.get_variable('special_symbol_embeddings',
                                            shape=(3, params.embed_size),
                                            trainable=True)
        embedding_initializer = tf.glorot_uniform_initializer()
        if tf.gfile.Exists(params.pretrain_word_embedding_file):
            pretrain_embedding = np.load(params.pretrain_word_embedding_file)['data']
            embedding_initializer = tf.constant_initializer(pretrain_embedding)
        general_embeddings = tf.get_variable('general_symbol_embeddings',
                                             shape=(params.word_vocab.size() - 3, params.embed_size),
                                             initializer=embedding_initializer,
                                             trainable=False)
        word_embeddings = tf.concat([symbol_embeddings, general_embeddings], 0)

        p_emb = tf.nn.embedding_lookup(word_embeddings, p)
        h_emb = tf.nn.embedding_lookup(word_embeddings, h)

    p_features = [p_emb]
    h_features = [h_emb]

    if params.enable_bert:
        p_features.append(features['bert_p_enc'])
        h_features.append(features['bert_h_enc'])

    if params.use_char:
        pc = features['pc']
        hc = features['hc']

        pc_mask = tf.to_float(tf.cast(pc, tf.bool))
        hc_mask = tf.to_float(tf.cast(hc, tf.bool))

        pc = tf.reshape(pc, [-1, tf.shape(pc)[-1]])
        hc = tf.reshape(hc, [-1, tf.shape(hc)[-1]])
        pc_mask = tf.reshape(pc_mask, [-1, tf.shape(pc_mask)[-1]])
        hc_mask = tf.reshape(hc_mask, [-1, tf.shape(hc_mask)[-1]])
        with tf.device('/cpu:0'):
            char_embeddings = tf.get_variable('char_embeddings',
                                              shape=(params.char_vocab.size(), params.char_embed_size),
                                              initializer=tf.glorot_uniform_initializer(),
                                              trainable=True)
            with tf.variable_scope('char_embedding'):
                pc_emb = tf.nn.embedding_lookup(char_embeddings, pc)
                hc_emb = tf.nn.embedding_lookup(char_embeddings, hc)
                if util.valid_dropout(params.dropout):
                    pc_emb = tf.nn.dropout(pc_emb, 1. - 0.5 * params.dropout)
                    hc_emb = tf.nn.dropout(hc_emb, 1. - 0.5 * params.dropout)

        with tf.variable_scope("char_encoding", reuse=tf.AUTO_REUSE):
            pc_emb = pc_emb * tf.expand_dims(pc_mask, -1)
            hc_emb = hc_emb * tf.expand_dims(hc_mask, -1)

            pc_shp = util.shape_list(features['pc'])
            pc_emb = tf.reshape(pc_emb, [pc_shp[0], pc_shp[1], pc_shp[2], params.char_embed_size])
            hc_shp = util.shape_list(features['hc'])
            hc_emb = tf.reshape(hc_emb, [hc_shp[0], hc_shp[1], hc_shp[2], params.char_embed_size])

            pc_state = func.linear(tf.reduce_max(pc_emb, 2), params.char_embed_size, scope="cmap")
            hc_state = func.linear(tf.reduce_max(hc_emb, 2), params.char_embed_size, scope="cmap")

        p_features.append(pc_state)
        h_features.append(hc_state)

    '''
    p_emb = func.highway(tf.concat(p_features, axis=2),
                         size=params.hidden_size, dropout=params.dropout, num_layers=2, scope='highway')
    h_emb = func.highway(tf.concat(h_features, axis=2),
                         size=params.hidden_size, dropout=params.dropout, num_layers=2, scope='highway')
    '''
    p_emb = tf.concat(p_features, axis=2)
    h_emb = tf.concat(h_features, axis=2)

    p_emb = p_emb * tf.expand_dims(p_mask, -1)
    h_emb = h_emb * tf.expand_dims(h_mask, -1)

    features.update({'p_emb': p_emb,
                     'h_emb': h_emb,
                     'p_mask': p_mask,
                     'h_mask': h_mask,
                     })
    return features


def match_layer(features, params):
    with tf.variable_scope("match", reuse=tf.AUTO_REUSE):
        p_emb = features["p_emb"]
        h_emb = features["h_emb"]
        p_mask = features["p_mask"]
        h_mask = features["h_mask"]

        p_seq_enc, p_vec_enc = wrap_rnn(
            p_emb,
            params.cell,
            1,
            params.hidden_size,
            mask=p_mask,
            use_ln=params.layer_norm,
            dropout=params.dropout,
            scope="enc_p"
        )

        with tf.variable_scope("h_init"):
            h_init = rnn.get_cell(
                params.cell,
                params.hidden_size,
                ln=params.layer_norm
            ).get_init_state(x=p_vec_enc)
            h_init = tf.tanh(h_init)

        _, (h_seq_enc, h_vec_enc), _, _ = rnn.cond_rnn(
            params.cell,
            h_emb,
            p_seq_enc,
            params.hidden_size,
            init_state=h_init,
            mask=h_mask,
            mem_mask=p_mask,
            ln=params.layer_norm,
            num_heads=params.num_heads
        )

        p_encs = [p_seq_enc]
        h_encs = [h_seq_enc]

        if params.enable_bert:
            p_encs.append(features['bert_p_enc'])
            h_encs.append(features['bert_h_enc'])

        p_enc = tf.concat(p_encs, -1)
        h_enc = tf.concat(h_encs, -1)

        p_seq_enc, _ = wrap_rnn(
            p_enc,
            params.cell,
            1,
            params.hidden_size,
            mask=p_mask,
            use_ln=params.layer_norm,
            dropout=params.dropout,
            scope="post_enc_p"
        )

        h_seq_enc, _ = wrap_rnn(
            h_enc,
            params.cell,
            1,
            params.hidden_size,
            mask=h_mask,
            use_ln=params.layer_norm,
            dropout=params.dropout,
            scope="post_enc_h"
        )

        features.update({
            'p_enc': p_seq_enc,
            'h_enc': h_seq_enc
        })

    return features


def loss_layer(features, params):
    p_enc = features['p_enc']
    h_enc = features['h_enc']

    p_mask = features['p_mask']
    h_mask = features['h_mask']

    feature_list = [
            tensor2vector(p_enc, params.hidden_size, mask=p_mask, scope="p_att")[0],
            tensor2vector(h_enc, params.hidden_size, mask=h_mask, scope="h_att")[0],
    ]
    if params.enable_bert:
        feature_list.append(features['feature'])

    feature = tf.concat(feature_list, -1)

    label_logits = func.linear(feature, params.label_size, ln=params.layer_norm, scope="label")

    def celoss(logits, labels):
        soft_label, normalizer = util.label_smooth(
            labels,
            util.shape_list(logits)[-1],
            factor=params.label_smooth)
        centropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits,
            labels=soft_label
        )
        centropy -= normalizer
        centropy = tf.reshape(centropy, tf.shape(labels))

        return tf.reduce_mean(centropy)

    loss = celoss(label_logits, features['l'])

    if params.weight_decay > 0:
        with tf.variable_scope('l2_loss'):
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
        loss += params.weight_decay * l2_loss

    features.update({
        'loss': loss
    })

    return features, tf.argmax(label_logits, -1)


def graph(features, params):
    if params.enable_bert:
        ps = features['ps']
        hs = features['hs']

        bert_input = tf.concat([ps, hs], 1)
        sequence_output = bert.bert_encoder(bert_input, params)
        sequence_feature = bert.bert_feature(sequence_output[0])

        p_len = tf.shape(ps)[1]
        # 1: remove the encoding for `cls`
        p_enc = sequence_feature[:, 1:p_len, :]
        h_enc = sequence_feature[:, p_len:, :]

        pb = features['pb']
        hb = features['hb']

        pb_shp = util.shape_list(pb)
        hb_shp = util.shape_list(hb)

        p_coord = tf.stack(
            [util.batch_coordinates(pb_shp[0], pb_shp[1]), pb],
            axis=2
        )
        p_enc = tf.gather_nd(p_enc, p_coord)

        h_coord = tf.stack(
            [util.batch_coordinates(hb_shp[0], hb_shp[1]), hb],
            axis=2
        )
        h_enc = tf.gather_nd(h_enc, h_coord)

        features['bert_p_enc'] = util.valid_apply_dropout(p_enc, params.dropout)
        features['bert_h_enc'] = util.valid_apply_dropout(h_enc, params.dropout)
        if not params.use_bert_single:
            features['feature'] = sequence_feature[:, 0, :]
        else:
            features['feature'] = sequence_output[1]

    features = embedding_layer(features, params)
    features = match_layer(features, params)
    features = loss_layer(features, params)

    return features


def train_fn(features, params, initializer=None):
    with tf.variable_scope(params.model_name or "model",
                           initializer=initializer,
                           reuse=tf.AUTO_REUSE):
        outputs, _ = graph(features, params)

        return {
            "loss": outputs['loss'],
        }


def infer_fn(params, features):
    params = copy.copy(params)
    params = util.closing_dropout(params)
    if params.enable_bert:
        util.closing_dropout(params.bert)

    with tf.variable_scope(params.model_name or "model",
                           reuse=tf.AUTO_REUSE):
        _, outputs = graph(features, params)

        return outputs
