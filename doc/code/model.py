# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf

import bert
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
                    mask=mask, ln=use_ln, sm=False)
            if bidir:
                with tf.variable_scope("bw_rnn"):
                    _, (o_bw, o_bw_s) = rnn.rnn(
                        cell_type, tf.reverse(outputs[-1], [1]), hidden_size,
                        mask=tf.reverse(mask, [1]), ln=use_ln, sm=False)
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
    t = features['t']

    t_mask = tf.to_float(tf.cast(t, tf.bool))

    with tf.device('/cpu:0'):
        symbol_embeddings = tf.get_variable('special_symbol_embeddings',
                                            shape=(3, params.embed_size),
                                            trainable=True)
        embedding_initializer = tf.glorot_uniform_initializer()
        if params.word_vocab.pretrained_embedding is not None:
            pretrain_embedding = params.word_vocab.pretrained_embedding
            embedding_initializer = tf.constant_initializer(pretrain_embedding)
        general_embeddings = tf.get_variable('general_symbol_embeddings',
                                             shape=(params.word_vocab.size() - 3, params.embed_size),
                                             initializer=embedding_initializer,
                                             trainable=params.word_vocab.pretrained_embedding is None)
        word_embeddings = tf.concat([symbol_embeddings, general_embeddings], 0)

        # apply word dropout
        wd_mask = util.valid_apply_dropout(t_mask, params.word_dropout)
        wd_mask = tf.to_float(tf.cast(wd_mask, tf.bool))

        t_emb = tf.nn.embedding_lookup(word_embeddings, t * tf.to_int32(wd_mask))
        t_emb = t_emb * tf.expand_dims(t_mask, -1)

    embed_features = [t_emb]

    if params.enable_bert:
        embed_features.append(features['bert_enc'])

    if params.use_char:
        c = features['c']
        c_mask = tf.to_float(tf.cast(c, tf.bool))

        c = tf.reshape(c, [-1, tf.shape(c)[-1]])
        c_mask = tf.reshape(c_mask, [-1, tf.shape(c_mask)[-1]])

        with tf.device('/cpu:0'):
            char_embeddings = tf.get_variable('char_embeddings',
                                              shape=(params.char_vocab.size(), params.char_embed_size),
                                              initializer=tf.glorot_uniform_initializer(),
                                              trainable=True)
            with tf.variable_scope('char_embedding'):
                c_emb = tf.nn.embedding_lookup(char_embeddings, c)
                c_emb = util.valid_apply_dropout(c_emb, 0.5 * params.dropout)

        with tf.variable_scope("char_encoding", reuse=tf.AUTO_REUSE):
            c_emb = c_emb * tf.expand_dims(c_mask, -1)

            c_shp = util.shape_list(features['c'])
            c_emb = tf.reshape(c_emb, [c_shp[0], c_shp[1], c_shp[2], params.char_embed_size])

            c_state = func.linear(tf.reduce_max(c_emb, 2), params.char_embed_size, scope="cmap")

        embed_features.append(c_state)

    t_emb = tf.concat(embed_features, axis=2) * tf.expand_dims(t_mask, -1)

    features.update({'t_emb': t_emb,
                     't_mask': t_mask,
                     })
    return features


def hierarchy_layer(features, params):
    with tf.variable_scope("hierarchy", reuse=tf.AUTO_REUSE):

        with tf.variable_scope("word_level"):
            word_enc, word_vec = wrap_rnn(
                features['t_emb'],
                params.cell,
                1,
                params.hidden_size,
                mask=features['t_mask'],
                use_ln=params.layer_norm,
                dropout=params.dropout,
                scope="rnn_enc"
            )

            word_vec, _ = tensor2vector(
                word_enc,
                params.hidden_size,
                mask=features['t_mask'],
                init=word_vec,
                use_ln=params.layer_norm,
                dropout=params.dropout,
                scope="att"
            )

            o = word_vec

        if params.enable_hierarchy:
            with tf.variable_scope("sent_level"):
                s_mask = tf.to_float(tf.cast(tf.reduce_sum(features['t_mask'], 1), tf.bool))
                batch_size = tf.shape(features['l'])[0]

                s_mask = tf.reshape(s_mask, [batch_size, -1])
                word_vec = tf.reshape(word_vec, [batch_size, -1, word_vec.get_shape().as_list()[-1]])

                sent_enc, sent_vec = wrap_rnn(
                    word_vec,
                    params.cell,
                    1,
                    params.hidden_size,
                    mask=s_mask,
                    use_ln=params.layer_norm,
                    dropout=params.dropout,
                    scope="rnn_enc"
                )

                sent_vec, _ = tensor2vector(
                    sent_enc,
                    params.hidden_size,
                    mask=s_mask,
                    init=sent_vec,
                    use_ln=params.layer_norm,
                    dropout=params.dropout,
                    scope="att"
                )

                o = sent_vec

        features.update({
            't_enc': o
        })

    return features


def loss_layer(features, params):
    t_enc = features['t_enc']

    feature = [t_enc]

    if params.enable_bert:
        s_mask = tf.to_float(tf.cast(tf.reduce_sum(features['t_mask'], 1), tf.bool))
        batch_size = tf.shape(features['l'])[0]

        s_mask = tf.reshape(s_mask, [batch_size, -1])
        bert_feature = features['feature']

        bert_feature = tf.reshape(bert_feature, [batch_size, -1, bert_feature.get_shape().as_list()[-1]])
        bert_vec, _ = tensor2vector(
            bert_feature,
            params.hidden_size,
            mask=s_mask,
            use_ln=params.layer_norm,
            dropout=params.dropout,
            scope="bert_att"
        )
        feature.append(bert_vec)

    feature = tf.concat(feature, -1)

    label_logits = func.linear(feature, params.label_size, ln=params.layer_norm, scope="label")

    # multi-label classification-based objective
    def mlceloss(logits, labels):
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

    loss = mlceloss(label_logits, features['l'])

    if params.weight_decay > 0:
        with tf.variable_scope('l2_loss'):
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
        loss += params.weight_decay * l2_loss

    features.update({
        'loss': loss
    })

    prediction = tf.argmax(label_logits, -1)
    label_output = tf.nn.softmax(label_logits, -1)

    return features, prediction, label_output


def graph(features, params):
    if params.enable_bert:
        s = features['s']

        bert_input = s
        sequence_output = bert.bert_encoder(bert_input, params)
        s_enc = tf.concat(sequence_output[0][-4:], -1)[:, 1:, :]

        sb = features['sb']
        sb_shp = util.shape_list(sb)

        s_coord = tf.stack(
            [util.batch_coordinates(sb_shp[0], sb_shp[1]), sb],
            axis=2
        )
        s_enc = tf.gather_nd(s_enc, s_coord)

        features['bert_enc'] = util.valid_apply_dropout(s_enc, params.dropout)
        if not params.use_bert_single:
            features['feature'] = s_enc[:, 0, :]
        else:
            features['feature'] = sequence_output[1]

    features = embedding_layer(features, params)
    features = hierarchy_layer(features, params)
    graph_output = loss_layer(features, params)

    return graph_output


def train_fn(features, params, initializer=None):
    params = copy.copy(params)
    if params.enable_bert:
        util.closing_dropout(params.bert)

    with tf.variable_scope(params.model_name or "model",
                           initializer=initializer,
                           reuse=tf.AUTO_REUSE):
        outputs, _, _ = graph(features, params)

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
        outputs = graph(features, params)

        return outputs[1:]
