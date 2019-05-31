# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib as tc

from . import encoder
from utils import util


def bert_encoder(sequence, params):

    # extract sequence mask information
    seq_mask = 1. - tf.to_float(tf.equal(sequence, params.bert.vocab.pad))

    # extract segment information
    seg_pos = tf.to_float(tf.equal(sequence, params.bert.vocab.sep))
    seg_ids = tf.cumsum(seg_pos, axis=1, reverse=True)
    seg_num = tf.reduce_sum(seg_pos, axis=1, keepdims=True)
    seg_ids = seg_num - seg_ids
    seg_ids = tf.to_int32(seg_ids * seq_mask)

    # sequence length information
    seq_shp = util.shape_list(sequence)
    batch_size, seq_length = seq_shp[:2]

    def custom_getter(getter, name, *args, **kwargs):
        kwargs['trainable'] = params.tune_bert
        return getter(name, *args, **kwargs)

    with tf.variable_scope("bert", custom_getter=custom_getter):

        # handling sequence embeddings: token_embedding pls segment embedding pls positional embedding
        embed_initializer = tf.truncated_normal_initializer(stddev=params.bert.initializer_range)
        with tf.variable_scope("embeddings"):
            word_embedding = tf.get_variable(
                name="word_embeddings",
                shape=[params.bert.vocab.size, params.bert.hidden_size],
                initializer=embed_initializer
            )
            seq_embed = tf.nn.embedding_lookup(word_embedding, sequence)

            segment_embedding = tf.get_variable(
                name="token_type_embeddings",
                shape=[2, params.bert.hidden_size],
                initializer=embed_initializer
            )
            seg_embed = tf.nn.embedding_lookup(segment_embedding, seg_ids)

            # word embedding + segment embedding
            seq_embed = seq_embed + seg_embed

            # add position embedding
            assert_op = tf.assert_less_equal(seq_length, params.bert.max_position_embeddings)
            with tf.control_dependencies([assert_op]):
                position_embedding = tf.get_variable(
                    name="position_embeddings",
                    shape=[params.bert.max_position_embeddings, params.bert.hidden_size],
                    initializer=embed_initializer
                )
                pos_embed = position_embedding[:seq_length]

                seq_embed = seq_embed + tf.expand_dims(pos_embed, 0)

            # post-processing, layer norm and segmentation
            seq_embed = tc.layers.layer_norm(
                inputs=seq_embed, begin_norm_axis=-1, begin_params_axis=-1)

            seq_embed = util.valid_apply_dropout(seq_embed, params.bert.hidden_dropout_prob)

        bert_outputs = []

        #  handling sequence encoding with transformer encoder
        with tf.variable_scope("encoder"):
            attention_mask = encoder.create_attention_mask_from_input_mask(
                sequence, seq_mask)

            # Run the stacked transformer.
            # `sequence_output` shape = [batch_size, seq_length, hidden_size].
            all_encoder_layers = encoder.transformer_model(
                input_tensor=seq_embed,
                attention_mask=attention_mask,
                hidden_size=params.bert.hidden_size,
                num_hidden_layers=params.bert.num_hidden_layers,
                num_attention_heads=params.bert.num_attention_heads,
                intermediate_size=params.bert.intermediate_size,
                intermediate_act_fn=encoder.get_activation(params.bert.hidden_act),
                hidden_dropout_prob=params.bert.hidden_dropout_prob,
                attention_probs_dropout_prob=params.bert.attention_probs_dropout_prob,
                initializer_range=params.bert.initializer_range,
                do_return_all_layers=True)

        sequence_output = all_encoder_layers

        bert_outputs.append(sequence_output)

        if params.use_bert_single:
            # The "pooler" converts the encoded sequence tensor of shape
            # [batch_size, seq_length, hidden_size] to a tensor of shape
            # [batch_size, hidden_size]. This is necessary for segment-level
            # (or segment-pair-level) classification tasks where we need a fixed
            # dimensional representation of the segment.
            with tf.variable_scope("pooler"):
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token. We assume that this has been pre-trained
                first_token_tensor = tf.squeeze(sequence_output[-1][:, 0:1, :], axis=1)
                pooled_output = tf.layers.dense(
                    first_token_tensor,
                    params.bert.hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=embed_initializer)

                bert_outputs.append(pooled_output)

        return bert_outputs


def bert_feature(layer_features, scope=None):
    '''
    with tf.variable_scope(scope or "fuse_bert_seq_feature"):
        layer_features = [tf.expand_dims(feature, 0) for feature in layer_features]
        layer_features = tf.concat(layer_features, 0)

        layer_logits = tf.layers.dense(layer_features, 1)
        layer_weights = tf.nn.softmax(layer_logits, 0)

        layer_feature = tf.reduce_sum(layer_weights * layer_features, 0)

        return layer_feature
    '''
    return layer_features[-1]
