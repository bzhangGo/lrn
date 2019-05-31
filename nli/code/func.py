# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from utils import util


def linear(x, dim, bias=True, ln=False,
           weight_initializer=None,
           bias_initializer=tf.zeros_initializer(),
           scope=None):
    """
    basic linear or feed forward layer
    :param x: input tensor or list
    :param dim: output dimension or list
    :param bias: whether use bias term
    :param ln: whether use layer normalization
    :param weight_initializer: you can set it if you want
    :param bias_initializer: you can set it if you want
    :param scope
    :return:
    """
    with tf.variable_scope(scope or "linear", values=[x]):
        if not isinstance(x, (list, tuple)):
            x = [x]
        if not isinstance(dim, (list, tuple)):
            dim = [dim]

        if not ln:
            # by default, we concatenate inputs
            x = [tf.concat(x, -1)]

        outputs = []
        for oidx, osize in enumerate(dim):

            results = []
            for iidx, ix in enumerate(x):
                x_shp = util.shape_list(ix)
                xsize = x_shp[-1]

                W = tf.get_variable(
                    "W_{}_{}".format(oidx, iidx), [xsize, osize],
                    initializer=weight_initializer)
                o = tf.matmul(tf.reshape(ix, [-1, xsize]), W)

                if ln:
                    o = layer_norm(
                        o, scope="ln_{}_{}".format(oidx, iidx))
                results.append(o)

            o = tf.add_n(results)

            if bias:
                b = tf.get_variable(
                    "b_{}".format(oidx), [osize],
                    initializer=bias_initializer)
                o = tf.nn.bias_add(o, b)
            x_shp = util.shape_list(x[0])[:-1]
            o = tf.reshape(o, tf.concat([x_shp, [osize]], 0))

            outputs.append(o)

        return outputs[0] if len(outputs) == 1 else outputs


def split_heads(inputs, num_heads, name=None):
    """ Split heads
    :param inputs: A tensor with shape [batch, length, channels]
    :param num_heads: An integer
    :param name: An optional string
    :returns: A tensor with shape [batch, heads, length, channels / heads]
    """

    with tf.name_scope(name or "split_heads"):
        x = inputs
        n = num_heads
        old_shape = x.get_shape().dims

        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
        ret.set_shape(new_shape)
        return tf.transpose(ret, [0, 2, 1, 3])


def combine_heads(inputs, name=None):
    """ Combine heads
    :param inputs: A tensor with shape [batch, heads, length, channels]
    :param name: An optional string
    :returns: A tensor with shape [batch, length, heads * channels]
    """

    with tf.name_scope(name or "combine_heads"):
        x = inputs
        x = tf.transpose(x, [0, 2, 1, 3])
        old_shape = x.get_shape().dims
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        x = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
        x.set_shape(new_shape)

        return x


def additive_attention(query, memory, mem_mask, hidden_size,
                       ln=False, proj_memory=None, num_heads=1,
                       dropout=None, scope=None):
    """
    additive attention model
    :param query: [batch_size, dim]
    :param memory: [batch_size, seq_len, mem_dim]
    :param mem_mask: [batch_size, seq_len]
    :param hidden_size: attention space dimension
    :param ln: whether use layer normalization
    :param proj_memory: this is the mapped memory for saving memory
    :param num_heads: attention head number
    :param dropout: attention dropout, default disable
    :param scope:
    :return: a value matrix, [batch_size, mem_dim]
    """
    with tf.variable_scope(scope or "additive_attention"):
        if proj_memory is None:
            proj_memory = linear(
                memory, hidden_size, ln=ln, scope="feed_memory")

        query = linear(tf.expand_dims(query, 1),
                       hidden_size, ln=ln, scope="feed_query")

        query = split_heads(query, num_heads)
        proj_memory = split_heads(proj_memory, num_heads)

        value = tf.tanh(query + proj_memory)

        logits = linear(value, 1, ln=False, scope="feed_logits")
        logits = tf.squeeze(logits, -1)
        logits = util.mask_scale(logits, tf.expand_dims(mem_mask, 1))

        weights = tf.nn.softmax(logits, -1)  # [batch_size, seq_len]

        weights = util.valid_apply_dropout(weights, dropout)

        memory = split_heads(memory, num_heads)
        value = tf.reduce_sum(
            tf.expand_dims(weights, -1) * memory, -2, keepdims=True)

        value = combine_heads(value)
        value = tf.squeeze(value, 1)

        results = {
            'weights': weights,
            'output': value,
            'cache_state': proj_memory
        }

        return results


def dot_attention(query, memory, mem_mask, hidden_size,
                  ln=False, num_heads=1, cache=None, dropout=None,
                  use_relative_pos=True, max_relative_position=16,
                  out_map=True, scope=None):
    """
    dotted attention model
    :param query: [batch_size, qey_len, dim]
    :param memory: [batch_size, seq_len, mem_dim] or None
    :param mem_mask: [batch_size, seq_len]
    :param hidden_size: attention space dimension
    :param ln: whether use layer normalization
    :param num_heads: attention head number
    :param dropout: attention dropout, default disable
    :param out_map: output additional mapping
    :param cache: cache-based decoding
    :param max_relative_position: maximum position considered for relative embedding
    :param use_relative_pos: whether use relative position information
    :param scope:
    :return: a value matrix, [batch_size, qey_len, mem_dim]
    """
    with tf.variable_scope(scope or "dot_attention"):
        if memory is None:
            # suppose self-attention from queries alone
            h = linear(query, hidden_size * 3, ln=ln, scope="qkv_map")
            q, k, v = tf.split(h, 3, -1)

            if cache is not None:
                k = tf.concat([cache['k'], k], axis=1)
                v = tf.concat([cache['v'], v], axis=1)
                cache = {
                    'k': k,
                    'v': v,
                }
        else:
            q = linear(query, hidden_size, ln=ln, scope="q_map")
            if cache is not None and ('mk' in cache and 'mv' in cache):
                k, v = cache['mk'], cache['mv']
            else:
                h = linear(memory, hidden_size * 2, ln=ln, scope="kv_map")
                k, v = tf.split(h, 2, -1)

            if cache is not None:
                cache['mk'] = k
                cache['mv'] = v

        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)

        q *= (hidden_size // num_heads) ** (-0.5)

        q_shp = util.shape_list(q)
        k_shp = util.shape_list(k)
        v_shp = util.shape_list(v)

        # q * k => attention weights
        if use_relative_pos:
            r = get_relative_positions_embeddings(
                q_shp[2], k_shp[2], k_shp[3],
                max_relative_position, name="relative_positions_keys")
            logits = relative_attention_inner(q, k, r, transpose=True)
        else:
            logits = tf.matmul(q, k, transpose_b=True)

        if mem_mask is not None:
            logits += mem_mask

        weights = tf.nn.softmax(logits)

        weights = util.valid_apply_dropout(weights, dropout)

        # weights * v => attention vectors
        if use_relative_pos:
            r = get_relative_positions_embeddings(
                q_shp[2], k_shp[2], v_shp[3],
                max_relative_position, name="relative_positions_values")
            o = relative_attention_inner(weights, v, r, transpose=False)
        else:
            o = tf.matmul(weights, v)

        o = combine_heads(o)

        if out_map:
            o = linear(o, hidden_size, ln=ln, scope="o_map")

        results = {
            'weights': weights,
            'output': o,
            'cache': cache
        }

        return results


def layer_norm(x, eps=1e-8, scope=None):
    """RMS-based Layer normalization layer"""
    with tf.variable_scope(scope or "rms_norm"):
        layer_size = util.shape_list(x)[-1]

        scale = tf.get_variable("scale", [layer_size],
                                initializer=tf.ones_initializer())

        ms = tf.reduce_mean(x ** 2, -1, keep_dims=True)

        return scale * x * tf.rsqrt(ms + eps)


def residual_fn(x, y, dropout=None):
    """Residual Connection"""
    y = util.valid_apply_dropout(y, dropout)
    return x + y


def ffn_layer(x, d, d_o, dropout=None, scope=None):
    """FFN layer in Transformer"""
    with tf.variable_scope(scope or "ffn_layer"):
        hidden = linear(x, d, scope="enlarge")
        hidden = tf.nn.relu(hidden)

        hidden = util.valid_apply_dropout(hidden, dropout)

        output = linear(hidden, d_o, scope="output")

        return output


def add_timing_signal(x, min_timescale=1.0, max_timescale=1.0e4,
                      time=None, name=None):
    """Transformer Positional Embedding"""

    with tf.name_scope(name, default_name="add_timing_signal", values=[x]):
        length = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        if time is None:
            position = tf.to_float(tf.range(length))
        else:
            # decoding position embedding
            position = tf.expand_dims(time, 0)
        num_timescales = channels // 2

        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                (tf.to_float(num_timescales) - 1)
        )
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment
        )

        scaled_time = (tf.expand_dims(position, 1) *
                       tf.expand_dims(inv_timescales, 0))
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])

        return x + signal


def attention_bias(inputs, mode, inf=-1e9, name=None):
    """ A bias tensor used in attention mechanism"""

    with tf.name_scope(name, default_name="attention_bias", values=[inputs]):
        if mode == "causal":
            length = inputs
            lower_triangle = tf.matrix_band_part(
                tf.ones([length, length]), -1, 0
            )
            ret = inf * (1.0 - lower_triangle)
            return tf.reshape(ret, [1, 1, length, length])
        elif mode == "masking":
            mask = inputs
            ret = (1.0 - mask) * inf
            return tf.expand_dims(tf.expand_dims(ret, 1), 1)
        elif mode == "aan":
            length = tf.shape(inputs)[1]
            diagonal = tf.eye(length)
            cum_factor = tf.expand_dims(tf.cumsum(diagonal, axis=0), 0)
            mask = tf.expand_dims(inputs, 1) * tf.expand_dims(inputs, 2)
            mask *= cum_factor
            weight = tf.nn.softmax(mask + (1.0 - mask) * inf)
            weight *= mask
            return weight
        else:
            raise ValueError("Unknown mode %s" % mode)


def cnn(inputs, hidden_size, mask=None, scope="cnn"):
    with tf.variable_scope(scope or "cnn"):
        ishp = util.shape_list(inputs)
        if mask is None:
            mask = tf.ones([ishp[0], ishp[1]])

        x = inputs
        x = x * tf.expand_dims(mask, -1)

        x0 = tf.pad(x, [[0,0], [1, 0], [0,0]])[:, :-1, :]
        x1 = tf.pad(x, [[0,0], [0, 1], [0,0]])[:, 1:, :]

        y = tf.concat([x0, x, x1], -1)
        y = linear(y, hidden_size * 2, ln=False, scope="ff")

        A = y[:, :, :hidden_size]
        B = y[:, :, hidden_size:]
        y = A * tf.sigmoid(B)

        y += x

        return layer_norm(y, scope="ln")


def depthwise_conv(inputs, hidden_size, kernel_size=1, bias=True,
                   activation=None, scope='depthwise_conv'):
    with tf.variable_scope(scope or "depthwise_conv"):
        shapes = util.shape_list(inputs)
        depthwise_filter = tf.get_variable('depthwise_filter', (kernel_size, 1, shapes[-1], 1))
        pointwise_filter = tf.get_variable('pointwise_filter', (1, 1, shapes[-1], hidden_size))

        outputs = tf.nn.separable_conv2d(inputs, depthwise_filter, pointwise_filter,
                                         strides=(1, 1, 1, 1), padding='SAME')
        if bias:
            b = tf.get_variable('bias', outputs.shape[-1], initializer=tf.zeros_initializer())
            outputs += b

        if activation is not None:
            return activation(outputs)
        else:
            return outputs


def highway(x, size=None, activation=None, num_layers=2, dropout=0.0, ln=False, scope='highway'):
    with tf.variable_scope(scope or "highway"):
        if size is None:
            size = x.shape.as_list()[-1]
        else:
            x = linear(x, size, ln=ln, scope="input_projection")

        for i in range(num_layers):
            T = linear(x, size, ln=ln, scope='gate_%d' % i)
            T = tf.nn.sigmoid(T)

            H = linear(x, size, ln=ln, scope='activation_%d' % i)
            if activation is not None:
                H = activation(H)

            H = util.valid_apply_dropout(H, dropout)
            x = H * T + x * (1.0 - T)

        return x


def trilinear_similarity(x1, x2, scope='trilinear'):
    with tf.variable_scope(scope or "trilinear"):
        x1_shape = util.shape_list(x1)
        x2_shape = util.shape_list(x2)

        if len(x1_shape) != 3 or len(x2_shape) != 3:
            raise ValueError('`args` must be 3 dims (batch_size, len, dimension)')
        if x1_shape[2] != x2_shape[2]:
            raise ValueError('the last dimension of `args` must equal')

        w1 = tf.get_variable('kernel_x1', [x1_shape[2], 1])
        w2 = tf.get_variable('kernel_x2', [x2_shape[2], 1])
        w3 = tf.get_variable('kernel_mul', [1, 1, x1_shape[2]])
        bias = tf.get_variable('bias', [1], initializer=tf.zeros_initializer())

        r1 = tf.einsum('aij,jk->aik', x1, w1)
        r2 = tf.einsum('aij,jk->aki', x2, w2)
        r3 = tf.einsum('aij,akj->aik', x1 * w3, x2)
        return r1 + r2 + r3 + bias


def relative_attention_inner(x, y, z=None, transpose=False):
    """Relative position-aware dot-product attention inner calculation.

    This batches matrix multiply calculations to avoid unnecessary broadcasting.

    Args:
    x: Tensor with shape [batch_size, heads, length, length or depth].
    y: Tensor with shape [batch_size, heads, length, depth].
    z: Tensor with shape [length, length, depth].
    transpose: Whether to transpose inner matrices of y and z. Should be true if
        last dimension of x is depth, not length.

    Returns:
    A Tensor with shape [batch_size, heads, length, length or depth].
    """
    batch_size = tf.shape(x)[0]
    heads = x.get_shape().as_list()[1]
    length = tf.shape(x)[2]

    # xy_matmul is [batch_size, heads, length, length or depth]
    xy_matmul = tf.matmul(x, y, transpose_b=transpose)
    if z is not None:
        # x_t is [length, batch_size, heads, length or depth]
        x_t = tf.transpose(x, [2, 0, 1, 3])
        # x_t_r is [length, batch_size * heads, length or depth]
        x_t_r = tf.reshape(x_t, [length, heads * batch_size, -1])
        # x_tz_matmul is [length, batch_size * heads, length or depth]
        x_tz_matmul = tf.matmul(x_t_r, z, transpose_b=transpose)
        # x_tz_matmul_r is [length, batch_size, heads, length or depth]
        x_tz_matmul_r = tf.reshape(x_tz_matmul, [length, batch_size, heads, -1])
        # x_tz_matmul_r_t is [batch_size, heads, length, length or depth]
        x_tz_matmul_r_t = tf.transpose(x_tz_matmul_r, [1, 2, 0, 3])
        return xy_matmul + x_tz_matmul_r_t
    else:
        return xy_matmul


def get_relative_positions_embeddings(length_x, length_y,
                                      depth, max_relative_position, name):
    """Generates tensor of size [length_x, length_y, depth]."""
    with tf.variable_scope(name):
        relative_positions_matrix = get_relative_positions_matrix(
            length_x, length_y, max_relative_position)
        vocab_size = max_relative_position * 2 + 1
        # Generates embedding for each relative position of dimension depth.
        embeddings_table = tf.get_variable("embeddings", [vocab_size, depth])
        embeddings = tf.gather(embeddings_table, relative_positions_matrix)
        return embeddings


def get_relative_positions_matrix(length_x, length_y, max_relative_position):
    """Generates matrix of relative positions between inputs."""
    range_vec_x = tf.range(length_x)
    range_vec_y = tf.range(length_y)

    # shape: [length_x, length_y]
    distance_mat = tf.expand_dims(range_vec_x, -1) - tf.expand_dims(range_vec_y, 0)
    distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position,
                                            max_relative_position)

    # Shift values to be >= 0. Each integer still uniquely identifies a relative
    # position difference.
    final_mat = distance_mat_clipped + max_relative_position
    return final_mat


def dnn(input, hidden_size, relu_size, mask=None,
        dropout=0.1,
        use_ln=False, bidir=True, scope=None):
    ishp = util.shape_list(input)

    if mask is None:
        mask = tf.ones([ishp[0], ishp[1]], dtype=tf.float32)

    with tf.variable_scope(scope or "dnn"):
        # perform self attention
        state_att = tf.nn.tanh(linear(input, hidden_size, ln=use_ln, scope="att"))
        state_ffn = tf.nn.tanh(linear(input, hidden_size, ln=use_ln, scope="ff"))

        att_v = tf.get_variable('v', [hidden_size])
        alpha = tf.reduce_sum(state_att * att_v, -1) * hidden_size ** (-0.5)
        alpha = alpha + (1. - mask) * (-1e9)
        alpha = tf.nn.softmax(alpha, -1, name="attention_weights")
        alpha = util.valid_apply_dropout(alpha, dropout)

        state_ffn = state_ffn * tf.expand_dims(alpha, -1)

        # shape: [#Batch, #Seq, #Dim]

        # left -> right
        cum_state_below = tf.cumsum(state_ffn, axis=1)
        cum_length = tf.cumsum(alpha, axis=1)
        cum_length = tf.where(tf.less_equal(cum_length, 0.), tf.ones_like(cum_length), cum_length)
        cum_state_below = cum_state_below / tf.expand_dims(cum_length, -1)

        if bidir:
            # right -> left
            cum_state_belowr = tf.cumsum(state_ffn, axis=1, reverse=True)
            cum_lengthr = tf.cumsum(alpha, axis=1, reverse=True)
            cum_lengthr = tf.where(tf.less_equal(cum_lengthr, 0.), tf.ones_like(cum_lengthr), cum_lengthr)
            cum_state_belowr = cum_state_belowr / tf.expand_dims(cum_lengthr, -1)
        else:
            cum_state_belowr = cum_state_below

        feature = (cum_state_below + cum_state_belowr) * 0.5
        feature = tf.nn.tanh(linear(feature, hidden_size, ln=use_ln, scope='feat'))
        feature = util.valid_apply_dropout(feature, dropout)
        feature = feature + input
        feature = layer_norm(feature, scope="ln2")

        # Feed Forward Layer
        ff_feature = tf.nn.relu(linear(feature, relu_size, ln=use_ln, scope="_C"))
        activation = tf.nn.tanh(linear(ff_feature, hidden_size, ln=use_ln, scope="_F"))
        activation = util.valid_apply_dropout(activation, dropout)

        activation = activation + feature
        activation = layer_norm(activation, scope="ln")

        return activation


def cond_dnn(input, memory, hidden_size, relu_size, mask=None, mem_mask=None,
             dropout=0.1,
             use_ln=False, scope=None):
    ishp = util.shape_list(input)
    mshp = util.shape_list(memory)

    if mask is None:
        mask = tf.ones([ishp[0], ishp[1]], tf.float32)
    if mem_mask is None:
        mem_mask = tf.ones([mshp[0], mshp[1]], tf.float32)

    with tf.variable_scope(scope or "cond_dnn"):
        # perform source=>target attention
        # shape: [#Batch, #Seq, #Dim]
        src_context = tf.nn.tanh(linear(memory, hidden_size, ln=use_ln, scope="src_att"))
        vle_context = tf.nn.tanh(linear(memory, hidden_size, ln=use_ln, scope="src_vle"))
        tgt_feature = tf.nn.tanh(linear(input, hidden_size, ln=use_ln, scope="tgt_att"))

        # shape: [#Batch, #Tgt, #Src]
        ctx_alpha = tf.matmul(tgt_feature, src_context, transpose_b=True) * hidden_size ** (-0.5)

        ctx_alpha = ctx_alpha + tf.expand_dims((1. - mem_mask) * (-1e9), 1)
        ctx_alpha = tf.nn.softmax(ctx_alpha)
        ctx_alpha = util.valid_apply_dropout(ctx_alpha, dropout)

        attention = tf.matmul(ctx_alpha, vle_context)

        attention = tf.nn.tanh(linear(attention, hidden_size, ln=use_ln, scope="feat"))
        attention = util.valid_apply_dropout(attention, dropout)
        attention = attention + input
        attention = layer_norm(attention, scope="ln2")

        # Feed Forward Layer
        ff_state = tf.nn.relu(linear(attention, relu_size, ln=use_ln, scope="X"))
        activation = tf.nn.tanh(linear(ff_state, hidden_size, ln=use_ln, scope="F"))
        activation = util.valid_apply_dropout(activation, dropout)
        activation = activation + attention  # assume that #src_dims == #tgt_dims
        activation = layer_norm(activation, scope="lns")

        return activation, ctx_alpha
