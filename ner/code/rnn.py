from __future__ import absolute_import
from __future__ import division

"""Implement different RNN layers so as to compare their performance"""

import keras
import keras.backend as K
from keras.layers import RNN


def _slice(tensor, index, size):
    tshp = K.int_shape(tensor)
    tdim = len(tshp)

    if tdim == 3:
        return tensor[:, :, index * size: (index+1) * size]
    else:
        return tensor[:, index * size: (index+1) * size]


# GRU
class GRUCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        self.output_size = units
        super(GRUCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self._W = self.add_weight(
            shape=(input_shape[-1]+self.units, self.units * 2),
            initializer="glorot_uniform",
            name="kernel"
        )
        self._W_b = self.add_weight(
            shape=(self.units * 2, ),
            initializer="zeros",
            name="kernel_bias"
        )

        self._U = self.add_weight(
            shape=(input_shape[-1]+self.units, self.units),
            initializer="glorot_uniform",
            name="candidate"
        )
        self._U_b = self.add_weight(
            shape=(self.units, ),
            initializer="zeros",
            name="candidate_bias"
        )

        self.built = True

    def call(self, inputs, states):
        x = inputs
        h_ = states[0]

        gate_inputs = K.dot(K.concatenate([x, h_], -1), self._W)
        gate_inputs = K.bias_add(gate_inputs, self._W_b)
        gate_inputs = K.sigmoid(gate_inputs)

        r = _slice(gate_inputs, 0, self.units)
        u = _slice(gate_inputs, 1, self.units)

        rh_ = r * h_

        candidate = K.dot(K.concatenate([x, rh_], -1), self._U)
        candidate = K.bias_add(candidate, self._U_b)
        candidate = K.tanh(candidate)

        h = u * h_ + (1. - u) * candidate

        return h, [h]

    def get_config(self):
        config = {'units': self.units,}
        base_config = super(GRUCell, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


# LSTM
class LSTMCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.state_size = (units, units)
        self.output_size = units
        self.units = units
        super(LSTMCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self._W = self.add_weight(
            shape=(input_shape[-1]+self.units, self.units * 4),
            initializer="glorot_uniform",
            name="kernel"
        )
        self._W_b = self.add_weight(
            shape=(self.units * 4, ),
            initializer="zeros",
            name="kernel_bias"
        )

        self.built = True

    def call(self, inputs, states):
        x = inputs
        h_, c_ = states[0], states[1]

        gate_inputs = K.dot(K.concatenate([x, h_], -1), self._W)
        gate_inputs = K.bias_add(gate_inputs, self._W_b)

        i = _slice(gate_inputs, 0, self.units)
        f = _slice(gate_inputs, 1, self.units)
        o = _slice(gate_inputs, 2, self.units)
        g = _slice(gate_inputs, 3, self.units)

        c = c_ * K.sigmoid(f) + K.tanh(g) * K.sigmoid(i)
        h = K.sigmoid(o) * K.tanh(c)

        return h, [h, c]

    def get_config(self):
        config = {'units': self.units, }
        base_config = super(LSTMCell, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


# ATR
class ATRCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        self.output_size = units
        super(ATRCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self._W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            name="kernel"
        )
        self._W_b = self.add_weight(
            shape=(self.units, ),
            initializer="zeros",
            name="kernel_bias"
        )

        self._U = self.add_weight(
            shape=(self.units, self.units),
            initializer="glorot_uniform",
            name="candidate"
        )
        self._U_b = self.add_weight(
            shape=(self.units, ),
            initializer="zeros",
            name="candidate_bias"
        )

        self.built = True

    def call(self, inputs, states):
        x = inputs
        h_ = states[0]

        p = K.bias_add(K.dot(x, self._W), self._W_b)
        q = K.bias_add(K.dot(h_, self._U), self._U_b)

        i = K.sigmoid(p + q)
        f = K.sigmoid(p - q)

        h = i * p + f * h_

        return h, [h]

    def get_config(self):
        config = {'units': self.units, }
        base_config = super(ATRCell, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


# LRN
class LRNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        self.output_size = units
        super(LRNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self._W = self.add_weight(
            shape=(input_shape[-1], self.units * 3),
            initializer="glorot_uniform",
            name="kernel"
        )
        self._W_b = self.add_weight(
            shape=(self.units * 3, ),
            initializer="zeros",
            name="kernel_bias"
        )

        self.built = True

    def call(self, inputs, states):
        x = inputs
        h_ = states[0]

        candidate = K.bias_add(K.dot(x, self._W), self._W_b)

        p = _slice(candidate, 0, self.units)
        q = _slice(candidate, 1, self.units)
        r = _slice(candidate, 2, self.units)

        i = K.sigmoid(p + h_)
        f = K.sigmoid(q - h_)

        h = K.tanh(i * r + f * h_)

        return h, [h]

    def get_config(self):
        config = {'units': self.units, }
        base_config = super(LRNCell, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


# SRU
class SRUCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.state_size = (units, units)
        self.output_size = units
        self.units = units
        super(SRUCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self._W = self.add_weight(
            shape=(input_shape[-1], self.units * 4),
            initializer="glorot_uniform",
            name="kernel"
        )
        self._W_b = self.add_weight(
            shape=(self.units * 4, ),
            initializer="zeros",
            name="kernel_bias"
        )

        self._Vr = self.add_weight(
            shape=(self.units, ),
            initializer="glorot_uniform",
            name="kernel_vr"
        )

        self._Vf = self.add_weight(
            shape=(self.units, ),
            initializer="glorot_uniform",
            name="kernel_vf"
        )

        self.built = True

    def call(self, inputs, states):
        x = inputs
        h_, c_ = states[0], states[1]

        g = K.bias_add(K.dot(x, self._W), self._W_b)

        g1 = _slice(g, 0, self.units)
        g2 = _slice(g, 1, self.units)
        g3 = _slice(g, 2, self.units)
        g4 = _slice(g, 3, self.units)

        f = K.sigmoid(g1 + self._Vf * c_)
        c = f * c_ + (1. - f) * g2
        r = K.sigmoid(g3 + self._Vr * c_)
        h = r * c + (1. - r) * g4

        return h, [h, c]

    def get_config(self):
        config = {'units': self.units, }
        base_config = super(SRUCell, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


def get_rnn_layer(cell_type, units, return_sequences=True):
    cell_type = cell_type.lower()

    print("RNN Type: **{}**".format(cell_type))

    if cell_type == "gru":
        cell = GRUCell(units)
    elif cell_type == "lstm":
        cell = LSTMCell(units)
    elif cell_type == "atr":
        cell = ATRCell(units)
    elif cell_type == "lrn":
        cell = LRNCell(units)
    elif cell_type == "sru":
        cell = SRUCell(units)
    else:
        raise NotImplementedError(
            "{} is not supported".format(cell_type))

    keras.utils.generic_utils.get_custom_objects()[cell.__class__.__name__] = cell.__class__
    return RNN(cell, return_sequences=return_sequences)
