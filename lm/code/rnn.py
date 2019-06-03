import torch
from torch import nn
from torch.autograd import Variable


# LSTM
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):

        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self._W = nn.Parameter(torch.FloatTensor(input_size + hidden_size, 4 * hidden_size))
        self._b = nn.Parameter(torch.FloatTensor(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._W.data)
        nn.init.constant_(self._b.data, 0)

    def forward(self, x, s_):
        h_, c_ = s_

        candidate = torch.mm(torch.cat([x, h_], -1), self._W) + self._b
        i, f, o, g = candidate.split(self.hidden_size, -1)

        c = i.sigmoid() * g.tanh() + f.sigmoid() * c_
        h = o.sigmoid() * c.tanh()

        return h, c


# GRU
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):

        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self._W = nn.Parameter(torch.FloatTensor(input_size + hidden_size, 2 * hidden_size))
        self._W_b = nn.Parameter(torch.FloatTensor(2 * hidden_size))
        self._U = nn.Parameter(torch.FloatTensor(input_size + hidden_size, hidden_size))
        self._U_b = nn.Parameter(torch.FloatTensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._W.data)
        nn.init.xavier_uniform_(self._U.data)
        nn.init.constant_(self._W_b.data, 0)
        nn.init.constant_(self._U_b.data, 0)

    def forward(self, x, h_):

        g = torch.mm(torch.cat([x, h_], -1), self._W) + self._W_b

        r, u = g.sigmoid().split(self.hidden_size, -1)

        c = torch.mm(torch.cat([x, r * h_], -1), self._U) + self._U_b

        h = u * h_ + (1. - u) * c.tanh()

        return h


# ATR
class ATRCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ATRCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self._W = nn.Parameter(torch.FloatTensor(input_size, hidden_size))
        self._W_b = nn.Parameter(torch.FloatTensor(hidden_size))
        self._U = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        self._U_b = nn.Parameter(torch.FloatTensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._W.data)
        nn.init.xavier_uniform_(self._U.data)
        nn.init.constant_(self._W_b.data, 0)
        nn.init.constant_(self._U_b.data, 0)

    def forward(self, x, h_):

        p = torch.mm(x, self._W) + self._W_b
        q = torch.mm(h_, self._U) + self._U_b

        i = (p + q).sigmoid()
        f = (p - q).sigmoid()

        h = (i * p + f * h_).tanh()

        return h


# LRN
class LRNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LRNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self._W = nn.Parameter(torch.FloatTensor(input_size, hidden_size * 3))
        self._W_b = nn.Parameter(torch.FloatTensor(hidden_size * 3))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._W.data)
        nn.init.constant_(self._W_b.data, 0)

    def forward(self, x, h_):

        p, q, r = (torch.mm(x, self._W) + self._W_b).split(self.hidden_size, -1)

        i = (p + h_).sigmoid()
        f = (q - h_).sigmoid()

        h = (i * r + f * h_).tanh()

        return h


# SRU
class SRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):

        super(SRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self._W = nn.Parameter(torch.FloatTensor(input_size, 4 * hidden_size))
        self._W_b = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        self._Vr = nn.Parameter(torch.FloatTensor(hidden_size))
        self._Vf = nn.Parameter(torch.FloatTensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._W.data)
        nn.init.constant_(self._Vr.data, 1)
        nn.init.constant_(self._Vf.data, 1)
        nn.init.constant_(self._W_b.data, 0)

    def forward(self, x, s_):
        h_, c_ = s_

        g = torch.mm(x, self._W) + self._W_b

        g1, g2, g3, g4 = g.split(self.hidden_size, -1)

        f = (g1 + self._Vf * c_).sigmoid()
        c = f * c_ + (1. - f) * g2
        r = (g3 + self._Vr * c_).sigmoid()
        h = r * c + (1. - r) * g4

        return h, c


def get_cell(cell_type):
    cell_type = cell_type.lower()

    print("RNN Type: **{}**".format(cell_type))

    if cell_type == "gru":
        cell = GRUCell
    elif cell_type == "lstm":
        cell = LSTMCell
    elif cell_type == "atr":
        cell = ATRCell
    elif cell_type == "lrn":
        cell = LRNCell
    elif cell_type == "sru":
        cell = SRUCell
    else:
        raise NotImplementedError(
            "{} is not supported".format(cell_type))

    return cell


class RNN(nn.Module):

    def __init__(self, cell_type, input_size, hidden_size,
                 num_layers=1, batch_first=False, dropout=0, **kwargs):
        super(RNN, self).__init__()
        self.cell_type = cell_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.c_on = self.cell_type == "lstm" or self.cell_type == "sru"

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = get_cell(cell_type)(
                input_size=layer_input_size, hidden_size=hidden_size, **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    def _forward_rnn(self, cell, x, h, L):
        max_time = x.size(0)
        output = []
        for time in range(max_time):
            if self.c_on:
                new_h, new_c = cell(x[time], h)
            else:
                new_h = cell(x[time], h)

            mask = (time < L).float().unsqueeze(1).expand_as(new_h)
            new_h = new_h*mask + h[0]*(1 - mask)

            if self.c_on:
                new_c = new_c*mask + h[1]*(1 - mask)
                h = (new_h, new_c)
            else:
                h = new_h

            output.append(new_h)

        output = torch.stack(output, 0)
        return output, h

    def forward(self, x, h=None, L=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        max_time, batch_size, _ = x.size()
        if L is None:
            L = Variable(torch.LongTensor([max_time] * batch_size))
            if x.is_cuda:
                L = L.cuda(x.get_device())
        if h is None:
            if self.c_on:
                h = (Variable(nn.init.xavier_uniform(x.new(self.num_layers, batch_size, self.hidden_size))),
                     Variable(nn.init.xavier_uniform(x.new(self.num_layers, batch_size, self.hidden_size))))
            else:
                h = Variable(nn.init.xavier_uniform(x.new(self.num_layers, batch_size, self.hidden_size)))

        layer_output = None
        states = []
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            if self.c_on:
                h_layer = (h[0][layer, :, :], h[1][layer, :, :])
            else:
                h_layer = h[layer, :, :]
            
            if layer == 0:
                layer_output, layer_state = self._forward_rnn(
                    cell, x, h_layer, L)
            else:
                layer_output, layer_state = self._forward_rnn(
                    cell, layer_output, h_layer, L)

            if layer != self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)
            states.append(layer_state)

        output = layer_output

        if self.c_on:
            states = list(zip(*states))
            return output, (torch.stack(states[0], 0), torch.stack(states[1], 0))
        else:
            return output, torch.stack(states, 0)
