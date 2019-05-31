# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# This is an abstract class that deals with
#   different learning rate decay strategy
# Generally, we decay the learning rate with GPU computation
# However, in this paper, we simply decay the learning rate
#   at CPU level, and feed the decayed lr into GPU for
#   optimization
class Lr(object):
    def __init__(self,
                 init_lrate,        # initial learning rate
                 name="lr",         # learning rate name, no use
                 ):
        self.name = name
        self.init_lrate = init_lrate    # just record the init learning rate
        self.lrate = init_lrate         # active learning rate, change with training

    # suppose the eidx starts from 1
    def before_epoch(self, eidx=None):
        pass

    def after_epoch(self, eidx=None):
        pass

    def step(self, step):
        pass

    def after_eval(self, eval_score):
        pass

    def get_lr(self):
        """Return the learning rate whenever you want"""
        return self.lrate
