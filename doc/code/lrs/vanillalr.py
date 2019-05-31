# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from lrs import lr


class VanillaLR(lr.Lr):
    """Very basic learning rate, constant learning rate"""
    def __init__(self,
                 init_lr,
                 name="vanilla_lr"
                 ):
        super(VanillaLR, self).__init__(init_lr, name=name)
