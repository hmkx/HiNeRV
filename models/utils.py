"""
Model utils
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from typing import Union, List, Optional
from functools import partial


def str_to_bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError


def cfg_override(cfg, **kwargs):
    cfg = cfg.copy()
    for k, v in kwargs.items():
        cfg[k] = v
    return cfg


def mod(a, b):
    return a - a.div(b, rounding_mode='floor') * b


def divide_to_multiple(a, b, factor):
    return int(math.ceil((a // b) / factor) * factor)


def assert_shape(x: torch.Tensor, shape: tuple):
    assert tuple(x.shape) == tuple(shape), f'shape: {x.shape}     expected: {shape}'