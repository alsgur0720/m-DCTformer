import torch
import torch.nn as nn
from model import common, dct
import numpy as np
import torch.nn.functional as nnf
import math
import sys
from .swin2sr import Swin2SR

def make_model(opt):
    return Swin2SR(opt)
