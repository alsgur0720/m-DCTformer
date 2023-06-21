import torch
import torch.nn as nn
from model import common, dct
import numpy as np
import torch.nn.functional as nnf
import math
import sys
from .swinIr import SwinIR
from .edsr import EDSR
from .rdn import RDN

def make_model(opt):
    return (opt, 2, 3, 64, 64, 16, 8)
