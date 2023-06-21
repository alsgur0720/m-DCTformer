import torch
import torch.nn as nn
from model import common, dct
import numpy as np
import torch.nn.functional as nnf
import math
import sys
from .swinIr import SwinIR

def make_model(opt):
    return SwinIR(opt)
