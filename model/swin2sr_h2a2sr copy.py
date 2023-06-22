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


class H2A2SR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(H2A2SR, self).__init__()
        self.scale = args.scale[0]
        self.int_scale = math.floor(self.scale)
        self.float_scale = args.float_scale
        self.total_scale = args.total_scale
        self.res_scale = self.total_scale / self.int_scale
        kernel_size = 3
        act = nn.ReLU(True)

        self.dct = dct.DCT_2D()
        self.idct = dct.IDCT_2D()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size, padding=1)
        self.R1 = common.RCAB(conv, 64, kernel_size, act=act)
        self.R2 = common.RCAB(conv, 64, kernel_size, act=act)
        self.R3 = common.RCAB(conv, 64, kernel_size, act=act)
        self.R4 = common.RCAB(conv, 64, kernel_size, act=act)
        self.R5 = common.RCAB(conv, 64, kernel_size, act=act)
        self.t = nn.Conv2d(64, 3, kernel_size, padding=1)

        self.x_cof = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)

    def forward(self, x, outH, outW):
        _,_, h, w = x.size()
        
        x = self.dct(x)
        x_org = h * w
        zeroPad2d = nn.ZeroPad2d((0, outW-w, 0, outH-h)).to('cuda:0')

        x = zeroPad2d(x)
        _,_, x_expandh, x_expandw = x.size()
        x_expand = x_expandh * x_expandw
        expand = x_expand / x_org
        x = x * expand

        mask = torch.ones((h, w), dtype=torch.int64, device = torch.device('cuda:0'))
        diagonal = w-2

        lf_mask = torch.fliplr(torch.triu(mask, diagonal)) == 1
        hf_mask = torch.fliplr(torch.triu(mask, diagonal)) != 1
        
        lf_mask = zeroPad2d(lf_mask)
        hf_mask = zeroPad2d(hf_mask)
        
        lf_mask = lf_mask.unsqueeze(0).expand(x.size())
        hf_mask = hf_mask.unsqueeze(0).expand(x.size())

        dhf = x * hf_mask

        x = self.idct(x)
        hf = self.idct(dhf)
        
        coefficent = self.x_cof
        # print(coefficent)
        # sys.exit()
        
        
        
        hf = hf * coefficent
        hf = self.conv1(hf)
        hf = self.R1(hf)
        hf = self.R2(hf)
        hf = self.R3(hf)
        hf = self.R4(hf)
        hf = self.R5(hf)
        hf = self.t(hf)

        result = x + hf
        return result
