from model import common

import torch.nn as nn


import numbers
import torch.nn.functional as nnf

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange

from model import common, dct

import sys



def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = 32
        n_feats = 256
        kernel_size = 3 
        scale = 3
        act = nn.ReLU(True)
        self.sub_mean = common.MeanShift(opt.rgb_range)
        self.add_mean = common.MeanShift(opt.rgb_range, sign=1)

        # define head module
        m_head = [conv(opt.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=0.1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, opt.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.h2a2sr = H2A2SR(opt)
        
        
    def forward(self, x, outH, outW):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        
        # return x
        return self.h2a2sr(x, outH, outW) 
    
    
    


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # self.weight_x_1 = nn.Conv2d(dim*3, dim*3, kernel_size=(3,1), stride=1, padding=(1,0), groups=dim*3, bias=bias)
        # self.weight_x_2 = nn.Conv2d(dim*3, dim*3, kernel_size=(3,1), stride=1, padding=(1,0), groups=dim*3, bias=bias)
        # self.weight_y_1 = nn.Conv2d(dim*3, dim*3, kernel_size=(1,3), stride=1, padding=(0,1), groups=dim*3, bias=bias)
        # self.weight_y_2 = nn.Conv2d(dim*3, dim*3, kernel_size=(1,3), stride=1, padding=(0,1), groups=dim*3, bias=bias)
        
        self.weight_modulation_x_q = nn.Conv2d(dim, dim, kernel_size=(3,1), stride=1, padding=(1,0), groups=dim, bias=bias)
        self.weight_modulation_y_q = nn.Conv2d(dim, dim, kernel_size=(1,3), stride=1, padding=(0,1), groups=dim, bias=bias)
        
        self.weight_modulation_x_k = nn.Conv2d(dim, dim, kernel_size=(3,1), stride=1, padding=(1,0), groups=dim, bias=bias)
        self.weight_modulation_y_k = nn.Conv2d(dim, dim, kernel_size=(1,3), stride=1, padding=(0,1), groups=dim, bias=bias)
        
        self.weight_modulation_x_v = nn.Conv2d(dim, dim, kernel_size=(3,1), stride=1, padding=(1,0), groups=dim, bias=bias)
        self.weight_modulation_y_v = nn.Conv2d(dim, dim, kernel_size=(1,3), stride=1, padding=(0,1), groups=dim, bias=bias)



        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape
        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        # coef = 1
        # w_x_1 = self.weight_x_1(x)*coef
        # w_x_2 = self.weight_x_2(w_x_1)*coef
        # w_y_1 = self.weight_y_1(x)*coef
        # w_y_2 = self.weight_y_2(w_y_1)*coef
        # qkv = qkv + w_x_1 + w_y_1
        q,k,v = qkv.chunk(3, dim=1)   
        
        coef = 0.125
        w_x_q = self.weight_modulation_x_q(q)*coef
        w_y_q = self.weight_modulation_y_q(q)*coef
        
        w_x_k = self.weight_modulation_x_k(k)*coef
        w_y_k = self.weight_modulation_y_k(k)*coef
        
        w_x_v = self.weight_modulation_x_v(v)*coef
        w_y_v = self.weight_modulation_y_v(v)*coef
        
        q = q + w_x_q + w_y_q
        k = k + w_x_k + w_y_k
        v = v + w_x_v + w_y_v
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        # self.bn = nn.BatchNorm2d(embed_dim)
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        
        # B,C,h,w = x.size()
        # for c in C:
        #     feature = []
        #     for i in h:
        #         for j in w:
        #             feature.append(x[])
                
        # print(x.size())
        # sys.exit()
        
        x = self.proj(x)
        # x = self.bn(x)
        return x



##########################################################################
## Resizing modules
class Downsample_rest(nn.Module):
    def __init__(self, n_feat):
        super(Downsample_rest, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample_rest(nn.Module):
    def __init__(self, n_feat):
        super(Upsample_rest, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        
    ):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample_rest(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample_rest(int(dim*2**1)) ## From Level 2 to Level 3
        # self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        # self.down3_4 = Downsample_rest(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        # self.up4_3 = Upsample_rest(int(dim*2**3)) ## From Level 4 to Level 3
        # self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        # self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample_rest(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample_rest(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        # self.dct = dct.DCT_2D()
        # self.idct = dct.IDCT_2D()
        self.x_cof = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        
        # self.upsample = SA_upsample(48)
        
        
        
    def forward(self, inp_img):

        _,_, h, w = inp_img.size()
        
        # inp_img = self.dct(inp_img)
        
        # mask = torch.ones((h, w), dtype=torch.int64, device = torch.device('cuda:0'))
        # diagonal = w-2
        
        # hf_mask = torch.fliplr(torch.triu(mask, diagonal)) != 1
        # lf_mask = torch.fliplr(torch.triu(mask, diagonal)) == 1
        # hf_mask = hf_mask.unsqueeze(0).expand(inp_img.size())
        # lf_mask = lf_mask.unsqueeze(0).expand(inp_img.size())
        
        # dhf = inp_img * hf_mask
        # dlf = inp_img * lf_mask
        
        # coefficent = self.x_cof
        # dhf = dhf * coefficent
        
        inp_enc_level1 = self.patch_embed(inp_img)
        # inp_enc_level1 = self.upsample(inp_enc_level1, 2.0, 2.0)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        latent = self.latent(inp_enc_level3)
         
        # out_enc_level3 = self.encoder_level3(inp_enc_level3) 
        # inp_enc_level4 = self.down3_4(out_enc_level3)        
                        
        # inp_dec_level3 = self.up4_3(latent)
        # inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        # inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        # out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)
        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_img)
            out_dec_level1 = self.output(out_dec_level1) 
        ###########################
        else:
            # out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            # out_dec_level1 = self.output(out_dec_level1)
            out_dec_level1 = self.output(out_dec_level1) + inp_img
            # result = dlf + out_dec_level1


        # return result
        return out_dec_level1



def grid_sample(x, offset, scale, scale2):
    # generate grids
    b, _, h, w = x.size()
    grid = np.meshgrid(range(round(scale2*w)), range(round(scale*h)))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid = torch.Tensor(grid).to(x.device)
    # project into LR space
    grid[:, :, 0] = (grid[:, :, 0] + 0.5) / scale2 - 0.5
    grid[:, :, 1] = (grid[:, :, 1] + 0.5) / scale - 0.5
    # normalize to [-1, 1]
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) -1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) -1
    
    
    grid = grid.permute(2, 0, 1).unsqueeze(0)
    grid = grid.expand([b, -1, -1, -1])

    # add offsets
    offset_0 = torch.unsqueeze(offset[:, 0, :, :] * 2 / (w - 1), dim=1)
    offset_1 = torch.unsqueeze(offset[:, 1, :, :] * 2 / (h - 1), dim=1)
    
    grid = grid + torch.cat((offset_0, offset_1),1)
    grid = grid.permute(0, 2, 3, 1)
    # sampling
    output = F.grid_sample(x, grid, padding_mode='zeros', align_corners = True)
    return output





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
        
        # self.conv1 = nn.Conv2d(3, 64, kernel_size, padding=1)
        # self.R1 = common.RCAB(conv, 64, kernel_size, act=act)
        # self.R2 = common.RCAB(conv, 64, kernel_size, act=act)
        # self.R3 = common.RCAB(conv, 64, kernel_size, act=act)
        # self.R4 = common.RCAB(conv, 64, kernel_size, act=act)
        # self.R5 = common.RCAB(conv, 64, kernel_size, act=act)
        # self.t = nn.Conv2d(64, 3, kernel_size, padding=1)

        self.x_cof = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.restormer = Restormer()
        # self.restormer.load_state_dict(torch.load('./weights/restormer_mae.pt'), strict=False)
        # print('Loading KernelNet pretrain model from {}'.format('./weights/gaussian_color_denoising_blind.pth'))
        # self.adapt = SA_conv(64, 64, 3, 1, 1)
        # self.upsample = SA_upsample_res(24)
        # self.upsample = SA_upsample(48)
        # self.proj = nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1, bias=True)
        # self.output = nn.Conv2d(48, 3, kernel_size=3, stride=1, padding=1, bias=True)
        
        
    def forward(self, x, outH, outW):
        
        _,_, h, w = x.size()
        # print(x)
        # sys.exit()
        # replicationPad2d = nn.ReplicationPad2d((0, outW-w, 0, outH-h)).to('cuda:0')
        # x = nnf.interpolate(x, size=(outH, outW), mode='bicubic', align_corners=False).to('cuda:0')
        
        # x = replicationPad2d(x)
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
        ## lf, hf 나누기
        lf_mask = torch.fliplr(torch.triu(mask, diagonal)) == 1
        hf_mask = torch.fliplr(torch.triu(mask, diagonal)) != 1
        
        lf_mask = zeroPad2d(lf_mask)
        hf_mask = zeroPad2d(hf_mask)
        
        lf_mask = lf_mask.unsqueeze(0).expand(x.size())
        hf_mask = hf_mask.unsqueeze(0).expand(x.size())

        
        ori_dct = x
        dhf = x * hf_mask
        
        coefficent = self.x_cof
        dhf = dhf * coefficent
        temp = dhf
        
        
        # dhf = self.idct(dhf)
        dhf = self.restormer(dhf)
        # x = self.upsample(x,self.total_scale,self.total_scale)
        # x = zeroPad2d(x)
        
        
        # ret = self.idct(x)
        # hf = self.idct(dhf)
        
        
        # hf = self.conv1(hf)
        # hf = self.R1(hf)
        # hf = self.R2(hf)
        # hf = self.R3(hf)
        # hf = self.R4(hf)
        # hf = self.R5(hf)
        
        
        
        # hf = self.t(hf)
        # x = self.output(x)
        # x = self.idct(x)
        
        result = x + dhf
        temp2 = result
        # result = self.idct(result)
        return result, dhf, temp, ori_dct
        # return result, dhf, temp, temp2