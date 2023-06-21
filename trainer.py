import torch
import numpy as np
import pandas as pd
import cv2
import utility
from decimal import Decimal
from tqdm import tqdm
from option import args
import os
from torchvision import transforms 
from torchvision.utils import save_image 
from PIL import Image
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
import math
import imageio
import torch.nn.functional as nnf
import sys
from model import common, dct

import matplotlib.pyplot as plt

from thop import profile


class Trainer():
    def __init__(self, opt, loader, my_model, my_loss, ckp):
        self.opt = opt
        self.scale = opt.scale
        self.float_scale = opt.float_scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(opt, self.model)
        self.scheduler = utility.make_scheduler(opt, self.optimizer)
        self.error_last = 1e8

    def train(self):
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_last_lr()[0]

        int_scale = max(self.scale)
        float_scale = self.float_scale
        total_scale = int_scale + float_scale
        res_scale = total_scale / int_scale 
        self.ckp.set_epoch(epoch)


        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        # for name, param in self.model.named_parameters():
            # splitname = name.split('.')
            # if not 'weight_modulation' in name :
                # param.requires_grad = False
            # if 'swinIR' in name:
                # param.requires_grad = False
                # print(name)
            # if not 'h2a2sr' in name:
                # param.requires_grad = False

        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, idx) in enumerate(self.loader_train):
            # lr, hr = self.prepare(lr, hr)
            hr_ori, lr_ori = self.prepare(hr, lr)

            timer_data.hold()
            timer_model.tic()
            
            self.optimizer.zero_grad()
            # _,_, outH, outW = hr.size()
            b,C,outH,outW = hr.size()
            for i in range (0, b):
                hr = hr_ori[i]
                lr = lr_ori[i]
                # hr = hr.unsqueeze(0)
                # lr - lr.unsqueeze(0)
                
                # lr = hr
                # _,C,H,W = hr.size()
                # N,C,H_l,W_l = lr.size()
                # outH, outW = int(total_scale * H_l), int(total_scale * W_l)
                
                
                # outH, outW = int(total_scale * inH), int(total_scale * inW)
                outH, outW, C = hr.size()
                # outH = outH - 2
                # outW = outW - 2
                
                
                if (outW % 4 != 0) :
                        outW = outW - (outW % 4)
                if (outH % 4 != 0) :
                        outH = outH - (outH % 4)
                hr = hr[:outH,  :outW, ...]
                
                
                
                
                ############## swinir
                
                inH , inW =  int(outH / total_scale), int(outW / total_scale)
                hr = hr[:outH, :outW, ...]
                
                hr_pil = hr.data.squeeze().float().cpu().numpy()
                hr_pil = (hr_pil * 255.0).round().astype(np.uint8)  # float32 to uint8
                hr_pil = np.squeeze(hr_pil)
                hr_pil = torch.from_numpy(hr_pil).to('cuda:0')
                
                
                
                
                img_hr_pil = Image.fromarray((hr_pil.cpu().numpy()).astype('uint8'))
                img_lr_pil = img_hr_pil.resize((inW, inH), Image.BICUBIC)
                lr = torch.from_numpy(np.array(img_lr_pil).astype('float32') / 255 ).permute(2, 0, 1).unsqueeze(0).to('cuda:0')
                
                
                ## real
                
                window_size = 16
                _, _, h_old, w_old = lr.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                lr = torch.cat([lr, torch.flip(lr, [2])], 2)[:, :, :h_old + h_pad, :]
                lr = torch.cat([lr, torch.flip(lr, [3])], 3)[:, :, :, :w_old + w_pad]
                
                ############## swinir
                
                
                
                ####### matplot 안쓰면 쓰는 코드 #######
                
                            
                # _,C,H,W = hr.size()
                # # inH , inW =  int(H / 1.25), int(W / 1.25)
                # inH , inW =  int(H / total_scale), int(W / total_scale)
                
                # _, _, hr_H, hr_W = hr.size()
                
                # img_hr_pil = Image.fromarray((hr.squeeze(0).permute(1, 2, 0).cpu().numpy()).astype('uint8'))
                # img_lr_pil = img_hr_pil.resize((inW, inH), Image.BICUBIC)
                # # img_lr_pil = img_hr_pil.resize((outW, outH), Image.BICUBIC)
                # lr = torch.from_numpy(np.array(img_lr_pil).astype('float32')).permute(2, 0, 1).unsqueeze(0).to('cuda:0')

                ####### matplot 안쓰면 쓰는 코드 #######
                
                
                # hr = utility.quantize(hr, self.opt.rgb_range)
                # lr = utility.quantize(lr, self.opt.rgb_range)
                
                # h_pad = (h_old // 8 + 1) * 8 - h_old
                # w_pad = (w_old // 8 + 1) * 8 - w_old
                # lr = torch.cat([lr, torch.flip(lr, [2])], 2)[:, :, :h_old + h_pad, :]
                # lr = torch.cat([lr, torch.flip(lr, [3])], 3)[:, :, :, :w_old + w_pad]
                # mask = torch.ones((H, W), dtype=torch.int64, device = torch.device('cuda:0'))
                # diagonal = W-2
                # hf_mask = torch.fliplr(torch.triu(mask, diagonal)) != 1
                
                dct_modoule = dct.DCT_2D()
                # hr_dct = dct_modoule(hr)
                
                sr = self.model(lr, outH, outW)
                sr = sr[..., :h_old * int(total_scale), :w_old * int(total_scale)]
                # sr, dct_x, hf, x = self.model(lr, outH, outW)
                
                
                # if isinstance(sr, list): sr = sr[-1]
                # # compute primary loss
                
                
                # hr_hf = hr_hf * hf_mask
                # x = np.log(abs(x))
                # hr_hf = np.log(abs(hr_hf))
                # loss_dct = self.loss(x, hr_hf)
                
                # loss_dct = self.loss(sr, hr_dct)
                dct_modoule_idct = dct.IDCT_2D()
                sr = dct_modoule_idct(sr)
                # sr = sr[:, :, : outH,  :outW]
                hr = hr.unsqueeze(0).permute(0,3,1,2)
                # print(hr.shape)
                # exit()
                loss_lr = self.loss(sr, hr)

                loss = loss_lr
                # loss = loss_dct
                # loss = loss_lr + loss_dct
                if loss.item() < self.opt.skip_threshold * self.error_last:
                    loss.backward()                
                    self.optimizer.step()
                else:
                    print('Skip this batch {}! (Loss: {})'.format(
                        i + 1, loss.item()
                    ))
                    
                timer_model.hold()

            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

                timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.step()
    
    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, 1))
        self.model.eval()


        timer_test = utility.timer()
        with torch.no_grad():

            int_scale = max(self.scale)
            float_scale = self.float_scale
            total_scale = int_scale + float_scale
            res_scale = total_scale / int_scale 

            for si, s in enumerate([int_scale]):
                eval_psnr = 0
                eval_simm = 0
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for batch, (lr, hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        hr, lr = self.prepare(hr, lr)
                    else:
                        lr = self.prepare(lr)

                    hr = hr[0]
                    lr = lr[0]
                    # _, _, h_old, w_old = lr.size()
                    # print(lr.size())
                    
                    # print(hr)
                    # print(lr)
                    # exit()
                    H,W,C = hr.size()
                   
                    outH,outW,C = hr.size()
                    
                    #### matplot 안쓰면 쓰는 코드 
                    
                    # inH , inW =  int(H / total_scale), int(W / total_scale)
                    
                    if (outW % 4 != 0) :
                        outW = outW - (outW % 4)
                    if (outH % 4 != 0) :
                        outH = outH - (outH % 4)
                    
                    # hr = hr[outH,outW, :]
                    
                    # total_scale = 1.03
                    # inH , inW =  int(H / 1.025), int(W / 1.025)
                    # outH = inH * 3
                    # outW = inW * 3
                    
                    ######################### swinir
                    
                    
                    inH , inW =  int(H / total_scale), int(W / total_scale)
                    hr = hr[:outH, :outW, ...]
                    
                    hr_pil = hr.data.squeeze().float().cpu().numpy()
                    hr_pil = (hr_pil * 255.0).round().astype(np.uint8)  # float32 to uint8
                    hr_pil = np.squeeze(hr_pil)
                    hr_pil = torch.from_numpy(hr_pil).to('cuda:0')
                    
                    
                    
                    
                    img_hr_pil = Image.fromarray((hr_pil.cpu().numpy()).astype('uint8'))
                    hr_real = torch.from_numpy(np.array(img_hr_pil).astype('float32') / 255).permute(2, 0, 1).unsqueeze(0).to('cuda:0')
                    
                    img_lr_pil = img_hr_pil.resize((inW, inH), Image.BICUBIC)
                    # lr = utility.rgb2ycbcr(np.array(img_lr_pil)) 
                    lr = torch.from_numpy(np.array(img_lr_pil).astype('float32') / 255).permute(2, 0, 1).unsqueeze(0).to('cuda:0')
                    # print(lr.size())
                    # exit()                  
                    
                   
                   
                    ######################### swinir
                    
                    
                    
                    
                    ################## edsr lr code
                    
                    # img_hr_pil = Image.fromarray((hr_pil.cpu().numpy()).astype('uint8'))
                    # img_lr_pil = img_hr_pil.resize((inW, inH), Image.BICUBIC)
                    # img_lr_pil = img_hr_pil.resize((outW, outH), Image.BICUBIC)
                    # lr = torch.from_numpy(np.array(img_lr_pil).astype('float32') / 255 ).permute(2, 0, 1).unsqueeze(0).to('cuda:0')
                    # print(lr)
                    # exit()
                    # hr = torch.from_numpy(np.array(img_hr_pil).astype('float32')).permute(2, 0, 1).unsqueeze(0).to('cuda:0')
                    ################# edsr lr code
                    
                    
                    
                    ################### rdn lr, hr code
                    # lr = torch.from_numpy(np.array(img_lr_pil).astype('float32') / 255).permute(2, 0, 1).unsqueeze(0).to('cuda:0')
                    # hr = torch.from_numpy(np.array(img_hr_pil).astype('float32') / 255).permute(2, 0, 1).unsqueeze(0).to('cuda:0')
                    ################### rdn lr, hr code
                    
                    
                    # print(lr)
                    
                    # sys.exit()
                    
                    # lr = nnf.interpolate(hr, size=(inH, inW), mode='bicubic', align_corners=False).to('cuda:0')
                    
                    #### matplot 안쓰면 쓰는 코드 
                    
                    dct_modoule = dct.DCT_2D()
                    idct_module = dct.IDCT_2D()
                    
                    # N,C,H,W = hr.size()
                    
                    # inH , inW =  int(H / total_scale), int(W / total_scale)
                    # outH, outW = int(total_scale * inH), int(total_scale * inW)
                    # hr = hr[:, :, : outH,  :outW]
                    # hr = utility.quantize(hr, self.opt.rgb_range)
                    # lr = nnf.interpolate(hr, size=(inH, inW), mode='bicubic', align_corners=False).to('cuda:0')
                    
                    
                    # lr = lr[0].to('cuda')
                    # _, _, h_old, w_old = lr.size()
                    # h_pad = (h_old // 8 + 1) * 8 - h_old
                    # w_pad = (w_old // 8 + 1) * 8 - w_old
                    # lr = torch.cat([lr, torch.flip(lr, [2])], 2)[:, :, :h_old + h_pad, :]
                    # lr = torch.cat([lr, torch.flip(lr, [3])], 3)[:, :, :, :w_old + w_pad]
                    
                    # print(hr[0].shape)
                    # exit()
                    
                    
                    # sr = self.model(lr, outH, outW)
                    
                    # print(sr.size())
                    # print(sr)
                    # sys.exit()
                    
                    
                    ##### real sr code
                    
                    # outH = int(outH * total_scale)
                    # outW = int(outW * total_scale)
                    # if (outW % 4 != 0) :
                    #     outW = outW - (outW % 4)
                    # if (outH % 4 != 0) :
                    #     outH = outH - (outH % 4)
                        
                    ##### real sr code
                    # print(lr.size())
                    # print(hr_real.size())
                    # exit()
                    
                    
                    ## real code
                    # lr = hr_real
                    # N,C,H,W = lr.size()
                    # outH = int(H * 2.9)
                    # outW = int(W * 2.9)
                    # if (outW % 4 != 0) :
                    #     outW = outW - (outW % 4)
                    # if (outH % 4 != 0) :
                    #     outH = outH - (outH % 4)
                    # print(lr.size())
                    # exit()
                    ## real code
                    
                    
                    ########## swinir
                    
                    window_size = 16
                    _, _, h_old, w_old = lr.size()
                    h_pad = (h_old // window_size + 1) * window_size - h_old
                    w_pad = (w_old // window_size + 1) * window_size - w_old
                    lr = torch.cat([lr, torch.flip(lr, [2])], 2)[:, :, :h_old + h_pad, :]
                    lr = torch.cat([lr, torch.flip(lr, [3])], 3)[:, :, :, :w_old + w_pad]
                    
                    
                    
                    
                    hr = hr.data.squeeze().float().cpu().numpy()
                    # hr = np.transpose(hr[[2, 1, 0], :, :], (1, 2, 0)) 
                    hr = (hr * 255.0).round().astype(np.uint8)  # float32 to uint8
                    # hr = hr[:h_old * int(total_scale), :w_old * int(total_scale), ...]  # crop gt
                    hr = np.squeeze(hr)
                    
                                    
                    ########## swinir
                    
                    
                    
                    # input = torch.randn(1, 3, 102, 102).cuda()
                    # macs, params = profile(self.model, inputs=(input, 256, 256), verbose=False)
                    # print(macs/1e9)
                    # print(params)
                    # print(1e9)
                    # exit()
                    # print(lr.size())
                    # exit()
                    
                    sr = self.model(lr, outH, outW, h_old, w_old)
                    ########## swinir
                    sr = idct_module(sr)
                    #  with torch.no_grad():
                    # pad input image to be a multiple of window_size
                    # output = test(img_lq, model, args, window_size)
                    # sr = sr[..., :h_old * 2, :w_old * 2]
                    # # save image
                    sr = sr.data.squeeze().permute(1,2,0).float().cpu().numpy()
                    # if sr.ndim == 3:
                        # sr = np.transpose(sr[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                    # sr = (sr * 255.0).round().astype(np.uint8)  # float32 to uint8
                    # print(sr)
                    # print(hr)
                    # cv2.imwrite(f'{save_dir}/{imgname}_SwinIR.png', output)

                    # evaluate psnr/ssim/psnr_b
                    # if img_gt is not None:
                    
                    ########## swinir
                    
                    
                    
                    
                    
                    
                    
                    
                    # sr, after_restormer, before_restormer, lr_dct = self.model(lr,outH,outW)
                    # print(after_restormer.size())
                    # if not no_eval:
                    #     eval_psnr += utility.calc_psnr_arb(
                    #         sr_arb, hr, [2.1, 2.1], self.opt.rgb_range,
                    #         benchmark=self.loader_test.dataset.benchmark
                    #     )
                        # eval_ssim += utility.calc_ssim(
                        #     sr, hr, [2.1, 2.1],
                        #     benchmark=self.loader_test.dataset.benchmark
                        # )
                    
                    
                    # output = sr
                    
                    # output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                    # output = output.data.squeeze().float().cpu().numpy()
                    # output = np.log(np.abs(output))
                    # output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
                    # output = (output).round().astype(np.uint8)
                    
                    
                    ############# 2023 05 11
                    
                    # sr = idct_module(sr)
                    
                    ############# 2023 05 11
                    
                    
                    # sr = utility.quantize(sr, self.opt.rgb_range)
                    
                    # sr = sr[:, :, : outH,  :outW]
                    
                    # sr = sr[..., :h_old * 2, :w_old * 2]
                    
                    
                    # sr_y = utility.convert_rgb_to_y(utility.denormalize(sr.squeeze(0)), dim_order='chw')
                    # hr_y = utility.convert_rgb_to_y(utility.denormalize(hr.squeeze(0)), dim_order='chw')
                    
                    # print(hr_y)
                    # print(sr_y)
                    # sys.exit()
                    
                    ############# edsr 하면 끄는거
                    
                    # print(sr)
                    # sr = sr.data.squeeze().float().cpu().numpy()
                    # sr = np.transpose(sr[[2, 1, 0], :, :], (1, 2, 0))
                    # sr = (sr).round().astype(np.uint8)
                    # print(sr)
                    # sys.exit()
                    
                    ############# edsr 하면 끄는거
                    
                    
                    # hr_dct = dct_modoule(hr)
                                            
                                            
                                            
                                            
                    ############# edsr 하면 끄는거                        
                    
                    # hr = hr.data.squeeze().float().cpu().numpy()
                    # hr = np.transpose(hr[[2, 1, 0], :, :], (1, 2, 0))
                    # hr = (hr).round().astype(np.uint8)  # float32 to uint8
                    
                    ############# edsr 하면 끄는거
                    
                    
                    
                    
                    
                    
                    
                    # hr_dct = hr_dct.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                    # hr_dct = hr_dct.data.squeeze().float().cpu().numpy()
                    # hr_dct = np.log(np.abs(hr_dct))
                    # hr_dct = np.transpose(hr_dct[[2, 1, 0], :, :], (1, 2, 0))
                    # hr_dct = (hr_dct).round().astype(np.uint8)  # float32 to uint8
                    # lr_dct = lr_dct.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                    
                    ######################### 2023 05 11
                    
                    # lr_dct = dct_modoule(hr)
                    # lr_dct = lr_dct.data.squeeze().float().cpu().numpy()
                    # lr_dct = cv2.magnitude(lr_dct, lr_dct)
                    # cv2.normalize(lr_dct, lr_dct, 0, 1000, cv2.NORM_MINMAX)
                    # lr_dct = np.transpose(lr_dct[[2, 1, 0], :, :], (1, 2, 0))
                    # lr_dct = (lr_dct).round().astype(np.uint8)  # float32 to uint8
                    
                    
                    #############################
                    
                    # print(lr_dct.max())
                    # sys.exit()
                    
                    
                    
                    # after_restormer = after_restormer.data.squeeze().float().cpu().clamp_(0,1).numpy()
                    # print("max :",after_restormer.max())
                    # print("min :",after_restormer.min())
                    # after_restormer = np.log(np.abs(after_restormer))
                    # print("max :",after_restormer.max())
                    # print("min :",after_restormer.min())
                    
                    ##################### 2023 05 11
                    
                    # after_restormer = after_restormer.data.squeeze().float().cpu().numpy()
                    # # print(after_restormer.shape)
                    
                    # after_restormer = cv2.magnitude(after_restormer, after_restormer)
                    # cv2.normalize(after_restormer, after_restormer, 0, 25000, cv2.NORM_MINMAX)
                    # after_restormer = np.transpose(after_restormer[[2, 1, 0], :, :], (1, 2, 0))
                    # # print(after_restormer.shape)
                    # # exit()
                    # after_restormer = (after_restormer).round().astype(np.uint8)
                    
                    ##################### 2023 05 11
                    
                    
                    # after_restormer = np.real(after_restormer)
                    # after_restormer = np.mean(after_restormer, axis=-1)
                    # print(after_restormer.max())
                    # print(after_restormer.min())
                    # plt.imshow(after_restormer, cmap='gray')
                    # plt.title('Discrete Cosine Transform (DCT) Coefficients')
                    # plt.show()
                    # sys.exit()
                    
                    # before_restormer = idct_module(before_restormer)
                    # before_restormer = before_restormer.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                    
                    
                    ##################### 2023 05 11
                    
                    # before_restormer = before_restormer.data.squeeze().float().cpu().numpy()
                    # before_restormer = cv2.magnitude(before_restormer, before_restormer)
                    # cv2.normalize(before_restormer, before_restormer, 0, 25000, cv2.NORM_MINMAX)
                    # before_restormer = np.transpose(before_restormer[[2, 1, 0], :, :], (1, 2, 0))
                    # before_restormer = (before_restormer).round().astype(np.uint8)
                    
                    ##################### 2023 05 11
                    
                    # before_restormer = (before_restormer * 255.0).round().astype(np.uint8)
                    
                    
                    
                    # hr = hr[:h_old * 2, :w_old * 2, ...]  # crop gt
                    # hr = np.squeeze(hr)
                    
                    # hr = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
                    # sr = sr.contiguous().view(N, C, outH, outW)  
                    # sr = utility.quantize(sr, self.opt.rgb_range)

                    timer_test.hold()


                    if not no_eval:
                    # if False:
                        # psnr = utility.calc_psnr(
                        #     sr, hr, s, self.opt.rgb_range,
                        #     benchmark=self.loader_test.dataset.benchmark
                        # )
                        
                        
                        # psnr = utility.calc_psnr_rdn(hr_y,sr_y)
                        # psnr = utility.calc_psnr_edsr(sr, hr, self.opt.scale, self.opt.rgb_range)
                        psnr = utility.calculate_psnr(sr, hr, crop_border=2)
                        ssim = utility.calculate_ssim(sr, hr, crop_border=2)
                        
                        # hr_numpy = hr[0].cpu().numpy().transpose(1, 2, 0)
                        # sr_numpy = sr[0].cpu().numpy().transpose(1, 2, 0)
                        # simm = utility.SSIM(hr_numpy, sr_numpy)

                        eval_simm += ssim
                        eval_psnr += psnr

                    # if self.opt.save_results:
                    self.ckp.save_img(filename, sr, lr)
                    # self.ckp.save_img_2(filename, after_restormer, before_restormer,lr_dct)
                        # self.ckp.save_results_nopostfix(filename, sr, s)

                self.ckp.log[-1, si] = eval_psnr / len(self.loader_test)
                eval_simm = eval_simm / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.2f} (Best: {:.2f} @epoch {})'.format(
                        self.opt.data_test, s,
                        self.ckp.log[-1, si],
                        best[0][si],
                        best[1][si] + 1
                    )
                )
                print('SIMM:',eval_simm)



        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.opt.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def step(self):
        self.scheduler.step()

    def prepare(self, hr, lr):
        device = torch.device('cpu' if self.opt.cpu else 'cuda')
        
        # if len(args) > 1:
        #     return [a.to(device) for a in args[0]], args[-1].to(device)
        # return [a.to(device) for a in args[0]],
        return hr.to(device), lr.to(device)

    def terminate(self):
        if self.opt.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.opt.epochs
        