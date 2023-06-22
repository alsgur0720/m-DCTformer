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
                
                
                
                
                
                inH , inW =  int(outH / total_scale), int(outW / total_scale)
                hr = hr[:outH, :outW, ...]
                
                hr_pil = hr.data.squeeze().float().cpu().numpy()
                hr_pil = (hr_pil * 255.0).round().astype(np.uint8)  # float32 to uint8
                hr_pil = np.squeeze(hr_pil)
                hr_pil = torch.from_numpy(hr_pil).to('cuda:0')
                
                
                
                
                img_hr_pil = Image.fromarray((hr_pil.cpu().numpy()).astype('uint8'))
                img_lr_pil = img_hr_pil.resize((inW, inH), Image.BICUBIC)
                lr = torch.from_numpy(np.array(img_lr_pil).astype('float32') / 255 ).permute(2, 0, 1).unsqueeze(0).to('cuda:0')
                
                
             
                
                window_size = 16
                _, _, h_old, w_old = lr.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                lr = torch.cat([lr, torch.flip(lr, [2])], 2)[:, :, :h_old + h_pad, :]
                lr = torch.cat([lr, torch.flip(lr, [3])], 3)[:, :, :, :w_old + w_pad]
                
                
               
                
                dct_modoule = dct.DCT_2D()
                # hr_dct = dct_modoule(hr)
                
                sr = self.model(lr, outH, outW)
                sr = sr[..., :h_old * int(total_scale), :w_old * int(total_scale)]
                
                
                
               

                dct_modoule_idct = dct.IDCT_2D()
                sr = dct_modoule_idct(sr)
        
                hr = hr.unsqueeze(0).permute(0,3,1,2)
              
                loss_lr = self.loss(sr, hr)

                loss = loss_lr
               
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
                    
                    H,W,C = hr.size()
                   
                    outH,outW,C = hr.size()
                    
                  
                    
                    if (outW % 4 != 0) :
                        outW = outW - (outW % 4)
                    if (outH % 4 != 0) :
                        outH = outH - (outH % 4)
                    
                    inH , inW =  int(H / total_scale), int(W / total_scale)
                    hr = hr[:outH, :outW, ...]
                    
                    hr_pil = hr.data.squeeze().float().cpu().numpy()
                    hr_pil = (hr_pil * 255.0).round().astype(np.uint8)  # float32 to uint8
                    hr_pil = np.squeeze(hr_pil)
                    hr_pil = torch.from_numpy(hr_pil).to('cuda:0')
                    
                    
                    img_hr_pil = Image.fromarray((hr_pil.cpu().numpy()).astype('uint8'))
                    hr_real = torch.from_numpy(np.array(img_hr_pil).astype('float32') / 255).permute(2, 0, 1).unsqueeze(0).to('cuda:0')
                    
                    img_lr_pil = img_hr_pil.resize((inW, inH), Image.BICUBIC)
                 
                    lr = torch.from_numpy(np.array(img_lr_pil).astype('float32') / 255).permute(2, 0, 1).unsqueeze(0).to('cuda:0')
                                  
                    
                    dct_modoule = dct.DCT_2D()
                    idct_module = dct.IDCT_2D()
                    
                    window_size = 16
                    _, _, h_old, w_old = lr.size()
                    h_pad = (h_old // window_size + 1) * window_size - h_old
                    w_pad = (w_old // window_size + 1) * window_size - w_old
                    lr = torch.cat([lr, torch.flip(lr, [2])], 2)[:, :, :h_old + h_pad, :]
                    lr = torch.cat([lr, torch.flip(lr, [3])], 3)[:, :, :, :w_old + w_pad]
                    
                    hr = hr.data.squeeze().float().cpu().numpy()
               
                    hr = (hr * 255.0).round().astype(np.uint8)  # float32 to uint8

                    hr = np.squeeze(hr)
                    
                    
                    sr = self.model(lr, outH, outW, h_old, w_old)
               
                    sr = idct_module(sr)
                    
                    sr = sr.data.squeeze().permute(1,2,0).float().cpu().numpy()
                    

                    timer_test.hold()


                    if not no_eval:
                   
                        psnr = utility.calculate_psnr(sr, hr, crop_border=2)
                        ssim = utility.calculate_ssim(sr, hr, crop_border=2)
                        
                       

                        eval_simm += ssim
                        eval_psnr += psnr

                
                    self.ckp.save_img(filename, sr, lr)
                    

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
        
