import os
import glob
import torch
from data import common
import numpy as np
import imageio
import torch.utils.data as data
import random
import math
import cv2
import sys
import utility
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_lmdb, scandir
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import imresize, rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY


# import PIL.Image as pil_image

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.scale = args.scale.copy()
        self.scale.reverse()
        self.total_scale = args.total_scale
        self._set_filesystem(args.data_dir)
        self._get_imgs_path(args)
        self._set_dataset_length()
    
    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        # lr, hr, filename = self._load_file_edsr(idx)
        # print(hr.shape)
        # exit()
        lr, hr = self.get_patch(lr, hr)
        # print(hr.shape)
        
        # lr, hr = common.set_channel(lr, hr, n_channels=self.args.n_colors)
        # lr_tensor, hr_tensor = common.np2Tensor(
        #     lr, hr, rgb_range=self.args.rgb_range
        # )
        # return lr_tensor, hr_tensor, filename 
        return lr, hr, filename 

    def __len__(self):
        return self.dataset_length

    def _get_imgs_path(self, args):
        list_hr, list_lr = self._scan()
        self.images_hr, self.images_lr = list_hr, list_lr

    def _set_dataset_length(self):
        if self.train:
            self.dataset_length = self.args.test_every * self.args.batch_size
            repeat = self.dataset_length // len(self.images_hr)
            self.random_border = len(self.images_hr) * repeat
        else:
            self.dataset_length = len(self.images_hr)

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, '{}{}'.format(
                        filename, self.ext[1]
                    )
                ))

        return names_hr, names_lr

    def _set_filesystem(self, data_dir):
        self.apath = os.path.join(data_dir, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('.jpg', '.jpg')

    def _get_index(self, idx):
        if self.train:
            if idx < self.random_border:
                return idx % len(self.images_hr)
            else:
                return np.random.randint(len(self.images_hr))
        else:
            return idx

    
    # def _Load_file_rdn(self, idx):
    #     idx = self._get_index(idx)
    #     f_hr = self.images_hr[idx]
    #     filename, _ = os.path.splitext(os.path.basename(f_hr))
    #     image = pil_image.open(f_hr).convert('RGB')
    #     hr = np.array(image)
        
    #     # image_width = (image.width // self.scale) * self.scale
    #     # image_height = (image.height // self.scale) * self.scale
        
    #     # hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    #     # lr = hr.resize((hr.width // self.scale, hr.height // self.scale), resample=pil_image.BICUBIC)
        
    #     hr = np.array(hr).astype(np.float32).transpose([2, 0, 1]) / 255.0
    #     hr = torch.from_numpy(hr).float()
    #     lr = hr
        
    #     return lr, hr, filename
        
    def _load_file_edsr(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        # f_lr = self.images_lr[self.idx_scale][idx]
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        # if self.args.ext == 'png' or self.benchmark:
        hr = imageio.imread(f_hr)
        lr = hr
        # print(f'{self.dir_lr}/{filename}.png')
        # sys.exit()
        # lr = imageio.imread(f'{self.dir_lr}/{filename}.png')
        # elif self.args.ext.find('sep') >= 0:
        #     with open(f_hr, 'rb') as _f:
        #         hr = pickle.load(_f)
        #     with open(f_lr, 'rb') as _f:
        #         lr = pickle.load(_f)

        return lr, hr, filename
    
    
    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        # f_lr = [self.images_lr[idx_scale][idx] for idx_scale in range(len(self.scale))]
        # f_lr = [self.images_lr[idx_scale][idx] for idx_scale in range(len(self.scale))]


        filename, _ = os.path.splitext(os.path.basename(f_hr))
        img_gt = cv2.imread(f'{self.dir_hr}/{filename}.png', cv2.IMREAD_COLOR).astype(np.float32) / 255
        # img_gt = imfrombytes(img_gt, float32=True)
        
        # print(img_gt)
        # exit()
        # print(f'{self.dir_lr}/{filename}.png')
        # exit()
        # img_lq = cv2.imread(f'{self.dir_lr}/{filename}.png', cv2.IMREAD_COLOR).astype(
                # np.float32) / 255.
        img_lq = img_gt.copy()
        # img_lq = img_gt
        # print(f'{self.dir_lr}/{filename}.png')
        # sys.exit()
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))# HCW-BGR to CHW-RGB
        
        # img_lq = utility.rgb2ycbcr(img_lq)
        img_lq = torch.from_numpy(img_lq.copy()).float().unsqueeze(0)  # CHW-RGB to NCHW-RGB
        
        
        
        # img_gt = np.transpose(img_gt if img_gt.shape[2] == 1 else img_gt[:, :, [2, 1, 0]], (2, 0, 1))   
        # img_gt = torch.from_numpy(img_gt).float().unsqueeze(0)  # CHW-RGB to NCHW-RGB
        
        # img_gt = np.transpose(img_gt if img_gt.shape[2] == 1 else img_gt[:, :, [2, 1, 0]], (2, 0, 1))   
        # img_gt = torch.from_numpy(img_gt).float().unsqueeze(0)  # CHW-RGB to NCHW-RGB
        
        # img_lq = img_gt
        
        # hr = imageio.imread(f_hr)
        # lr = [imageio.imread(f_lr[idx_scale]) for idx_scale in range(len(self.scale))]
        return img_lq, img_gt, filename

    
    def get_patch(self, lr, hr):
        scale = self.total_scale
        multi_scale = len(self.scale) > 1
        if self.train:
            lr, hr = new_gp(
                lr,
                hr,
                patch_size=self.args.patch_size,
                scale=scale,
                scale2=scale
            )
            if not self.args.no_augment:
                lr, hr = common.augment(lr, hr)
        else:
            if isinstance(lr, list):
                ih, iw = lr[0].shape[:2]
            else:
                ih, iw = lr.shape[:2]


        return lr, hr


  

def new_gp(*args, patch_size=96, scale=1, scale2=1):
    
    ih, iw = args[0].shape[2:4]  ## LR image
    
    ih, iw = int(ih / scale), int(iw / scale2)
    ih , iw =  int(ih / scale), int(iw / scale2)
    
    tp = int(round(scale * patch_size))
    tp2 = int(round(scale2 * patch_size))
    
    ip = patch_size

    if scale==int(scale):
        step = 1
    elif (scale*2)== int(scale*2):
        step = 2
    elif (scale*5) == int(scale*5):
        step = 5
    else:
        step = 10
    if scale2==int(scale2):
        step2 = 1
    elif (scale2*2)== int(scale2*2):
        step2 = 2
    elif (scale2*5) == int(scale2*5):
        step2 = 5
    else:
        step2 = 10

    # print("ih : ", ih)
    # print("ip : ", ip)
    # print("ih - ip: ", ih-ip)
    # print("ih - ip // step-2: ", (ih-ip)//step-2)
    iy = random.randrange(2, ((ih-ip)//step-2)+5) * step
    ix = random.randrange(2, ((iw-ip)//step-2)+5) * step
    
    tx, ty = int(round(scale2 * ix)), int(round(scale * iy))
    # print(tx)
    # print(tx+tp2)
    # print(ty)
    # print(ty+tp)
    # exit()
    ret = [
        args[0][:,:, iy:iy + ip, ix:ix + ip],
        *[a[ty:ty + tp, tx:tx + tp2,:] for a in args[1:]]
    ]
    
    # ret = [
    #     args[0][iy:iy + ip, ix:ix + ip, :],
    #     *[a[ty:ty + tp, tx:tx + tp2, : ] for a in args[1:]]
    # ]
    
    return ret





