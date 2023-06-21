import os
import torch
import datetime
import numpy as np
import imageio
from skimage.util import img_as_float
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from option import args


class Checkpoint():
    def __init__(self, opt):
        self.opt = opt
        self.ok = True

        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if opt.save == '.': opt.save = '../experiment/EXP/' + now
        self.dir = opt.save

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(opt):
                f.write('{}: {}\n'.format(arg, getattr(opt, arg)))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)

        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.opt.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate([self.opt.scale[0]]):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs') 
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.opt.data_test))
        plt.close(fig)

    def set_epoch(self, ep):
        self.epoch = ep
    
    
    
    def save_img(self, fimename, sr, lr) :
        
        # sr = sr.data.squeeze().float().cpu().numpy()
        # sr = np.transpose(sr[[2, 1, 0], :, :], (1, 2, 0))
        # sr = (sr).round().astype(np.uint8)
        
        # lr = lr.data.squeeze().float().cpu().numpy()
        # lr = np.transpose(lr[[2, 1, 0], :, :], (1, 2, 0))
        # lr = (lr).round().astype(np.uint8)  # float32 to uint8
        
        
        save_dir = self.dir + '/results'
        cv2.imwrite(f'{save_dir}/{fimename}_SR.png', sr)
        # cv2.imwrite(f'{save_dir}/{fimename}_LR.png', lr)
        
        
    def save_img_2(self, fimename, after_restormer, before_restormer, lr_dct) :
        save_dir = self.dir + '/results'
        after_restormer = cv2.cvtColor(after_restormer, cv2.COLOR_BGR2GRAY)
        # hr_dct = cv2.cvtColor(hr_dct, cv2.COLOR_BGR2GRAY)
        before_restormer = cv2.cvtColor(before_restormer, cv2.COLOR_BGR2GRAY)
        lr_dct = cv2.cvtColor(lr_dct, cv2.COLOR_BGR2GRAY)
        # output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'{save_dir}/{fimename}_after_restormer.png', after_restormer)
        cv2.imwrite(f'{save_dir}/{fimename}_before_restormer.png', before_restormer)
        # cv2.imwrite(f'{save_dir}/{fimename}_hr_dct.png', hr_dct)
        # cv2.imwrite(f'{save_dir}/{fimename}_output.png', output)
        cv2.imwrite(f'{save_dir}/{fimename}_lr_dct.png', lr_dct)
        
        
        
    def save_results_nopostfix(self, filename, sr, scale):
        apath = '{}/results/{}/x{}'.format(self.dir, self.opt.data_test, scale)
        if not os.path.exists(apath):
            os.makedirs(apath)
        filename = os.path.join(apath, filename)

        normalized = sr.data.mul(255 / self.opt.rgb_range)
        ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
        imageio.imwrite('{}.jpg'.format(filename), ndarr)