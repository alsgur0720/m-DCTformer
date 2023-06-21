import os
from data import srdata


class b100(srdata.SRData):
    def __init__(self, args, name='b100', train=True, benchmark=False):
        super(b100, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, data_dir):
        super(b100, self)._set_filesystem(data_dir)
        s = str(self.args.total_scale).split('.')
        
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic/X'+str(self.args.int_scale))
        self.dir_hr = 'D:/swin2sr-main/swin2sr-main/testsets/Set5/HR'
        self.dir_lr = 'D:/swin2sr-main/swin2sr-main/testsets/Set5/LR_bicubic/X2'
        self.ext = ('', '.png')

