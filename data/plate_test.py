import os
from data import srdata


class plate_test(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(plate_test, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, data_dir):
        self.apath = os.path.join(data_dir, 'benchmark', self.name)
        s = str(self.args.total_scale).split('.')
        
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR')
        self.ext = ('', '.jpg')

