
from multiprocessing.spawn import freeze_support

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import utility
import data
import model
import loss
from option import args
from checkpoint import Checkpoint
from trainer import Trainer
from ptflops import get_model_complexity_info
import torch

# print("main scale >>"+str(args.scale[0]))
utility.set_seed(args.seed)
checkpoint = Checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    
    # with torch.cuda.device(0):
        
        # net = model
        # macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                                # print_per_layer_stat=True, verbose=True)
        # print(macs)
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        # exit()
        
        
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    def main():
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()

    if __name__ == '__main__':
        freeze_support()
        main()




