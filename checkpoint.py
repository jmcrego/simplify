import os
import sys
import glob
import time
import shutil
import torch
from model import Model
from optim import Optimizer

class Checkpoint(object):

    def __init__(self, path=None):
        if path is not None:
            self.path = path
            self.contains_model = self.contains_model()

    def save(self, cfg, mod, opt, loss):
        if not os.path.exists(self.path): os.makedirs(self.path)
#        date_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
        date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        checkpoint = os.path.join(self.path, 'checkpoint_{}_{:0>6}_{}.pt'.format(date_time,cfg.n_iters_sofar,"{:.5f}".format(loss)[0:7]))
        chk = {'mod': mod.state_dict(), 'opt': opt.state_dict(), 'cfg': cfg}
        torch.save(chk, checkpoint) 
        sys.stderr.write("Saved checkpoint [{}]\n".format(checkpoint))

    def load(self, name=None):
        if name is None:
            if not os.path.exists(self.path): sys.exit('error: no experiments found in {}'.format(self.path))
            all_saves = sorted(glob.glob(self.path+'/checkpoint_*.pt'), reverse=True)
            if len(all_saves)==0: sys.exit('error: no checkpoint found in dir={}'.format(self.path))
            checkpoint = all_saves[0]
        else: 
            if not os.path.exists(name): sys.exit('error: no checkpoint found in {}'.format(name))
            checkpoint = name
        chk = torch.load(checkpoint) 
        ### load cfg
        cfg = chk['cfg'] 
        ### load model
        mod = Model(cfg)
        mod.load_state_dict(chk['mod'])
        if cfg.cuda: mod.cuda() ### move to GPU
        ### load optimizer
        opt = Optimizer(cfg,mod)
        opt.optimizer.load_state_dict(chk['opt'])
        sys.stderr.write("Loaded It={} {}\n".format(cfg.n_iters_sofar,checkpoint)) 
        return cfg, mod, opt

    def contains_model(self):
        all_saves = glob.glob(self.path+'/checkpoint_*.pt')
        if len(all_saves)==0: return False
        return True



