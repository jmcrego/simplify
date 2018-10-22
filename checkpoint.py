import os
import sys
import glob
import time
import shutil
import torch

class Checkpoint(object):

    def __init__(self, path):
        self.path = path
        self.contains_model = self.contains_model()

    def save(self, cfg, mod, opt, voc, loss):
        if not os.path.exists(self.path): os.makedirs(self.path)
        if not os.path.exists(os.path.join(self.path, 'voc.pt')): torch.save(voc, os.path.join(self.path, 'voc.pt'))
        if not os.path.exists(os.path.join(self.path, 'cfg.pt')): torch.save(cfg, os.path.join(self.path, 'cfg.pt'))
        date_time = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
        checkpoints_path = os.path.join(self.path, 'checkpoint_{}_{}_{:.4f}'.format(date_time,mod.niters,loss))
        if os.path.exists(checkpoints_path): shutil.rmtree(checkpoints_path)

        os.makedirs(checkpoints_path)
        torch.save(mod, os.path.join(checkpoints_path, 'mod.pt'))
        torch.save(opt, os.path.join(checkpoints_path, 'opt.pt'))
        sys.stderr.write("Saved checkpoint [{}]\n".format(checkpoints_path))

    def load(self, name=None):
        if not os.path.exists(self.path): sys.exit('error: no experiments found in dir='.format(self.path))
        cfg = torch.load(os.path.join(self.path, 'cfg.pt'))
        voc = torch.load(os.path.join(self.path, 'voc.pt'))

        if name is None:
            #all_saves = sorted(os.listdir(checkpoints_path), reverse=True)
            all_saves = sorted(glob.glob(self.path+'/checkpoint_*'), reverse=True)
            if len(all_saves)==0: sys.exit('error: no checkpoint found in dir={}'.format(self.path))
            last_checkpoint = all_saves[0]
        else: last_checkpoint = os.path.join(self.path, name)

        mod = torch.load(os.path.join(last_checkpoint, 'mod.pt'))
        print("Loaded It={} {}".format(mod.niters,last_checkpoint))
        opt = torch.load(os.path.join(last_checkpoint, 'opt.pt'))

        return cfg, mod, opt, voc

    def contains_model(self):
        if not os.path.exists(os.path.join(self.path, 'cfg.pt')): return False
        if not os.path.exists(os.path.join(self.path, 'voc.pt')): return False
        all_saves = glob.glob(self.path+'/checkpoint_*')
        if len(all_saves)==0: return False
        return True



