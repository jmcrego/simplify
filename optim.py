# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import itertools
import math
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Optimizer(object):

    def __init__(self, cfg, mod):
        self.max_grad_norm = cfg.max_grad_norm
        if cfg.method == "adam":  self.optimizer = torch.optim.Adam(mod.parameters(), lr=cfg.par.lr)
        elif cfg.method == "sgd": self.optimizer = torch.optim.SGD(mod.parameters(), lr=cfg.par.lr)
        else: sys.exit("error: bad -method {} option. Use: adam OR sgd\n".format(cfg.method))
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=cfg.par.decay, verbose=True)

    def step(self):
        # Performs a single optimization step, including gradient norm clipping if necessary
        if self.max_grad_norm > 0:
            params = itertools.chain.from_iterable([group['params'] for group in self.optimizer.param_groups])
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()

    def update_lr(self, loss):
        #Update the learning rate if the criteria of the scheduler are met
        self.scheduler.step(loss)
        return self.optimizer.state_dict()['param_groups'][0]['lr']





