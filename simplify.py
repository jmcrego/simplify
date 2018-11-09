#!/usr/local/bin/python3 -u
# -*- coding: utf-8 -*-

import sys
import torch
import random
from utils.config import Config, Params
from utils.data import Dataset, Vocab, Embed
from utils.checkpoint import Checkpoint
from utils.utils import print_time
from model.optim import Optimizer
from model.model import Model
from model.training import Training
from model.inference import Inference

import torch.nn as nn
import torch.nn.functional as F


def main():

    par = Params(sys.argv)
    random.seed(par.seed)
    torch.manual_seed(par.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(par.seed)

    if par.trn and par.val:
        chk = Checkpoint(par.dir)

        if chk.contains_model: ####### resume training ####################################
            cfg, mod, opt = chk.load(par) ### also moves to GPU if cfg.cuda
#            cfg.update_par(par) ### updates par in cfg
            print_time('Learning [resume It={}]...'.format(cfg.n_iters_sofar))

        else: ######################## training from scratch ##############################
            cfg = Config(par) ### reads cfg and par (reads vocabularies)
            mod = Model(cfg)
#            if cfg.cuda: mod.cuda() ### moves to GPU
            opt = Optimizer(cfg, mod) #build Optimizer
            print_time('Learning [from scratch]...')

        trn = Dataset(par.trn, cfg.svoc, cfg.tvoc, par.batch_size, par.max_src_len, par.max_tgt_len, do_shuffle=True, do_filter=True, is_test=False)
        val = Dataset(par.val, cfg.svoc, cfg.tvoc, par.batch_size, par.max_src_len, par.max_tgt_len, do_shuffle=True, do_filter=True, is_test=True)
        Training(cfg, mod, opt, trn, val, chk)

    elif par.tst: #################### inference ##########################################
        chk = Checkpoint()
        cfg, mod, opt = chk.load(par, par.chk)
#        cfg.update_par(par) ### updates cfg options with pars
        tst = Dataset(par.tst, cfg.svoc, cfg.tvoc, par.batch_size, 0, 0, do_shuffle=False, do_filter=False, is_test=True)
        print_time('Inference [model It={}]...'.format(cfg.n_iters_sofar))
        Inference(cfg, mod, tst)

if __name__ == "__main__":
    main()

