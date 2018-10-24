#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import sys
import torch
from optim import Optimizer
from config import Config, Params
from model import Model
from data import Dataset, Vocab, Embed
from training import Training
from inference import Inference
from checkpoint import Checkpoint

def main():

    par = Params(sys.argv)
    torch.manual_seed(par.seed)

    if par.trn and par.val:
        chk = Checkpoint(par.dir)

        if chk.contains_model: ####### resume training ####################
            cfg, mod, opt = chk.load() ### also moves to GPU if cfg.cuda
            cfg.update_par(par) ### updates par in cfg
            sys.stderr.write('Learning [resume It={}]...\n'.format(cfg.n_iters_sofar))

        else: ######## training from scratch ##############################
            cfg = Config(par) ### reads cfg and par (reads vocabularies)
            mod = Model(cfg)
            if cfg.cuda: mod.cuda() ### moves to GPU
            opt = Optimizer(cfg, mod) #build Optimizer
            sys.stderr.write('Learning [from scratch]...\n')

        trn = Dataset(par.trn, cfg.svoc, cfg.tvoc, par.batch_size, par.max_src_len, par.max_tgt_len, do_shuffle=True, do_filter=True)
        val = Dataset(par.val, cfg.svoc, cfg.tvoc, par.batch_size, par.max_src_len, par.max_tgt_len, do_shuffle=True, do_filter=True)
        Training(cfg, mod, opt, trn, val, chk)

    elif par.tst: ######## inference ######################################
        chk = Checkpoint()
        cfg, mod, opt = chk.load(par.chk)
        cfg.update_par(par) ### updates cfg options with pars
        tst = Dataset(par.tst, cfg.svoc, cfg.tvoc, par.batch_size, 0, 0, do_shuffle=False, do_filter=False)
        sys.stderr.write('Inference [It={}]...\n'.format(cfg.n_iters_sofar))
        Inference(cfg, mod, tst)

if __name__ == "__main__":
    main()

