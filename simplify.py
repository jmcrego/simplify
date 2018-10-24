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
        if chk.contains_model:
            cfg, mod, opt = chk.load()            
            cfg.add(par) ### updates cfg options read in checkpoint 
            sys.stderr.write('Learning [resume It={}]...\n'.format(cfg.n_iters_sofar))
        else:
            voc = Vocab(par.voc)
            cfg = Config(par.cfg)
            cfg.add(par, voc)
#            emb = Embed(par.emb, cfg, voc) #embeddings read from file OR randomly initialised (length of vectors must match with that given in config file)
            mod = Model(cfg)
            if cfg.cuda: mod.cuda()
            opt = Optimizer(cfg, mod) #build Optimizer
            sys.stderr.write('Learning [from scratch]...\n')

        trn = Dataset(par.trn, cfg.voc, par.batch_size, par.max_src_len, par.max_tgt_len, do_shuffle=True, do_filter=True)
        val = Dataset(par.val, cfg.voc, par.batch_size, par.max_src_len, par.max_tgt_len, do_shuffle=True, do_filter=True)
        Training(cfg, mod, opt, trn, val, chk)

    elif tst:
        sys.stderr.write('Inference...\n')
        cfg, mod, opt = chk.load(par.mod)
        tst = Dataset(par.tst, cfg.voc, par.batch_size, 0, 0, do_shuffle=False, do_filter=False)
        Inference(cfg, mod, tst)

if __name__ == "__main__":
    main()

