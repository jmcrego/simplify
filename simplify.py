#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import sys
from config import Config, Params
from model import Model
from data import Dataset, Vocab, Embed
from training import Training
from inference import Inference
from checkpoint import Checkpoint

def main():

    par = Params(sys.argv)
    chk = Checkpoint(par.dir)

    if par.trn and par.val:
        if chk.contains_model:
            cfg, mod, opt, voc = chk.load()            
            sys.stderr.write('Learning [resume It={}]...\n'.format(mod.niters))
        else:
            cfg = Config(par.cfg)
            voc = Vocab(par.voc)
            emb = Embed(par.emb, cfg, voc) #embeddings read from file OR randomly initialised (length of vectors must match with that given in config file)
            mod = Model(cfg, voc, emb, dropout=par.dropout)
            opt = None
            sys.stderr.write('Learning [from scratch]...\n')

        trn = Dataset(par.trn, voc, par.batch_size, par.max_src_len, par.max_tgt_len, do_shuffle=True, do_filter=True)
        val = Dataset(par.val, voc, par.batch_size, par.max_src_len, par.max_tgt_len, do_shuffle=True, do_filter=True)
        Training(mod, trn, val, opt, voc, cfg, par, chk)

    elif tst:
        sys.stderr.write('Inference...\n')
        cfg, mod, opt, voc = chk.load(par.mod)
        tst = Dataset(par.tst, voc, par.batch_size, 0, 0, do_shuffle=False, do_filter=False)
        Inference(mod, tst, voc, cfg, par)

if __name__ == "__main__":
    main()

