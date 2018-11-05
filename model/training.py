# -*- coding: utf-8 -*-

import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.optim import Optimizer
from utils.utils import print_time
from utils.bleu import *

class Training():

    def __init__(self, cfg, mod, opt, trn, val, chk):
        if cfg.n_iters_sofar is None: cfg.n_iters_sofar = 0
        ini_time = time.time()
        print_time('Start TRAIN')
        lr = cfg.par.lr
        loss_total_N_iters = 0  # Reset every [print_every]
        Iter = 0
        for (src_batch, tgt_batch, ref_batch, raw_src_batch, raw_tgt_batch, len_src_batch, len_tgt_batch) in trn.minibatches():
            assert(len(src_batch)==len(tgt_batch)==len(ref_batch)==len(ref_batch)==len(raw_src_batch)==len(raw_tgt_batch)==len(len_src_batch))
            if cfg.cuda:
                src_batch = src_batch.cuda()
                tgt_batch = tgt_batch.cuda()
                ref_batch = ref_batch.cuda()
            dec_outputs, dec_output_words = mod(src_batch, tgt_batch, len_src_batch, len_tgt_batch) # forward returns: [S,B,V] [S,B]
            dec_outputs = dec_outputs.permute(1,0,2).contiguous().view(-1, cfg.tvoc.size)
            ref_batch = ref_batch.contiguous().view(-1)
            loss = F.nll_loss(dec_outputs, ref_batch, ignore_index=cfg.tvoc.idx_pad) #loss normalized by word
            loss_total_N_iters += loss.item() 
            mod.zero_grad() # reset gradients
            loss.backward() # Backward propagation
            opt.step()
            cfg.n_iters_sofar += 1 
            Iter += 1
            if Iter % cfg.par.print_every == 0: 
                print_time('TRAIN iter:{} lr={:.5f} loss={:.4f}'.format(cfg.n_iters_sofar,lr,loss_total_N_iters/cfg.par.print_every))
                loss_total_N_iters = 0
            if Iter % cfg.par.valid_every == 0: 
                lr = self.validation(cfg, mod, opt, val, chk)
            if Iter >= cfg.par.n_iters: 
                break

        print_time('End of TRAIN seconds={:.2f}'.format(time.time() - ini_time))


    def validation(self, cfg, mod, opt, val, chk):
        print_time('Start VALID')
        hyp_data = []
        ref_data = []
        loss_total = 0
        Iter = 0
        for (src_batch, tgt_batch, ref_batch, raw_src_batch, raw_tgt_batch, len_src_batch, len_tgt_batch) in val.minibatches():
            if cfg.cuda:
                src_batch = src_batch.cuda()
                tgt_batch = tgt_batch.cuda()
                ref_batch = ref_batch.cuda()
#            print("tgt_batch={}".format(tgt_batch.shape))
            dec_outputs, dec_output_words = mod(src_batch, tgt_batch, len_src_batch, len_tgt_batch) ### forward  returns: [T,B,V] [T,B]
            loss = F.nll_loss(dec_outputs.permute(1,0,2).contiguous().view(-1, cfg.tvoc.size), ref_batch.contiguous().view(-1), ignore_index=cfg.tvoc.idx_pad) #loss normalized by word
            loss_total += loss.item()
            Iter += 1
            hyp_data.extend(dec_output_words.permute(1,0))
            ref_data.extend(ref_batch)
        #update learning rate
        lr = opt.update_lr(loss_total)
        if Iter > 0:
            print_time('VALID iter:{} loss={:.4f} bleu={:.2f}'.format(cfg.n_iters_sofar, loss_total/Iter, self.get_bleu(hyp_data,ref_data,cfg)))
            chk.save(cfg, mod, opt, loss_total/Iter)
        return lr

    def get_bleu(self, hyp_data, ref_data, cfg):
        assert(len(hyp_data) == len(ref_data))

        hyp_str = []
        for sent in hyp_data: 
            sent_str = []
            for wrdid in sent:
#                if (wrdid < 4): break
                sent_str.append(cfg.tvoc.get(int(wrdid)))
            hyp_str.append(' '.join(sent_str))

        ref_str = []
        for sent in ref_data: 
            sent_str = []
            for wrdid in sent:
                if (wrdid < 4): break
                sent_str.append(cfg.tvoc.get(int(wrdid)))
            ref_str.append(' '.join(sent_str))

        bleu, addition = corpus_bleu(hyp_str, ref_str) #print(bleu[0]*100)
        return bleu[0]*100


