# -*- coding: utf-8 -*-

import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from config import Config
#from model import Model
#from data import Dataset, Vocab, Embed

class Training():

    def __init__(self, mod, trn, val, opt, voc, cfg, par, chk):
        ini_time = time.time()
        curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())    
        if opt is None: opt = Optimizer(cfg.method, cfg.max_grad_norm, par.lr, par.decay, mod)
        lr = par.lr
        # Move to GPU
        if cfg.cuda: 
            mod.cuda()
            opt.cuda()
        # loop over training batchs
        trn_loss_total = 0  # Reset every print_every
        trn_iter = 0
        for (src_batch, tgt_batch, ref_batch, raw_src_batch, raw_tgt_batch, len_src_batch, len_tgt_batch) in trn.minibatches():
            batch_size = src_batch.size(0)
            ### move to GPU
            if cfg.cuda:
                src_batch = src_batch.cuda()
                tgt_batch = tgt_batch.cuda()
                ref_batch = ref_batch.cuda()
            #######################
            # learn on trainset ###
            #######################
            dec_outputs, _ = mod(src_batch, tgt_batch, len_src_batch, len_tgt_batch) # forward
            loss = F.nll_loss(dec_outputs.permute(1,0,2).contiguous().view(-1, voc.length), ref_batch.contiguous().view(-1), ignore_index=voc.idx_pad) # Get loss
            trn_loss_total += loss.item()
            mod.zero_grad() # reset gradients
            loss.backward() # Backward propagation
            opt.step()
            mod.niters += 1
            trn_iter += 1
            if trn_iter % par.print_every == 0:
                curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
                sys.stdout.write('{} TRAIN iter:{} lr={:.5f} loss={:.4f}\n'.format(curr_time,trn_iter,lr,trn_loss_total/trn_iter))
            #######################
            # valid on validset ###
            #######################
            if trn_iter % par.valid_every == 0:
                val_loss_total = 0
                for val_iter, (src_batch, tgt_batch, ref_batch, raw_src_batch, raw_tgt_batch, len_src_batch, len_tgt_batch) in enumerate(val.minibatches()):
                    dec_outputs, _ = mod(src_batch, tgt_batch, len_src_batch, len_tgt_batch) ### forward
                    loss = F.nll_loss(dec_outputs.permute(1,0,2).contiguous().view(-1, voc.length), ref_batch.contiguous().view(-1), ignore_index=voc.idx_pad)
                    val_loss_total += loss.item()
                #update learning rate
                lr = opt.update_lr(val_loss_total)
                curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
                sys.stdout.write('{} VALID overall_iters:{} loss={:.4f}\n'.format(curr_time,mod.niters,val_loss_total/val_iter))
                chk.save(cfg, mod, opt, voc, val_loss_total/val_iter)

            if trn_iter >= par.n_iters: break

        curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
        seconds = "{:.2f}".format(time.time() - ini_time)
        sys.stdout.write('{} End of TRAIN seconds={}\n'.format(curr_time,seconds))

