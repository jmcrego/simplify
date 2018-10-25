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
from utils import print_time

class Training():

    def __init__(self, cfg, mod, opt, trn, val, chk):
        if cfg.n_iters_sofar is None: cfg.n_iters_sofar = 0
        ini_time = time.time()
        ###############################
        # loop over training batchs ###
        ###############################
#        curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())    
        sys.stdout.write('Start TRAIN')
        lr = cfg.par.lr
        trn_loss_total = 0  # Reset every print_every
        trn_iter = 0
        for (src_batch, tgt_batch, ref_batch, raw_src_batch, raw_tgt_batch, len_src_batch, len_tgt_batch) in trn.minibatches():
            if cfg.cuda:
                src_batch = src_batch.cuda()
                tgt_batch = tgt_batch.cuda()
                ref_batch = ref_batch.cuda()
            #######################
            # learn on trainset ###
            #######################
            dec_outputs, _ = mod(src_batch, tgt_batch, len_src_batch, len_tgt_batch) # forward
            loss = F.nll_loss(dec_outputs.permute(1,0,2).contiguous().view(-1, cfg.tvoc.size), ref_batch.contiguous().view(-1), ignore_index=cfg.tvoc.idx_pad) # Get loss
            trn_loss_total += loss.item()
            mod.zero_grad() # reset gradients
            loss.backward() # Backward propagation
            opt.step()
            cfg.n_iters_sofar += 1 
            trn_iter += 1
            if trn_iter > 0 and trn_iter % cfg.par.print_every == 0:
                print_time('TRAIN iter:{} lr={:.5f} loss={:.4f}'.format(cfg.n_iters_sofar,lr,trn_loss_total/trn_iter))
            ############################
            # validation on validset ###
            ############################
            if trn_iter > 0 and trn_iter % cfg.par.valid_every == 0:
                val_loss_total = 0
                val_iter = 0
                for (src_batch, tgt_batch, ref_batch, raw_src_batch, raw_tgt_batch, len_src_batch, len_tgt_batch) in val.minibatches():
                    if cfg.cuda:
                        src_batch = src_batch.cuda()
                        tgt_batch = tgt_batch.cuda()
                        ref_batch = ref_batch.cuda()
                    dec_outputs, _ = mod(src_batch, tgt_batch, len_src_batch, len_tgt_batch) ### forward
                    loss = F.nll_loss(dec_outputs.permute(1,0,2).contiguous().view(-1, cfg.tvoc.size), ref_batch.contiguous().view(-1), ignore_index=cfg.tvoc.idx_pad)
                    val_loss_total += loss.item()
                    val_iter += 1
                #update learning rate
                lr = opt.update_lr(val_loss_total)
                if val_iter > 0:
                    print_time('VALID iter:{} loss={:.4f}'.format(cfg.n_iters_sofar, val_loss_total/val_iter))
                    chk.save(cfg, mod, opt, val_loss_total/val_iter)
            ############################
            # end if reached n_iters ###
            ############################
            if trn_iter >= cfg.par.n_iters: break

        seconds = "{:.2f}".format(time.time() - ini_time)
        print_time('End of TRAIN seconds={}'.format(seconds))


