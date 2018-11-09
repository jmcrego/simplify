#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import sys
import os
import torch
import time
from utils.utils import print_time


class Inference(): 

    def __init__(self, cfg, mod, tst):
        self.cfg = cfg
        ini_time = time.time()
        print_time('Start TEST')
        with torch.no_grad():
            for val_iter, (src_batch, tgt_batch, ref_batch, raw_src_batch, raw_tgt_batch, len_src_batch, len_tgt_batch) in enumerate(tst.minibatches()):
                if cfg.cuda:
                    src_batch = src_batch.cuda()
                    tgt_batch = tgt_batch.cuda()

                _, hyp_batch = mod(src_batch, tgt_batch, len_src_batch, len_tgt_batch) ### forward      
                self.display(src_batch, ref_batch, hyp_batch)

        print_time('End of TEST seconds={:.2f})\n'.format(time.time() - ini_time))


    def display(self, src_batch, ref_batch, hyp_batch):
        hyp_batch = hyp_batch.permute(1,0)
        assert(len(hyp_batch) == len(src_batch))
        for b in range(len(hyp_batch)):
            source, target, predic = [], [], []
            print("--- SRC ------------------")
            for word_id in src_batch[b]: 
#                if word_id < 4: break
                source.append("{}".format(self.cfg.svoc.get(int(word_id))))
            print(' '.join(source))
            print("--- REF ------------------")
            for word_id in ref_batch[b]: 
#                if word_id < 4: break
                target.append("{}".format(self.cfg.tvoc.get(int(word_id))))
            print(' '.join(target))
            print("--- HYP ------------------")
            for word_id in hyp_batch[b]: 
#                if word_id < 4: break
                predic.append("{}".format(self.cfg.tvoc.get(int(word_id))))
            print(' '.join(predic))
