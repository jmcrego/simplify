#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import sys
import os
import time
from utils.utils import print_time
#reload(sys)
#sys.setdefaultencoding('utf8')


class Inference(): 

    def __init__(self, cfg, mod, tst):
        self.cfg = cfg
        ini_time = time.time()
        print_time('Start TEST')
        for val_iter, (src_batch, tgt_batch, ref_batch, raw_src_batch, raw_tgt_batch, len_src_batch) in enumerate(tst.minibatches()):
            if cfg.cuda:
                src_batch = src_batch.cuda()
                tgt_batch = tgt_batch.cuda()
            _, hyp_batch = mod(src_batch, tgt_batch, len_src_batch) ### forward
            self.display(hyp_batch, src_batch)
        print_time('End of TEST seconds={:.2f})\n'.format(time.time() - ini_time))


    def display(self, hyp_batch, src_batch):
#        assert(len(hyp_batch) == len(src_batch))
        for b in range(len(hyp_batch)):
            source, target = [], []
            for word_id in src_batch[b]: source.append("{}".format(self.cfg.svoc.get(int(word_id))))
            for word_id in hyp_batch[b]: target.append("{}".format(self.cfg.tvoc.get(int(word_id))))
            print("---{}---{}-------------------".format(len(src_batch),len(hyp_batch)))
            print(' '.join(source))
            print(' '.join(target))
