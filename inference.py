#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import sys
import os
import time
from utils import print_time
#from config import Config
#from model import Model
#from data import Dataset, Vocab, Embed
reload(sys)
sys.setdefaultencoding('utf8')


class Inference(): 

    def __init__(self, cfg, mod, tst):
        self.cfg = cfg
        ini_time = time.time()
        print_time('Start TEST')
        for val_iter, (src_batch, tgt_batch, ref_batch, raw_src_batch, raw_tgt_batch, len_src_batch) in enumerate(tst.minibatches()):
            if cfg.cuda:
                src_batch = src_batch.cuda()
                tgt_batch = tgt_batch.cuda()
            _, predict_batch = mod(src_batch, tgt_batch, len_src_batch) ### forward
            self.display(predict_batch, src_batch)

        seconds = "{:.2f}".format(time.time() - ini_time)
        print_time('End of TEST seconds={})\n'.format(seconds))


    def display(self, predict_batch, src_batch):
        for b in range(len(predict_batch)):
            sentence = []
            for word_id in predict_batch[b]:
                word = self.cfg.tvoc.get(int(word_id))
                sentence.append("{}".format(word))
            print(' '.join(sentence))
            print(src_batch[b])
