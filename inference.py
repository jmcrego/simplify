#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import sys
import os
import time
#from config import Config
#from model import Model
#from data import Dataset, Vocab, Embed

class Inference(): 

    def __init__(self, cfg, mod, tst):
        ini_time = time.time()
        for val_iter, (src_batch, tgt_batch, target_batch, raw_src_batch, raw_tgt_batch, len_src_batch, len_tgt_batch) in enumerate(tst.minibatches()):
            _, predict_batch = mod(src_batch, tgt_batch, len_src_batch, len_tgt_batch) ### forward
            self.display(mod, predict_batch)

        curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
        seconds = "{:.2f}".format(time.time() - ini_time)
        sys.stdout.write('{} End of TEST seconds={})\n'.format(curr_time,seconds))


    def display(self, mod, predict_batch):

        for predict_sentence in predict_batch:
            for word_id in predict:
                word = mod.voc.get(int(word_id))
                sentence.append("{}".format(word))
            print(sentence)
