# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

from torch import optim
from encoder import EncoderRNN
from decoder import DecoderRNN, DecoderRNN_Attn
from utils import print_time

########################################################
### Model ##############################################
########################################################

class Model(nn.Module):

    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg

        self.embeddings_src = nn.Embedding(self.cfg.svoc.size, self.cfg.emb_src_size) ### embeddings for encoder
        self.encoder = EncoderRNN(self.embeddings_src, self.cfg)

        if self.cfg.reuse_words: self.embeddings_tgt = self.embeddings_src ### same embeddings for encoder and decoder
        else: self.embeddings_tgt = nn.Embedding(self.cfg.tvoc.size, self.cfg.emb_tgt_size) ### new embeddings for decoder
        self.decoder = DecoderRNN_Attn(self.embeddings_tgt, self.cfg) 

        sys.stderr.write('Initializing model pars\n')
        for param in self.encoder.parameters(): param.data.uniform_(-0.08, 0.08)
        for param in self.decoder.parameters(): param.data.uniform_(-0.08, 0.08)

    def forward(self, src_batch, tgt_batch, len_src_batch, teacher_forcing=1.0):
        enc_outputs, enc_final = self.encoder(src_batch,len_src_batch)
        dec_outputs, dec_output_words = self.decoder(tgt_batch, enc_final, enc_outputs, teacher_forcing)
        return dec_outputs, dec_output_words
