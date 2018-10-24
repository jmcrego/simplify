# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

from torch import optim
from encoder import EncoderRNN
from decoder import DecoderRNN, DecoderRNN_Attn

########################################################
### Model ##############################################
########################################################

class Model(nn.Module):

    def __init__(self, cfg, voc, emb, dropout=0.0):
        super(Model, self).__init__()
        self.cfg = cfg
        self.niters = 0
        self.best_loss = None

        self.embeddings = nn.Embedding(emb.matrix.shape[0], emb.matrix.shape[1]) ### input embeddings into a Model variable
        self.embeddings.weight.data.copy_(torch.from_numpy(emb.matrix))
        ### model
        self.encoder = EncoderRNN(self.embeddings, self.cfg, dropout)
        self.decoder = DecoderRNN_Attn(self.embeddings, self.cfg, dropout, voc.idx_ini, voc.idx_end, voc.idx_pad, voc.idx_unk)

#        sys.stderr.write('Initializing pars\n')
#        for param in self.encoder.parameters(): param.data.uniform_(-0.08, 0.08) #do not initialize the embeddings
#        for param in self.decoder.parameters(): param.data.uniform_(-0.08, 0.08)


    def forward(self, src_batch, tgt_batch, len_src_batch, len_tgt_batch, teacher_forcing=1.0):
        enc_outputs, enc_final = self.encoder(src_batch,len_src_batch)
        dec_outputs, dec_output_words = self.decoder(tgt_batch, enc_final, enc_outputs, teacher_forcing)
        return dec_outputs, dec_output_words
