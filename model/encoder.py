# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

import numpy as np

###############################################################################
### EncoderRNN ################################################################
###############################################################################

class EncoderRNN(nn.Module):

    def __init__(self, embeddings, cfg): #rnn_type, num_layers, bidirectional, hidden_size, dropout):
        super(EncoderRNN, self).__init__()
        self.embeddings = embeddings # [V x E]
        self.D = 2 if cfg.bidirectional else 1
        ### if bidirectional the number of hidden_size (units) is divided by 2 for each direction
        if cfg.hidden_size % self.D != 0: sys.exit('error: hidden units {} must be an even number'.format(cfg.hidden_size)) 
        self.H = cfg.hidden_size // self.D #half the size for each direction
        self.L = cfg.num_layers
        self.E = self.embeddings.embedding_dim #embedding dimension
        self.V = self.embeddings.num_embeddings #vocabulary size
        ### rnn cell
        dropout = cfg.par.dropout if self.L>1 else 0.0 #dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1
        if cfg.cell == "lstm": self.rnn = nn.LSTM(input_size=self.E, hidden_size=self.H, num_layers=self.L, dropout=dropout, batch_first=True, bidirectional=(self.D==2))
        elif cfg.cell == "gru": self.rnn = nn.GRU(input_size=self.E, hidden_size=self.H, num_layers=self.L, dropout=dropout, batch_first=True, bidirectional=(self.D==2))
        else: sys.exit("error: bad -cell {} option. Use: lstm OR gru\n".format(cfg.cell))

    def forward(self, src_batch, len_src_batch):
        #print("src_batch={}".format(src_batch.shape)) #[B, S]
        #print("len_src_batch={}".format(len_src_batch.shape)) #[B]
        self.B = src_batch.size(0)
        self.S = src_batch.size(1)
        ### embed inputs
        input_emb = self.embeddings(src_batch) #[B,S,E]
        assert(input_emb.size() == (self.B,self.S,self.E))
        packed_emb = pack_padded_sequence(input_emb, len_src_batch, batch_first=True)
        ### rnn
        outputs, encoder_final = self.rnn(packed_emb)
        outputs, _ = pad_packed_sequence(outputs) ### unpack and take only the outputs, discard the sequence length
        assert(outputs.size() == (self.S,self.B,self.H*self.D)) 
        if isinstance(encoder_final, tuple):
            assert(encoder_final[0].size() == (self.L*self.D,self.B,self.H))
            assert(encoder_final[1].size() == (self.L*self.D,self.B,self.H))
        else: 
            assert(encoder_final.size() == (self.L*self.D,self.B,self.H))
        return outputs, encoder_final



