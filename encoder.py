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

    def __init__(self, embeddings, cfg, dropout): #rnn_type, num_layers, bidirectional, hidden_size, dropout):
        super(EncoderRNN, self).__init__()
        num_of_states = 2 if cfg.cell == "lstm" else 1
        num_directions = 2 if cfg.bidirectional else 1
        ### if bidirectional the number of hidden_size (units) is divided by 2 for each direction
        if cfg.hidden_size % num_directions != 0: sys.exit('error: hidden units {} must be an even number'.format(cfg.hidden_size)) 
        self.embeddings = embeddings # [voc_length x input_size]
        self.num_layers = cfg.num_layers
        self.hidden_size = cfg.hidden_size // num_directions
        self.input_size = self.embeddings.embedding_dim #embedding dimension
        ### rnn cell
        if cfg.cell == "lstm": self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=dropout, batch_first=True, bidirectional=(num_directions==2))
        elif cfg.cell == "gru": self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=dropout, batch_first=True, bidirectional=(num_directions==2))
        else: sys.exit("error: bad -cell {} option. Use: lstm OR gru\n".format(cfg.cell))
        ### bridge (not used)
        #self.total_hidden_dim = hidden_size * num_layers
        #self.bridgelayers = nn.ModuleList([nn.Linear(self.total_hidden_dim, self.total_hidden_dim, bias=True) for _ in range(num_of_states)]) ### there will be 1:GRU or 2:LSTM Linear layers

    def forward(self, src_batch, lengths):
        #src_batch: [batch_size, seq_len]
        #lengths: [batch_size]
        #print("src_batch={}".format(src_batch.shape)) 
        #print("lengths={}".format(lengths.shape)) 
        ###
        ### embed inputs
        ###
        input_emb = self.embeddings(src_batch)
        #input_emb [batch_size, len_seq, emb_size]
        #print("input_emb={}".format(input_emb.shape)) 
        packed_emb = pack_padded_sequence(input_emb, lengths, batch_first=True)
        ###
        ### rnn
        ###
        outputs, encoder_final = self.rnn(packed_emb)
        #encoder_final = (h,c) = ([num_layers*num_directions, batch_size, hidden_size], [num_layers*num_directions, batch_size, hidden_size])
        #or                  h = [num_layers*num_directions, batch_size, hidden_size]      
        #print("encoder_final = ({}, {})".format(encoder_final[0].shape, encoder_final[1].shape))
        outputs, _ = pad_packed_sequence(outputs) ### unpack and take only the outputs, discard the sequence length
        #print("outputs={}".format(outputs.shape)) #outputs [len_seq, batch_size, hidden_size]
        #if isinstance(encoder_final, tuple): 
        #    print("encoder_final[0]={}".format(encoder_final[0].shape)) #[num_layers*num_directions, batch_size, hidden_size/2]
        #    print("encoder_final[1]={}".format(encoder_final[1].shape)) #[num_layers*num_directions, batch_size, hidden_size/2]
        #else: print("encoder_final={}".format(encoder_final.shape)) #[num_layers*num_directions, batch_size, hidden_size/2]

        return outputs, encoder_final ### outputs [len_seq, batch_size, hidden_size], encoder_final_states [num_layers*num_directions, batch_size, hidden_size/num_directions]

        #### next is bridge (not used)

        ### function to forward bridges
        def fwd_bridge(linear, states):
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)
        ###
        ### bridge for the final hidden states
        ###
        if isinstance(encoder_final, tuple): 
            ### LSTM
            h = fwd_bridge(self.bridgelayers[0], encoder_final[0])
            c = fwd_bridge(self.bridgelayers[1], encoder_final[1])
            encoder_final = tuple([h,c]) # ([num_layers*num_directions, batch_size, hidden_size], [num_layers*num_directions, batch_size, hidden_size/2])
            #print("encoder_final = h, c = ({}, {})".format(encoder_final[0].shape, encoder_final[1].shape)) 
        else:  
            ### GRU
            encoder_final = fwd_bridge(self.bridgelayers[0], encoder_final) # h = [num_layers*num_directions, batch_size, hidden_size]
            #print("encoder_final = h = {}".format(encoder_final.shape))

        return outputs, encoder_final ### outputs [len_seq, batch_size, hidden_size], encoder_final_states [num_layers*num_directions, batch_size, hidden_size/num_directions]

