# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from model.attention import Attention
from utils.data import idx_ini
from utils.utils import print_time, assert_size

class DecoderRNN_Attn(nn.Module):

    def __init__(self, embedding, cfg):
        super(DecoderRNN_Attn, self).__init__()
        ### embedding layer
        self.embedding = embedding # [voc_length x emb_size] contains nn.Embedding()
        self.V = self.embedding.num_embeddings #vocabulary size
        self.E = self.embedding.embedding_dim #embedding size
        self.L = cfg.num_layers
        self.D = 2 if cfg.bidirectional else 1 ### num of directions
        self.H = cfg.hidden_size 
        self.cuda = cfg.cuda
        self.pointer = cfg.pointer
        self.coverage = cfg.coverage
        ### dropout layer to apply on top of the embedding layer
        self.dropout = nn.Dropout(cfg.par.dropout)
        ### set up the RNN
        dropout = cfg.par.dropout if self.L>1 else 0.0 #dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1
        if cfg.cell == "lstm": self.rnn = nn.LSTM(self.E+self.H, self.H, self.L, dropout=dropout) #input is embedding+hidden (to allow feed-input)
        elif cfg.cell == "gru": self.rnn = nn.GRU(self.E+self.H, self.H, self.L, dropout=dropout)
        else: sys.exit("error: bad -cell {} option. Use: lstm OR gru\n".format(cfg.cell))
        ### Attention mechanism
        self.attn = Attention(self.H, cfg.attention, cfg.coverage, cfg.cuda)
        ### pgen layer
        if self.pointer : self.pgen = nn.Linear(self.H*2+self.E, 1)
        ### concat layer
        self.concat = nn.Linear(self.H*2, self.H) 
        ### output layer
        self.output = nn.Linear(self.H, self.V)

    def forward(self, tgt_batch, len_src_batch, len_tgt_batch, enc_final, enc_outputs, teacher_forcing):
        # tgt_batch [B, T]
        self.S = enc_outputs.shape[0] #source seq_size
        self.T = tgt_batch.shape[1] #target seq_size
        self.B = tgt_batch.shape[0] #batch_size
        assert(len(tgt_batch) == len(len_tgt_batch))
        if isinstance(enc_final, tuple):
            assert(enc_final[0].size() == (self.L*self.D,self.B,self.H/self.D))
            assert(enc_final[1].size() == (self.L*self.D,self.B,self.H/self.D))
        else: 
            assert(enc_final.size() == (self.L*self.D,self.B,self.H/self.D))
        assert(enc_outputs.size() == (self.S,self.B,self.H)) 

        ### these are the output vectors that will be filled at the end of the loop
        dec_output_words = torch.zeros([self.T-1, self.B], dtype=torch.int64) #[T-1, B]
        dec_outputs = torch.zeros([self.T-1, self.B, self.V], dtype=torch.float32) #[T-1, B, V]
        if self.cuda: 
            dec_output_words = dec_output_words.cuda()
            dec_outputs = dec_outputs.cuda()
        ### tgt_batch must be seq_len x batch
        tgt_batch = tgt_batch.transpose(1,0) # [T,B]
        ### initialize dec_hidden (with enc_final)
        rnn_hidden = self.init_state(enc_final) #([L,B,H], [L,B,H])
        if isinstance(rnn_hidden, tuple):
            assert(rnn_hidden[0].size() == (self.L,self.B,self.H))
            assert(rnn_hidden[1].size() == (self.L,self.B,self.H))
        else: 
            assert(rnn_hidden.size() == (self.L,self.B,self.H))
        ### initialize attn_hidden (Eq 5 in Luong) used for input-feeding
        attn_hidden = Variable(torch.zeros(1, self.B, self.H)) #[1, B, H]
        if self.cuda: 
            attn_hidden = attn_hidden.cuda()
        ### initialize coverage vector (Eq 10 in See)
        enc_coverage =  None
        if self.coverage:
            enc_coverage = torch.zeros([self.B, self.S], dtype=torch.float32) #[B, S]
            if self.cuda: enc_coverage = enc_coverage.cuda()
        ###
        ### loop
        ###
        dec_output_word = None
        for t in range(self.T-1): #loop to produce target words step by step
            ### current input/output words
            input_word = self.get_input_word(t, teacher_forcing, tgt_batch, dec_output_word) #[B]
            ### run forward step
            dec_output, rnn_hidden, attn_hidden, dec_attn, enc_coverage = self.forward_step(input_word, attn_hidden, rnn_hidden, enc_outputs, len_src_batch, enc_coverage)
            #dec_output [B,V]
            #nrr_hidden ([L,B,H],[L,B,H]) or [L,B,H]
            #attn_hidden [1,B,H]
            #dec_attn [B,1,S]
            #enc_coverage [B,S]
            ### get the 1-best
            dec_output_word = self.get_one_best(dec_output) #[B]
            ### update final output vectors
            dec_output_words[t] = dec_output_word 
            dec_outputs[t] = dec_output 

        assert(dec_output_words.size() == (self.T-1, self.B))
        assert(dec_outputs.size() == (self.T-1, self.B, self.V))
        #print("dec_outputs={}".format(dec_outputs.shape)) #[T-1,B,V]
        #print("dec_output_words={}".format(dec_output_words.shape)) #[T-1,B] (the index of the best entry for each batch)
        return dec_outputs, dec_output_words

    def forward_step(self, input_word, attn_hidden, rnn_hidden, enc_outputs, len_src_batch, enc_coverage):
        # input_word [B] previous target word 
        # attn_hidden [1,B,H] previous attn_hidden
        # rnn_hidden (h,c) = ([L,B,H], [L,B,H]) or h = [L,B,H]
        # enc_outputs [S,B,H] 
        # enc_coverage [B,S]
        ### get the embedding of the current input word (is the previous target word)
        input_emb = self.embedding(torch.tensor(input_word)) #[B, E]
        input_emb = self.dropout(input_emb) #[B, E]
        input_emb = input_emb.unsqueeze(0) # [1, B, E]
        ### input feeding: input_emb + attn_hidden
        input_emb_attn = torch.cat((input_emb, attn_hidden), 2) #[1, B, E+H]
        ### rnn layer
        rnn_output, rnn_hidden = self.rnn(input_emb_attn, rnn_hidden) 
        rnn_output = rnn_output.squeeze(0) # [1, B, H] -> [B, H]
        # rnn_output is equal to rnn_hidden[0][-1] (last layer h state)
        # to calculate attention weights for each encoder output
        # we consider the last layer h state (or rnn_output) and all encoder outputs 
        align_weights = self.attn(rnn_output, enc_outputs, len_src_batch, enc_coverage) # [B, S] this is a_t(s) Equation 7 in Luong
        ### accumulate align_weights in coverage
        if enc_coverage is not None:
            enc_coverage = enc_coverage + align_weights #[B,S]
        ### context is the weighted (align_weights) average over all the source hidden states 
        align_weights = align_weights.unsqueeze(0) # [1, B, S]
        align_weights = align_weights.transpose(1,0) # [B, 1, S]
        enc_outputs = enc_outputs.transpose(1, 0) # [B, S, H]
        context = torch.bmm(align_weights, enc_outputs) # batched multiplication [B, 1, S] x [B, S, H] => [B, 1, H]
        context = context.squeeze(1)  #[B, H]
        ### concatenate together the current hidden state of the rnn and context and apply concat layer and tanh (Luong eq. 5)
        cat_rnnoutput_context = torch.cat((rnn_output, context), 1) # [B, 2*H]
        attn_hidden = torch.tanh(self.concat(cat_rnnoutput_context)) #[B, H]

        if self.pointer:
            pointer_input = torch.cat((context, rnn_output + input_emb.squeeze(0)), 1) #[B, 2*H+E]
            pgen = F.sigmoid(self.pgen(pointer_input))

        # aply output layer (Luong eq. 6)
        dec_output = self.output(attn_hidden) # [B, V]
        # apply softmax layer
        dec_output = F.log_softmax(dec_output, dim=1) #[B, V]

        if self.pointer:
            dec_output = pgen * dec_output #[B,V]
            attn_weights = (1-pgen) * attn_weights #[1,B,S]
            if extra_zeros is not None: dec_output = torch.cat([dec_output, extra_zeros], 1) #[B, V+extra]
            dec_output = dec_output.scatter_add(1, enc_batch_extend_vocab, attn_weights)

        attn_hidden = attn_hidden.unsqueeze(0) #[B, H] => [1, B, H] (vector used for input-feeding)

        return dec_output, rnn_hidden, attn_hidden, align_weights, enc_coverage

    def init_state(self, encoder_hidden):
        if encoder_hidden is None: return None
        if isinstance(encoder_hidden, tuple): encoder_hidden = tuple([self.cat_directions(h) for h in encoder_hidden]) ### lstm
        else: encoder_hidden = self.cat_directions(encoder_hidden) ### gru
        return encoder_hidden

    def cat_directions(self, h):
        #if D is 1 (a UNIdirectional encoder) there is nothing to do
        if self.D == 1: return h
        #otherwise, h is: [L*D, B, dim] and h should be: [L, B, D*dim] and dim is H/2
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def get_one_best(self, dec_output):
        top_val, dec_output_word = dec_output.topk(1) #dec_output_word is [batch_size, 1] (the best entry of each batch)   
        dec_output_word = dec_output_word.squeeze(1)
        return dec_output_word

    def get_input_word(self, t, teacher_forcing, tgt_batch, dec_output_word=None):
        if t==0: 
            input_word = tgt_batch[t] ### it should be <ini>
        elif teacher_forcing < 1.0 and random.uniform() > teacher_forcing: 
            input_word = dec_output_word #use t-1 predicted words
        else: 
            input_word = tgt_batch[t] ### teacher forcing: the t-th words of each batch 
        return input_word



