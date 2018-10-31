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
from utils.utils import print_time

class DecoderRNN_Attn(nn.Module):

    def __init__(self, embedding, cfg):
        super(DecoderRNN_Attn, self).__init__()
        ### embedding layer
        self.embedding = embedding # [voc_length x emb_size] contains nn.Embedding()
        self.V = self.embedding.num_embeddings #vocabulary size
        self.E = self.embedding.embedding_dim #embedding size
        self.L = cfg.num_layers
        self.D = 2 if cfg.bidirectional else 1
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

    def forward(self, tgt_batch, enc_final, enc_outputs, teacher_forcing):
        #print("forward")
        #tgt_batch is [B, S]
        self.S = enc_outputs.shape[0] #source seq_size
        self.T = tgt_batch.shape[1] #target seq_size
        self.B = tgt_batch.shape[0] #batch_size
        #print("decoder fwd B={} S={} H={}".format(self.B, self.S, self.H))

        tgt_batch = tgt_batch.transpose(1,0) #tgt_batch is [S, B]
        ### these are the output vectors that will be filled at the end of the loop
        dec_output_words = torch.zeros([self.T-1, self.B], dtype=torch.int64) #[T-1, B]
        dec_outputs = torch.zeros([self.T-1, self.B, self.V], dtype=torch.float32) #[T-1, B, V]
        if self.cuda: 
            dec_output_words = dec_output_words.cuda()
            dec_outputs = dec_outputs.cuda()
        ### initialize dec_hidden (with dec_final)
        dec_hidden = self.init_state(enc_final) #[L, B, D*dim] #dim is H/2
        ### initialize attn_hidden (Eq 5 in Luong) used for input-feeding
        attn_hidden = Variable(torch.zeros(1, self.B, self.H)) #[1, B, H]
        if self.cuda: 
            attn_hidden = attn_hidden.cuda()
        ### initialize coverage vector (Eq 10 in See)
        enc_coverage =  None
        if self.coverage:
            enc_coverage = torch.zeros([self.B, self.S], dtype=torch.float32) #[B, S]
            if self.cuda: enc_coverage = enc_coverage.cuda()

        for t in range(self.T-1): #loop to produce target words step by step
            ### decide which is the input word: consider teacher forcing
            if t==0: input_word = tgt_batch[t] ### it should be <ini>
            elif teacher_forcing < 1.0 and random.uniform() > teacher_forcing: input_word = dec_output_word #use t-1 predicted words
            else: input_word = tgt_batch[t] ### teacher forcing: the t-th words of each batch 
            ### this is the reference for the words to be predicted
            #target_word = tgt_batch[t + 1] 
            ### run forward step
            dec_output, dec_hidden, attn_hidden, dec_attn, enc_coverage = self.forward_step(input_word, attn_hidden, dec_hidden, enc_outputs, enc_coverage)
            #print("dec_output={}".format(dec_output.shape)) #[B,V]
            ### get the 1-best
            top_val, dec_output_word = dec_output.topk(1) #dec_output_word is [batch_size, 1] (the best entry of each batch)   
            dec_output_word = dec_output_word.squeeze(1)
            #print("dec_output_word={}".format(dec_output_word.shape)) #[B] (the index of the best entry for each batch)
            ### add to final output vectors
            dec_output_words[t] = dec_output_word #[t, B]
            dec_outputs[t] = dec_output #[t, B, V]

        #print("dec_outputs={}".format(dec_outputs.shape)) #[T-1,B,V]
        #print("dec_output_words={}".format(dec_output_words.shape)) #[T-1,B] (the index of the best entry for each batch)
        return dec_outputs, dec_output_words

    def forward_step(self, input_word, attn_hidden, dec_hidden, enc_outputs, enc_coverage):
        #print("input_word={}".format(input_word.shape))    #[B] previous target word 
        #print("attn_hidden={}".format(attn_hidden.shape))  #[1, B, H] previous attn_hidden
        #print("dec_hidden[0]={}".format(dec_hidden[0].shape)) #[L, B, H] previous dec_hidden 
        #print("dec_hidden[1]={}".format(dec_hidden[1].shape)) #[L, B, H] previous dec_hidden 
        #print("enc_outputs={}".format(enc_outputs.shape))  #[S, B, H] encoder outputs
        #if enc_coverage is not None: print("enc_coverage={}".format(enc_coverage.size()))  #[B, S] encoder coverage vector
        # get the embedding of the current input word (is the previous target word)
        input_emb = self.embedding(torch.tensor(input_word))
        input_emb = self.dropout(input_emb) #[B, E]
        input_emb = input_emb.unsqueeze(0) # [1, B, E]
        #input feeding: input_emb + attn_hidden
        input_emb_attn = torch.cat((input_emb, attn_hidden), 2) #[1, B, E+H]
        #rnn layer
        rnn_output, dec_hidden = self.rnn(input_emb_attn, dec_hidden)
        rnn_output = rnn_output.squeeze(0) # [1, B, H] -> [B, H]
        #print("rnn_output={}".format(rnn_output.shape)) #[B, H]
        #print("dec_hidden[0]={}".format(dec_hidden[0].shape)) #[L, B, H] (h)
        #print("dec_hidden[1]={}".format(dec_hidden[1].shape)) #[L, B, H] (c)

        # to calculate attention weights for each encoder output
        # we consider the last layer [-1] h state [0] (c is not used) of dec_hidden => dec_hidden[0][-1]
        # and all encoder outputs => dec_outputs
        # apply to encoder outputs to get weighted average
        laststate = dec_hidden[0][-1] #[B, H]
        align_weights = self.attn(laststate, enc_outputs, enc_coverage) # [B, S]
        #print("align_weights={}".format(align_weights.shape))

        ### accumulate coverage with align_weights
        if enc_coverage is not None:
            enc_coverage = enc_coverage + align_weights #[B,S]

        # context is the weighted (align_weights) average over all the source hidden states 
        align_weights = align_weights.unsqueeze(0) # [1, B, S]
        align_weights = align_weights.transpose(1,0) # [B, 1, S]
        enc_outputs = enc_outputs.transpose(1, 0) # [B, S, H]
        context = torch.bmm(align_weights, enc_outputs) # batched multiplication [B, 1, S] x [B, S, H] => [B, 1, H]
        context = context.squeeze(1)  #[B, H]
        #print('context={}'.format(context.shape))

        # concatenate together the current hidden state of the rnn and context and apply concat layer (Luong eq. 5)
        cat_laststate_context = torch.cat((laststate, context), 1) # [B, 2*H]
        #print('cat_rnnoutput_context={}'.format(cat_rnnoutput_context.shape)) 
        attn_hidden = torch.tanh(self.concat(cat_laststate_context)) #[B, H] applied concat layer 
        #print('attn_hidden={}'.format(attn_hidden.shape)) #[B, H]

        if self.pointer:
            pointer_input = torch.cat((context, rnn_output + input_emb.squeeze(0)), 1) #[B, 2*H+E]
            pgen = F.sigmoid(self.pgen(pointer_input))

        # aply output layer (Luong eq. 6)
        dec_output = self.output(attn_hidden) # [B, V]
        #print("dec_output={}".format(dec_output.shape))
        # apply softmax layer
        dec_output = F.log_softmax(dec_output, dim=1) #[B, V]
        #print("dec_output={}".format(dec_output.shape))

        if self.pointer:
            dec_output = pgen * dec_output #[B,V]
            attn_weights = (1-pgen) * attn_weights #[1,B,S]
            if extra_zeros is not None: dec_output = torch.cat([dec_output, extra_zeros], 1) #[B, V+extra]
            dec_output = dec_output.scatter_add(1, enc_batch_extend_vocab, attn_weights)

        attn_hidden = attn_hidden.unsqueeze(0) #[B, H] => [1, B, H] (vector used for input-feeding)
        #print("attn_hidden={}".format(attn_hidden.shape))

        return dec_output, dec_hidden, attn_hidden, align_weights, enc_coverage

    def init_state(self, encoder_hidden):
        if encoder_hidden is None: return None
        if isinstance(encoder_hidden, tuple): encoder_hidden = tuple([self.cat_directions(h) for h in encoder_hidden]) ### lstm
        else: encoder_hidden = self.cat_directions(encoder_hidden) ### gru
        return encoder_hidden

    def cat_directions(self, h):
        #if D is 1 (not a bidirectional encoder) there is nothing to do
        if self.D == 1: return h
        #otherwise, h is: [L*D, B, dim] and h should be: [L, B, D*dim] and dim is H/2
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h














###############################################################################
### DecoderRNN ################################################################
###############################################################################

class DecoderRNN(nn.Module):

    def __init__(self, embedding, rnn_type, num_layers, bidirectional_encoder, hidden_size, dropout):
        super(DecoderRNN, self).__init__()
        self.embedding = embedding # [voc_length x input_size] contains nn.Embedding()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.bidirectional_encoder = bidirectional_encoder
        self.hidden_size = hidden_size
        self.voc_size = self.embedding.num_embeddings #vocabulary size
        self.input_size = self.embedding.embedding_dim #embedding dimension
        ### set up the RNN
        if rnn_type == "lstm": self.rnn = nn.LSTM(self.input_size, hidden_size, num_layers, dropout=dropout) #StackedLSTM(num_layers, input_size, hidden_size, dropout)
        elif rnn_type == "gru": self.rnn = nn.GRU(self.input_size, hidden_size, num_layers, dropout=dropout) #StackedGRU(num_layers, input_size, hidden_size, dropout)
        else: sys.exit("error: bad -cell {} option. Use: lstm OR gru\n".format(rnn_type))
        ### projection layer
        self.project = nn.Linear(self.hidden_size, self.voc_size)
 
    def forward(self, tgt_batch, enc_final, enc_outputs, teacher_forcing):
        #tgt_batch is [batch_size, seq_size]
        batch_size = tgt_batch.shape[0]
        seq_size = tgt_batch.shape[1]
        tgt_batch = tgt_batch.transpose(1,0) 
        #tgt_batch is [seq_size, batch_size]
        #print("tgt_batch={}".format(tgt_batch.shape))
        dec_hidden = enc_final
        #dec_hidden is [num_layers*num_directions, batch_size, hidden_size/2], [num_layers*num_directions, batch_size, hidden_size/2] (the second element only when LSTM)
        #print("dec_hidden[0]={}".format(dec_hidden[0].shape)) 
        if self.bidirectional_encoder:
            h_size = int(self.hidden_size/2)
            # dec_hidden is [(layers*num_directions) x batch_size x dim] # We need to convert it to [layers x batch x (num_directions*dim)] #dim is hidden_size/2
            h = dec_hidden[0].view(self.num_layers, 2, batch_size, h_size) #2 is num_directions
            h.permute(0,2,1,3)
            h = h.view(self.num_layers, batch_size, -1)
            #print("h={}".format(h.shape))
            c = dec_hidden[1].view(self.num_layers, 2, batch_size, h_size)
            c.permute(0,2,1,3)
            c = c.view(self.num_layers, batch_size, -1)
            #print("c={}".format(c.shape))
            dec_hidden = tuple([h,c])
            #print("dec_hidden[0]={}".format(dec_hidden[0].shape))
            #print("dec_hidden[1]={}".format(dec_hidden[1].shape))

        dec_output_words = torch.zeros([seq_size - 1, batch_size], dtype=torch.int32)
        dec_outputs = torch.zeros([seq_size - 1, batch_size, self.voc_size], dtype=torch.float32)
        for t in range(seq_size - 1):
            ### teacher forcing
            if t==0: input_word = tgt_batch[t]
            elif teacher_forcing < 1.0 and random.uniform() > teacher_forcing: input_word = dec_output_word #use previous prediction
            else: input_word = tgt_batch[t] ### the t-th words of each batch (teacher_forcing)
            #print("input[t]={}".format(input_word))
            target_word = tgt_batch[t + 1] ### this is the reference for the words to be predicted
            #print("intput[t+1]={}".format(target_word))
            dec_output, dec_hidden = self.forward_step(input_word, dec_hidden)
            top_val, dec_output_word = dec_output.topk(1) #dec_output_word is [batch_size, 1] (the best entry of each batch)   
            dec_output_word = dec_output_word.squeeze(1)
            ### add to final vectors
            dec_output_words[t] = dec_output_word #[t x batch_size]
            dec_outputs[t] = dec_output #[t x batch_size x voc_size]

        return dec_outputs, dec_output_words

    def forward_step(self, input_word, dec_hidden):
        ### input [batch_size] is the word at instance t for all batches 
        #print("forward_step input_word={}".format(input_word.shape))
        # get the embedding of the current input word (last output word)
        batch_size = input_word.shape[0]
        input_emb = self.embedding(torch.tensor(input_word))
        input_emb = self.dropout(input_emb) #[batch_size, emb_size]
        #print("input_emb={}".format(input_emb.shape))
        input_emb = input_emb.unsqueeze(0) # [1, batch_size, emb_size]
        #print("input_emb={}".format(input_emb.shape)) 
        #print("dec_hidden[0]={}".format(dec_hidden[0].shape)) #[num_layers*num_directions, batch_size, hidden_size]
        #print("dec_hidden[1]={}".format(dec_hidden[1].shape)) #[num_layers*num_directions, batch_size, hidden_size]
        rnn_output, dec_hidden = self.rnn(input_emb, dec_hidden)
        #print("rnn_output={}".format(rnn_output[0].shape))
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        #print("rnn_output={}".format(rnn_output.shape))
        # Finally predict next token (Luong eq. 6)
        output = F.log_softmax(torch.tanh(self.project(rnn_output)), dim=1) #output is [batch_size, voc_size]        
        ### to get the 1-best word you must:
        ### top_val, output_word = output.topk(1) #output_word is [batch_size, 1] (the best entry of each batch)   
        ### output_word = output_word.squeeze(1)
        return output, dec_hidden



