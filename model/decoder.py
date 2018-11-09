# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
#from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from model.attention import Attention
from utils.data import idx_ini
from utils.utils import print_time, assert_size
#from model.beam_search import Beam

class DecoderRNN_Attn(nn.Module):

    def __init__(self, embedding, cfg):
        super(DecoderRNN_Attn, self).__init__()
        self.b = cfg.par.beam_size
        self.n = cfg.par.n_best
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

        self.tt = torch.cuda if self.cuda else torch        
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
        # tgt_batch [B,T]
        # len_src_batch [B]
        # len_tgt_batch [B]
        # enc_final ([L*D,B,H/D],[L*D,B,H/D]) or [L*D,B,H/D]
        # enc_outputs [S,B,H]
        self.S = enc_outputs.shape[0] #source seq_size
        self.T = tgt_batch.shape[1] #target seq_size
        self.B = tgt_batch.shape[0] #batch_size

        ### tgt_batch must be seq_len x batch
        tgt_batch = tgt_batch.transpose(1,0) # [T,B]
        ### initialize dec_hidden (with enc_final)
        rnn_hidden = self.init_state(enc_final) #([L,B,H], [L,B,H]) or [L,B,H]
        print("rnn_hidden={}".format(rnn_hidden[0].shape))
        sys.exit()
        ### initialize attn_hidden (Eq 5 in Luong) used for input-feeding
        attn_hidden = self.tt.FloatTensor(1, self.B, self.H).fill_(0.0)
        ### initialize coverage vector (Eq 10 in See)
        enc_coverage =  None
        if self.coverage: enc_coverage = self.tt.FloatTensor(self.B, self.S)
        ###
        ### loop
        ###
        ### these are the output vectors that will be filled at the end of the loop
        dec_output_words = self.tt.LongTensor(self.T-1, self.B)
        dec_outputs = self.tt.FloatTensor(self.T-1, self.B, self.V)
        for t in range(self.T-1): #loop to produce target words step by step
            ### current input/output words
            input_word = self.get_input_word(t, teacher_forcing, tgt_batch, dec_output_words) #[B]
            ### run forward step
            dec_output, rnn_hidden, attn_hidden, dec_attn, enc_coverage = self.forward_step(input_word, attn_hidden, rnn_hidden, enc_outputs, enc_coverage, len_src_batch)
            #dec_output   [B,V]
            #rnn_hidden   ([L,B,H],[L,B,H]) or [L,B,H]
            #attn_hidden  [1,B,H]
            #dec_attn     [B,1,S]
            #enc_coverage [B,S]
            ### get the 1-best
            dec_outputs[t] = dec_output
            dec_output_words[t] = self.get_one_best(dec_output) #[B]

        return dec_outputs, dec_output_words


    def forward_step(self, input_word, attn_hidden, rnn_hidden, enc_outputs, enc_coverage, len_src_batch):
        # input_word [B] previous target word 
        # attn_hidden [1,B,H] previous attn_hidden
        # rnn_hidden (h,c) = ([L,B,H], [L,B,H]) or h = [L,B,H] previous rnn_hidden
        # enc_outputs [S,B,H] 
        # enc_coverage [B,S]
        # len_src_batch [B]
        ### get the embedding of the current input word (is the previous target word)
        input_emb = self.embedding(torch.tensor(input_word)) #[B, E]
        input_emb = self.dropout(input_emb) #[B, E]
        input_emb = input_emb.unsqueeze(0) # [1, B, E]
        ### input feeding: input_emb + attn_hidden
        input_emb_attn = torch.cat((input_emb, attn_hidden), 2) #[1, B, E+H]
        ### rnn layer
        rnn_output, rnn_hidden = self.rnn(input_emb_attn, rnn_hidden)
        rnn_output = rnn_output.squeeze(0) # [1, B, H] -> [B, H]  # rnn_output is equal to rnn_hidden[0][-1] (last layer h state)
        align_weights = self.attn(rnn_output, enc_outputs, len_src_batch, enc_coverage) # [B, S] this is a_t(s) Equation 7 in Luong
        ### accumulate align_weights in coverage
        if enc_coverage is not None: enc_coverage = enc_coverage + align_weights #[B,S]
        ### context is the weighted (align_weights) average over all the source hidden states 
        align_weights = align_weights.unsqueeze(0).transpose(1,0) # [1, B, S] => [B, 1, S]
        enc_outputs = enc_outputs.transpose(1, 0) # [B, S, H]
        context = torch.bmm(align_weights, enc_outputs).squeeze(1) # batched multiplication [B, 1, S] x [B, S, H] => [B, 1, H] => [B,H]
        ### concatenate together the current hidden state of the rnn and context and apply concat layer and tanh (Luong eq. 5)
        attn_hidden = torch.tanh(self.concat( torch.cat((rnn_output, context), 1)) ) #tanh(concat([B,2*H])) ---> [B, H]
        ### pointer layer
        if self.pointer:
            pointer_input = torch.cat((context, rnn_output + input_emb.squeeze(0)), 1) #[B, 2*H+E]
            pgen = F.sigmoid(self.pgen(pointer_input))
        # output layer (Luong eq. 6)
        dec_output = self.output(attn_hidden) # [B, V]
        # softmax layer
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

    def get_input_word(self, t, teacher_forcing, tgt_batch, dec_output_words):
        if t==0: 
            input_word = tgt_batch[t] ### it should be <ini>
        elif teacher_forcing < 1.0 and random.uniform() > teacher_forcing: 
            input_word = dec_output_words[t-1] #use t-1 predicted words
        else: 
            input_word = tgt_batch[t] ### teacher forcing: the t-th words of each batch 
        return input_word


    def beam_search(self, cfg, len_src_batch, max_tgt_len, enc_final, enc_outputs, teacher_forcing):
        # len_src_batch [B]
        # enc_final ([L*D,B,H/D], [L*D,B,H/D]) or [L*D,B,H/D]
        # enc_outputs [S,B,H]
        self.S = enc_outputs.shape[0] #source seq_size
        self.B = enc_outputs.shape[1] #batch_size
        self.T = max_tgt_len

        ### initialize dec_hidden (with enc_final)
        rnn_hidden = self.init_state(enc_final) #([L,B,H], [L,B,H]) or [L,B,H]
        ### initialize attn_hidden (Eq 5 in Luong) used for input-feeding
        attn_hidden = torch.zeros(1, self.B, self.H) #[1, B, H]
        ### initialize coverage vector (Eq 10 in See)
        enc_coverage =  None
        if self.coverage:
            enc_coverage = torch.zeros([self.B, self.S], dtype=torch.float32) #[B, S]


        beams = [Beam(self.b, self.n, cuda=self.cuda) for _ in range(self.B)] #one beam per sentence in batch
        for t in range(self.T):
            if all(beam.done() for beam in beams): break #all beam finished in batches

            input_word = torch.stack([beam.get_current_state() for beam in beams]) # inp [B,b]            
            input_word = inp.t().contiguous().view(1, -1) #[b,B] => [1,b*B]

            # Run one step over the [b*B] input words
            dec_output, rnn_hidden, attn_hidden, align_weights, enc_coverage = self.forward_step(input_word, attn_hidden, rnn_hidden, enc_outputs, len_src_batch, enc_coverage)




            # Turn any copied words to UNKs. 0 is unk
            if self.copy_attn:
                inp = inp.masked_fill(inp.gt(len(self.fields["tgt"].vocab) - 1), 0)

            # Temporary kludge solution to handle changed dim expectation in the decoder
            inp = inp.unsqueeze(2)

            # Run one step.
            dec_out, attn = DecoderRNN_Attn(inp, memory_bank, memory_lengths=memory_lengths, step=i)

            dec_out = dec_out.squeeze(0)
            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch x beam word scores.
            if not self.copy_attn:
                out = self.model.generator.forward(dec_out).data
                out = unbottle(out)
                # beam x tgt_vocab
                beam_attn = unbottle(attn["std"])
            else:
                out = self.model.generator.forward(dec_out, attn["copy"].squeeze(0), src_map)
                # beam x (tgt_vocab + extra_vocab)
                out = data.collapse_copy_scores(unbottle(out.data), batch, self.fields["tgt"].vocab, data.src_vocabs)
                # beam x tgt_vocab
                out = out.log()
                beam_attn = unbottle(attn["copy"])

            # (c) Advance each beam.
            select_indices_array = []
            for j, b in enumerate(beam):
                b.advance(out[:, j], beam_attn.data[:, j, :memory_lengths[j]])
                select_indices_array.append(b.get_current_origin() * batch_size + j)
            select_indices = torch.cat(select_indices_array).view(batch_size, beam_size).transpose(0, 1).contiguous().view(-1)
            self.model.decoder.map_state(lambda state, dim: state.index_select(dim, select_indices))

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size
        if "tgt" in batch.__dict__: ret["gold_score"] = self._run_target(batch, data)
        ret["batch"] = batch

        return ret

