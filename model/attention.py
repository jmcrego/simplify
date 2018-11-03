# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.autograd import Variable
from utils.utils import print_time, lens2mask


class Attention(nn.Module):

    def __init__(self, hidden_size, method, coverage, cuda):
        super(Attention, self).__init__()
        self.method = method
        self.cuda = cuda

        if method == 'dot': 
            pass #no need to use a layer

        elif method == 'general':
            self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)
            if cuda: self.W_a.cuda()

        elif method == 'concat':
            self.W_a_h = nn.Linear(hidden_size, hidden_size, bias=False) #linear layer for enc hidden states (enc_output) 'context'
            self.W_a_s = nn.Linear(hidden_size, hidden_size, bias=True)  #linear layer for dec_state (dec_hidden) 'query' (contains bias: b_attn)
            self.v = nn.Linear(hidden_size, 1, bias=False) 
            if coverage: self.W_a_c = nn.Linear(1, hidden_size, bias=False) #lineary layer for coverage vector
            if cuda:
                self.W_a_h.cuda()
                self.W_a_s.cuda()
                self.v.cuda()
                if coverage: self.W_a_c.cuda()

        else: sys.exit("error: bad attention method {} option. Use: dot OR general OR concat\n".format(method))


    def forward(self, query, context, len_src_batch, coverage=None):
        #query [B, H] (dec_output, h_t in Luong)
        #context [S, B, H] (enc_hidden, \hat{h}_s in Luong)
        #coverage [B, S] sum over the previous attention vectors
        self.B = query.size(0)
        self.H = query.size(1)
        self.S = context.size(0)

        if self.method in ["general", "dot"]:
            if self.method == "general": 
                query = self.W_a(query) #[B,H]
            query = query.view(self.B, 1, self.H)  #[B,1,H]
            context = context.permute(1,2,0) #[B,H,S]
            scores = torch.bmm(query, context).squeeze(1) # [B,1,H] x [B,H,S] --> [B,1,S] --> [B,S]

        else: 
            ### context
            context = context.permute(1,0,2) #[B,S,H]
            context = context.contiguous().view(-1, self.H) #[B*S,H]
            context = self.W_a_h(context) 
            ### query
            query = self.W_a_s(query) #query [B, H]
            query = query.unsqueeze(1).expand(self.B, self.S, self.H) #[B, 1, H] => [B, S, H]
            query = query.contiguous().view(-1, self.H) # [B*S,H]
            scores = context + query # [B*S,H]
            ### coverage
            if coverage is not None:
                coverage = coverage.view(-1, 1) # [B*S,1] 
                coverage = self.W_a_c(coverage) # [B*S,H]
                scores = scores + coverage # [B*S,H]
            scores = torch.tanh(scores) # [B*S,H]
            scores = self.v(scores)  # [B*S,1]
            scores = scores.view(-1, self.S)  # [B,S]

        mask = lens2mask(len_src_batch, self.S) #[B,S]
        if self.cuda:
            mask = mask.cuda()
#        mask = mask.unsqueeze(1)  #[B,1,S]
        scores.masked_fill_(1-mask, -float('inf')) ### Fills elements of scores with -inf where 1-mask is one (padded words)

        #normalize scores
        align = F.softmax(scores, dim=1) #[B, S]
        return align

