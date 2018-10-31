# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.autograd import Variable
from utils.utils import print_time


class Attention(nn.Module):

    def __init__(self, hidden_size, method, coverage, cuda):
        super(Attention, self).__init__()
        self.method = method

        if method == 'dot': 
            pass #no need to use a layer

        elif method == 'general':
            self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)
            if cuda: self.W_a.cuda()

        elif method == 'concat':
            self.W_h = nn.Linear(hidden_size, hidden_size, bias=False) #linear layer for enc hidden states (enc_output) 'context'
            self.W_s = nn.Linear(hidden_size, hidden_size, bias=True)  #linear layer for dec_state (dec_hidden) 'query' (contains bias: b_attn)
            self.v = nn.Linear(hidden_size, 1, bias=False) 
            if coverage: self.W_c = nn.Linear(1, hidden_size, bias=False) #lineary layer for coverage vector
            if cuda:
                self.W_h.cuda()
                self.W_s.cuda()
                self.v.cuda()
                if coverage: self.W_c.cuda()

        else: sys.exit("error: bad attention method {} option. Use: dot OR general OR concat\n".format(method))


    def forward(self, query, context, coverage=None):
        #print("query={}".format(query.shape)) #dec_output [B, H]  
        #print("context={}".format(context.shape)) #enc_hidden [S, B, H] 
        #if coverage is not None: print("coverage={}".format(coverage.shape)) # [B, S]
        B = query.size(0)
        H = query.size(1)
        S = context.size(0)
        #print("B={} S={} H={}".format(B,S,H))

        if self.method in ["general", "dot"]:
            if self.method == "general": h_t = self.W_a(query) #[B,H]
            query = query.view(B, 1, H)  #[B,1,H]
            context = context.permute(1,2,0) #[B,H,S]
            scores = torch.bmm(query, context).squeeze(1) # [B,1,H] x [B,H,S] --> [B,1,S] --> [B,S]

        else: 
            ### context
            context = context.permute(1,0,2) #[B,S,H]
            context = context.contiguous().view(-1, H) #[B*S,H]
            context = self.W_h(context) 
            ### query
            query = self.W_s(query) #query [B, H]
            query = query.unsqueeze(1).expand(B, S, H) #[B, 1, H] => [B, S, H]
            query = query.contiguous().view(-1, H) # [B*S,H]
            scores = context + query # [B*S,H]
            ### coverage
            if coverage is not None:
                coverage = coverage.view(-1, 1) # [B*S,1] enc_coverage is None or the sum over the previous attention vectors [B,S]
                coverage = self.W_c(coverage) # [B*S,H]
                scores = scores + coverage # [B*S,H]
            scores = torch.tanh(scores) # [B*S,H]
            scores = self.v(scores)  # [B*S,1]
            scores = scores.view(-1, S)  # [B, S]

        return scores

