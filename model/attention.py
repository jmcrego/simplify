# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.autograd import Variable
from utils import print_time


class Attention(nn.Module):

    def __init__(self, hidden_size, method, cuda):
        super(Attention, self).__init__()
        self.method = method
        self.cuda = cuda
        if method == 'dot': pass #no need to use a layer
        elif method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
            if self.cuda: self.attn.cuda()
        elif method == 'concat':
            self.attn = nn.Linear(hidden_size*2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))
            if self.cuda:
                self.attn.cuda()
                self.v.cuda()
        else: sys.exit("error: bad attention method {} option. Use: dot OR general OR concat\n".format(method))

    def forward(self, dec_hidden, enc_outputs):
        #print("dec_hidden={}".format(dec_hidden.shape)) #[B, H]
        #print("enc_outputs={}".format(enc_outputs.shape)) #[S, B, H]
        if self.method == 'dot':
            attn_energies = torch.bmm(dec_hidden.unsqueeze(1), enc_outputs.permute(1,2,0)) # (B, 1, H) x (B, H, S) --> (B, 1, S)
            attn_energies = attn_energies.squeeze(1) # [B, S]
        elif self.method == 'general':
            enc_output = self.attn(enc_output) #[S, B, H]
            attn_energies = torch.bmm(dec_hidden.unsqueeze(1), enc_outputs.permute(1,2,0)) # (B, 1, H) x (B, H, S) --> (B, 1, S)
            attn_energies = attn_energies.squeeze(1) # [B, S]
        elif self.method == 'concat':
            attn_energies = self.attn(torch.cat((dec_hidden, enc_output), 1))
            attn_energies = self.v.dot(both)

        # Normalize energies to weights in range 0 to 1
        attn_energies_norm = F.softmax(attn_energies, dim=1)
        return attn_energies_norm #[B, S]
