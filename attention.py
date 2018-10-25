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
        S = enc_outputs.size(0)
        B = enc_outputs.size(1)
        # Create variable to store attention energies

        print_time('attn_energies before')
        # (B, 1, H) x (B, H, S) --> (B, 1, S)
        attn_energies = torch.bmm(dec_hidden.unsqueeze(1), enc_outputs.permute(1,2,0))
        attn_energies = attn_energies.squeeze(1) # [B, S]


#        attn_energies = Variable(torch.zeros(B, S)) # [B, S]
#        if self.cuda: attn_energies = attn_energies.cuda()
#        print_time('attn_energies before')
#        for b in range(B): #for all batches
#            for j in range(S): # Calculate energy for each enc_output (referred to each source word)
#                attn_energies[b, j] = self.score(dec_hidden[b], enc_outputs[j, b])
        print_time('attn_energies after')

        # Normalize energies to weights in range 0 to 1
        attn_energies_norm = F.softmax(attn_energies, dim=1)
        return attn_energies_norm #[B, S]

    def score(self, dec_hidden, enc_output):
        #print("dec_hidden={}".format(dec_hidden.shape)) #[1, H]
        #print("enc_output={}".format(enc_output.shape)) #[1, H]
        if self.method == 'dot':
            energy = dec_hidden.dot(enc_output)
        elif self.method == 'general':
            energy = self.attn(enc_output)
            energy = dec_hidden.dot(energy)
        elif self.method == 'concat':
            energy = self.attn(torch.cat((dec_hidden, enc_output), 1))
            energy = self.v.dot(energy)
        #print("energy={}".format(energy)) #value
        return energy
