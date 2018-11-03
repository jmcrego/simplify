# -*- coding: utf-8 -*-

import sys
import os
import time
import torch

def print_time(desc, milli=False):
	if torch.cuda.is_available(): 
		torch.cuda.synchronize()
	if not milli: 
		curr_time = time.strftime("%Y-%m-%d_%X", time.localtime())
	else:
		curr_time = int(round(time.time() * 1000))
	sys.stdout.write('[{}] {}\n'.format(curr_time,desc))

def assert_size(t,size):
	assert(t.size() == size)

def assert_tuple_size(t,size):
	if isinstance(t, tuple):
		assert(t[0].size() == size)
		assert(t[1].size() == size)
	else:
		assert(t.size() == size)

def lens2mask(lengths, max_lengths=None):
	batch_size = lengths.numel()
	max_len = max_lengths or lengths.max()
	mask = torch.arange(0, max_len).type_as(lengths).repeat(batch_size, 1).lt(lengths.unsqueeze(1))
	return mask

