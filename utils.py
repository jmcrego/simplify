# -*- coding: utf-8 -*-

import sys
import os
import time
import torch

def print_time(desc, cuda=False, milli=False):
	if cuda: 
		torch.cuda.synchronize()
	if not milli: 
		curr_time = time.strftime("%Y-%m-%d_%X", time.localtime())
	else:
		curr_time = int(round(time.time() * 1000))
	sys.stdout.write('[{}] {}\n'.format(curr_time,desc))


