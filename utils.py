# -*- coding: utf-8 -*-

import sys
import os
import time
import torch

def print_time(desc):
	torch.cuda.synchronize()
	curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
	sys.stdout.write('{} {}\n'.format(curr_time,desc))
   	#curr_milli = int(round(time.time() * 1000))
   	#sys.stdout.write('[{}] {}\n'.format(curr_milli,desc))

