# -*- coding: utf-8 -*-

import sys
import os
import time

def print_time(desc):
    curr_time = time.strftime("[%Y-%m-%d_%X]", time.localtime())
    sys.stdout.write('{} {}\n'.format(curr_time,desc))

