# -*- coding: utf-8 -*-

import yaml
import sys

class Config():

    def __init__(self, file):
        self.cuda = False
        self.cell = None
        self.num_layers = None
        self.bidirectional = None
        self.hidden_size = None
        self.emb_size = None
        self.attention = 'dot'
        self.coverage = None
        self.pointer = None
        self.method = None         
        self.max_grad_norm = None
        self.voc_size = None
        self.par = None
        self.voc = None
        self.n_iters_sofar = None

        with open(file, 'r') as stream: opts = yaml.load(stream)
        for o,v in opts.items():
            if   o=="cuda":          self.cuda = bool(v)
            elif o=="cell":          self.cell = v.lower()
            elif o=="num_layers":    self.num_layers = int(v)
            elif o=="bidirectional": self.bidirectional = bool(v)
            elif o=="hidden_size":   self.hidden_size = int(v)
            elif o=="emb_size":      self.emb_size = int(v)
            elif o=="attention":     self.attention = v
            elif o=="coverage":      self.coverage = bool(v)
            elif o=="pointer":       self.pointer = bool(v)
            elif o=="method":        self.method = v
            elif o=="max_grad_norm": self.max_grad_norm = float(v)
            else: sys.exit("error: unparsed {} config option.".format(o))

    def add(self, par, voc=None):
        self.par = par
        if voc is not None: self.voc = voc
        self.out()

    def out(self):
        sys.stderr.write("CFG:")
        for k, v in sorted(vars(self).items()): 
            if (k!='par' and k!='voc'): sys.stderr.write(" {}: {}".format(k,v))
        sys.stderr.write("\n")
        self.par.out()
        sys.stderr.write("VOC: size: {}\n".format(self.voc.size))



class Params():

    def __init__(self, argv):

        usage = """usage: {}
           -batch_size  INT : batch size [32]
           -seed        INT : seed for randomness [12345]
           -h               : this message           
        TRAINING:
*          -dir        PATH : model checkpoints saved/restored in PATH
*          -trn        FILE : run training over FILE
*          -val        FILE : run validation over FILE
+          -cfg        FILE : topology config FILE (needed when training from scratch)
+          -voc        FILE : vocabulary FILE (needed when training from scratch)
           -max_src_len INT : maximum length of source sentences [400]
           -max_tgt_len INT : maximum length of source sentences [50]
           -emb        FILE : embeddings FILE (when training from scratch and if used)
           -n_iters     INT : number of iterations to run [10000]
           -dropout   FLOAT : dropout probability used on all layers [0.3]
           -lr        FLOAT : learning rate [1.0]
           -decay     FLOAT : decay [0.05]
           -print_every INT : print information every INT iterations [1000]
           -valid_every INT : validate and save every INT iterations [10000]
        INFERENCE:
*          -chk        FILE : checkpoint file to load when testing (or used the last one)
*          -tst        FILE : run inference over FILE
           -beam_size   INT : size of beam when decoding [5]
""".format(argv.pop(0))

        self.max_src_len = 400
        self.max_tgt_len = 50
        self.n_iters = 10000
        self.beam_size = 5
        self.batch_size = 16
        self.dropout = 0.3
        self.lr = 1.0
        self.decay = 0.05
        self.print_every = 1
        self.valid_every = 10
        self.seed = 12345
        self.cfg = None
        self.dir = None
        self.trn = None
        self.val = None
        self.tst = None
        self.chk = None
        self.voc = None
        self.emb = None
        while len(argv):
            tok = argv.pop(0)
            if   (tok=="-cfg"         and len(argv)): self.cfg = argv.pop(0)
            elif (tok=="-dir"         and len(argv)): self.dir = argv.pop(0)
            elif (tok=="-trn"         and len(argv)): self.trn = argv.pop(0)
            elif (tok=="-val"         and len(argv)): self.val = argv.pop(0)
            elif (tok=="-n_iters"     and len(argv)): self.n_iters = int(argv.pop(0))
            elif (tok=="-batch_size"  and len(argv)): self.batch_size = int(argv.pop(0))
            elif (tok=="-dropout"     and len(argv)): self.dropout = float(argv.pop(0))
            elif (tok=="-lr"          and len(argv)): self.lr = float(argv.pop(0))
            elif (tok=="-decay"       and len(argv)): self.decay = float(argv.pop(0))
            elif (tok=="-tst"         and len(argv)): self.tst = argv.pop(0)
            elif (tok=="-chk"         and len(argv)): self.chd = argv.pop(0)
            elif (tok=="-voc"         and len(argv)): self.voc = argv.pop(0)
            elif (tok=="-emb"         and len(argv)): self.emb = argv.pop(0)
            elif (tok=="-beam_size"   and len(argv)): self.beam_size = int(argv.pop(0))
            elif (tok=="-print_every" and len(argv)): self.print_every = int(argv.pop(0))
            elif (tok=="-valid_every" and len(argv)): self.valid_every = int(argv.pop(0))
            elif (tok=="-seed"        and len(argv)): self.seed = int(argv.pop(0))
            elif (tok=="-h"): sys.exit("{}".format(usage))
            else: sys.exit('error: unparsed {} option\n{}'.format(tok,usage))

        ### Checking some options
        if not self.trn and not self.tst: sys.stderr.write('error: missing -trn or -tst options\n{}'.format(usage))
        if self.trn and not self.dir: sys.stderr.write('error: missing -dir option\n{}'.format(usage))
        if self.trn and not self.val: sys.stderr.write('error: missing -val option\n{}'.format(usage))
        if self.tst and not self.chk: sys.stderr.write('error: missing -chk option\n{}'.format(usage))

    def out(self):
        sys.stderr.write("PAR: "+', '.join(['{0}: {1}'.format(k, v) for k,v in sorted(vars(self).items())])+"\n")

