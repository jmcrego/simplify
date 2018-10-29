# -*- coding: utf-8 -*-

import io
import os
import sys
import gzip
import torch
import numpy as np
from random import shuffle
from collections import defaultdict

idx_unk = 0
idx_pad = 1
idx_ini = 2
idx_end = 3
str_unk = "<unk>"
str_pad = "<pad>"
str_ini = "<ini>"
str_end = "<end>"

########################################################
### Vocab ##############################################
########################################################

class Vocab():

    def __init__(self, file):
        self.tok_to_idx = {}
        self.idx_to_tok = []
        self.idx_to_tok.append(str_unk)
        self.tok_to_idx[str_unk] = len(self.tok_to_idx) #0
        self.idx_to_tok.append(str_pad)
        self.tok_to_idx[str_pad] = len(self.tok_to_idx) #1
        self.idx_to_tok.append(str_ini)
        self.tok_to_idx[str_ini] = len(self.tok_to_idx) #2
        self.idx_to_tok.append(str_end)
        self.tok_to_idx[str_end] = len(self.tok_to_idx) #3
        self.size = 4

        self.idx_unk = idx_unk
        self.idx_pad = idx_pad
        self.idx_ini = idx_ini
        self.idx_end = idx_end

        ### the file contains real words (not the previous special tokens)
        with io.open(file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line not in self.tok_to_idx:
                    self.idx_to_tok.append(line)
                    self.tok_to_idx[line] = len(self.tok_to_idx)
                    self.size += 1
        #self.size = len(self.idx_to_tok)
        sys.stderr.write('Read vocab ({} entries)\n'.format(self.size))

    def __iter__(self):
        for tok in self.idx_to_tok:
            yield tok

    def exists(self, s):
        return s in self.tok_to_idx

    def get(self,s):
        if type(s) == int: ### I want the string
            if s < self.size: return self.idx_to_tok[s]
            else:
                sys.exit('error: key \'{}\' not found in vocab\n'.format(s))
        ### I want the index
        if s not in self.tok_to_idx: return idx_unk
        return self.tok_to_idx[s]

########################################################
### Embed ##############################################
########################################################

class Embed():

    def __init__(self, file, cfg, voc):
        emb_size = cfg.emb_size
        w2e = {}
        if file is not None:
            #with io.open(file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            if file.endswith('.gz'): f = gzip.open(file, 'rb')
            else: f = io.open(file, 'r', encoding='utf-8', newline='\n', errors='ignore')
            ### first line
            num, self.dim = map(int, f.readline().split())
            for line in f:
                tokens = line.rstrip().split(' ')
                if voc.exists(tokens[0]): w2e[tokens[0]] = tokens[1:] 
            f.close()
            if emb_size != self.dim: 
                sys.stderr.write('warning: emb_size option in config file does not match with dimension of embedding file: replacing -emb_size {} by {}\n'.format(cfg.emb_size,self.dim))
                cfg.emb_size = self.dim

            sys.stderr.write('Read {} embeddings ({} missing in voc)\n'.format(len(w2e),voc.length-len(w2e)))
        else:
            sys.stderr.write('Embeddings randomly initialized dim={}\n'.format(emb_size))
            self.dim = emb_size

        # i need an embedding for each word in voc
        # embedding matrix must have tokens in same order than voc 0:<unk>, 1:<pad>, 2:le, ...
        self.matrix = []
        for tok in voc:
            if not tok in w2e: ### random initialize these tokens
                self.matrix.append(np.random.normal(0, 1.0, self.dim))
            else:
                self.matrix.append(np.asarray(w2e[tok], dtype=np.float32))

        self.matrix = np.asarray(self.matrix, dtype=np.float32)
        self.matrix = self.matrix / np.sqrt((self.matrix ** 2).sum(1))[:, None]

########################################################
### Dataset ############################################
########################################################

class Dataset():

    def __init__(self, file, svoc, tvoc, batch_size, max_src_len, max_tgt_len, do_shuffle, do_filter, is_test):
        if file is None: return
        self.is_test = is_test
        self.do_shuffle = shuffle
        self.do_filter = do_filter
        self.file = file
        self.batch_size = batch_size
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.svoc = svoc
        self.tvoc = tvoc
        self.data = []
        if self.file.endswith('.gz'): f = gzip.open(self.file, 'rb')
        else: f = io.open(self.file, 'r', encoding='utf-8', newline='\n', errors='ignore')
        for line in f: self.data.append(line)
        f.close()
        sys.stderr.write('Read dataset {} (contains {} examples)\n'.format(self.file, len(self.data)))
        self.length = len(self.data)

    def __iter__(self):
        ### every iteration i get shuffled data examples if do_shuffle
        indexs = [i for i in range(len(self.data))]
        if self.do_shuffle: shuffle(indexs)
        while True:
            self.nsent = 0
            self.nsrc = 0
            self.ntgt = 0
            self.nunk_src = 0
            self.nunk_tgt = 0
            for index in indexs:
                tokens = self.data[index].strip().split('\t')
                if len(tokens) > 2 or len(tokens)<1:
                    #sys.stderr.write("warning: bad data entry in line={} [skipped]\n".format(index+1))
                    continue
                ### filter out sentences not respecting limits
                src, tgt = [], []
                if len(tokens)>=1:
                    src = tokens[0].split(' ')
                    if len(src) == 0 or (self.do_filter and self.max_src_len > 0 and len(src) > self.max_src_len): 
                        #sys.stderr.write("filtered entry by src_len={} in line={}\n".format(len(src),index+1))
                        continue
                if len(tokens)==2:
                    tgt = tokens[1].split(' ')
                    if len(tgt) == 0 or (self.do_filter and self.max_tgt_len > 0 and len(tgt) > self.max_tgt_len): 
                        #sys.stderr.write("filtered entry by tgt_len={} in line={}\n".format(len(tgt),index+1))
                        continue
                ### src tokens
                isrc = []
                for s in src: 
                    isrc.append(self.svoc.get(s))
                    if isrc[-1]==idx_unk: self.nunk_src += 1
                    self.nsrc += 1
                ### tgt tokens
                itgt = []
                for t in tgt: 
                    itgt.append(self.tvoc.get(t))
                    if itgt[-1]==idx_unk: self.nunk_tgt += 1
                    self.ntgt += 1
                yield isrc, itgt, src, tgt ### return for iterator
                self.nsent += 1
            sys.stderr.write('Finished loop over {}: unpruned examples={} out of {}, nwords=({}/{}), nunks=({},{})\n'.format(self.file, self.nsent, len(indexs), self.nsrc, self.ntgt, self.nunk_src, self.nunk_tgt))
            if self.is_test: break

    def minibatches(self):
        minibatch_size = self.batch_size
        SRC, TGT, RAW_SRC, RAW_TGT = [], [], [], []
        max_src_batch, max_tgt_batch = 0, 0
        for (src, tgt, raw_src, raw_tgt) in self: ### uses the iterator defined just before
            if len(SRC) == minibatch_size:
                yield self.build_batch(SRC, TGT, RAW_SRC, RAW_TGT, max_src_batch, max_tgt_batch)
                SRC, TGT, RAW_SRC, RAW_TGT = [], [], [], []
                max_src_batch, max_tgt_batch = 0, 0
            if len(src) > max_src_batch: max_src_batch = len(src)
            if len(tgt) > max_tgt_batch: max_tgt_batch = len(tgt)
            SRC.append(src)
            TGT.append(tgt)
            RAW_SRC.append(raw_src)
            RAW_TGT.append(raw_tgt)
        if len(SRC) != 0:
            yield self.build_batch(SRC, TGT, RAW_SRC, RAW_TGT, max_src_batch, max_tgt_batch)

    def build_batch(self, SRC, TGT, RAW_SRC, RAW_TGT, max_src_batch, max_tgt_batch):
        len2entries = defaultdict(list)
        src_batch, tgt_batch, ref_batch, raw_src_batch, raw_tgt_batch, len_src_batch = [], [], [], [], [], []
        for i in range(len(SRC)):
            raw_src = list(RAW_SRC[i])
            raw_tgt = list(RAW_TGT[i])
            len_src = len(SRC[i]) ### J
            #### src: s1 s2 s3 ... sj ... sJ <pad> <pad> <pad> ...
            src = list(SRC[i])    
            while len(src) < max_src_batch: src.append(idx_pad)
            #### tgt: <ini> t1 t2 t3 ... ti ... tI <end> <pad> ...
            tgt = [idx_ini]       
            tgt.extend(TGT[i])
            tgt.extend([idx_end])
            while len(tgt) < max_tgt_batch + 2: tgt.append(idx_pad) 
            #### ref: t1 t2 t3 ... ti ... tI <end> <pad> ...
            ref = list(TGT[i]) 
            ref.extend([idx_end])
            while len(ref) < max_tgt_batch + 1: ref.append(idx_pad) 
            ### batch entries have to be sorted in length decreasing order
            len2entries[len_src].append([src,tgt,ref,len_src,raw_src,raw_tgt])

        ### batch entries have to be sorted in length decreasing order
        for l,entries in sorted(len2entries.items(), reverse=True):
            for entry in entries:
                src_batch.append(entry[0])
                tgt_batch.append(entry[1])
                ref_batch.append(entry[2])
                len_src_batch.append(entry[3])
                raw_src_batch.append(entry[4])
                raw_tgt_batch.append(entry[5])
        return torch.tensor(src_batch), torch.tensor(tgt_batch), torch.tensor(ref_batch), raw_src_batch, raw_tgt_batch, torch.tensor(len_src_batch)

    def __len__(self):
        return self.length
