# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

from torch import optim
from model.encoder import EncoderRNN
from model.decoder import DecoderRNN_Attn
from utils.utils import print_time

########################################################
### Model ##############################################
########################################################

class Model(nn.Module):

    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg

        self.embeddings_src = nn.Embedding(cfg.svoc.size, cfg.emb_src_size) ### embeddings for encoder
        self.encoder = EncoderRNN(self.embeddings_src, self.cfg)

        if self.cfg.reuse_words: self.embeddings_tgt = self.embeddings_src ### same embeddings for encoder and decoder
        else: self.embeddings_tgt = nn.Embedding(self.cfg.tvoc.size, self.cfg.emb_tgt_size) ### new embeddings for decoder
        self.decoder = DecoderRNN_Attn(self.embeddings_tgt, self.cfg) 

        sys.stderr.write('Initializing model pars\n')
        for param in self.encoder.parameters(): param.data.uniform_(-0.08, 0.08)
        for param in self.decoder.parameters(): param.data.uniform_(-0.08, 0.08)

    def forward(self, src_batch, tgt_batch, len_src_batch, len_tgt_batch, teacher_forcing=1.0):
        enc_outputs, enc_final = self.encoder(src_batch,len_src_batch)
        dec_outputs, dec_output_words = self.decoder(tgt_batch, len_src_batch, len_tgt_batch, enc_final, enc_outputs, teacher_forcing)
        return dec_outputs, dec_output_words

    def beam_search(self, src_batch, tgt_batch, len_src_batch, len_tgt_batch, teacher_forcing=1.0):
        enc_outputs, enc_final = self.encoder(src_batch,len_src_batch)

        beams = [beam_search(cfg)]
        for t in range(cfg.max_tgt_len):
            if all(beam.done() for beam in beams): break #all beam finished in batches

            # Construct batch x beam_size nxt words. Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([beam.get_current_state() for beam in beams]).t().contiguous().view(1, -1))

            # Turn any copied words to UNKs. 0 is unk
            if self.copy_attn:
                inp = inp.masked_fill(inp.gt(len(self.fields["tgt"].vocab) - 1), 0)

            # Temporary kludge solution to handle changed dim expectation in the decoder
            inp = inp.unsqueeze(2)

            # Run one step.
            dec_out, attn = DecoderRNN_Attn(inp, memory_bank, memory_lengths=memory_lengths, step=i)

            dec_out = dec_out.squeeze(0)
            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch x beam word scores.
            if not self.copy_attn:
                out = self.model.generator.forward(dec_out).data
                out = unbottle(out)
                # beam x tgt_vocab
                beam_attn = unbottle(attn["std"])
            else:
                out = self.model.generator.forward(dec_out, attn["copy"].squeeze(0), src_map)
                # beam x (tgt_vocab + extra_vocab)
                out = data.collapse_copy_scores(unbottle(out.data), batch, self.fields["tgt"].vocab, data.src_vocabs)
                # beam x tgt_vocab
                out = out.log()
                beam_attn = unbottle(attn["copy"])

            # (c) Advance each beam.
            select_indices_array = []
            for j, b in enumerate(beam):
                b.advance(out[:, j], beam_attn.data[:, j, :memory_lengths[j]])
                select_indices_array.append(b.get_current_origin() * batch_size + j)
            select_indices = torch.cat(select_indices_array).view(batch_size, beam_size).transpose(0, 1).contiguous().view(-1)
            self.model.decoder.map_state(lambda state, dim: state.index_select(dim, select_indices))

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size
        if "tgt" in batch.__dict__: ret["gold_score"] = self._run_target(batch, data)
        ret["batch"] = batch

        return ret
