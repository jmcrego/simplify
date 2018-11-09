import torch
from collections import defaultdict
from utils.data import idx_unk, idx_pad, idx_ini, idx_end

class Beam_state(object, attn_hidden, dec_output, enc_coverage, father, cost):
    self.attn_hidden = attn_hidden
    self.dec_output = dec_output
    self.enc_coverage = enc_coverage
    self.father = father
    self.cost = cost ### prob of this timestep

class Beam(object):

    def __init__(self, beam_size, n_best, rnn_hidden, enc_output, idx_pad, idx_ini, idx_end, cuda=False):

        self.b = beam_size # beam size
        self.n = n_best # n best 
        self.idx_pad = idx_pad
        self.idx_ini = idx_ini
        self.idx_end = idx_end
        self.tt = torch.cuda if cuda else torch

#        states = [] #list of Beam_state
#        scores = defaultdict(float) #score of each state in self.states {0: score, 1: score, ...} we will need to sort before we expand
#        sorted_scores = sorted(scores.items(), key=lambda (k, v): v)

        ### next are the structures to keep the active search space (expanded states) used at the end of the search to trace back the (n-) best hypotheses
        self.cur_scores = self.tt.FloatTensor(self.b).zero_() # The score for each translation on the beam.
        self.cur_father = [] # The backpointers at each time-step.
        self.cur_output = self.tt.LongTensor(self.b).fill_(idx_pad)
        self.cur_output[0] = idx_ini

        self.all_scores = []
        self.all_output = [] # The outputs at each time-step

        self.finished = [False * self.b]
 

    def get_current_state(self):
        return self.next_ys[-1]

    def get_current_origin(self):
        return self.prev_ks[-1]

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            attn.append(self.attn[j][k])
            k = self.prev_ks[j][k]
        return hyp[::-1], torch.stack(attn[::-1])

    def forward(self, word_probs, attn_out):
        #word_probs [B,V] [B,S]
        """
        Given prob over words for every last beam `wordLk` and attention `attn_out`: Compute and update the beam search.
        Parameters:
        * `word_probs`- probs of advancing from the last step (K x words)
        * `attn_out`- attention at the last step
        Returns: True if beam search is complete.
        """
        num_words = word_probs.size(1)
#        if self.stepwise_penalty: self.global_scorer.update_score(self, attn_out)

        # force the output to be longer than self.min_length
        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_probs)):
                word_probs[k][self._eos] = -1e20

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = word_probs + self.scores.unsqueeze(1).expand_as(word_probs)
            # Don't let EOS have children.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20

            # Block ngram repeats
#            if self.block_ngram_repeat > 0:
#                ngrams = []
#                le = len(self.next_ys)
#                for j in range(self.next_ys[-1].size(0)):
#                    hyp, _ = self.get_hyp(le - 1, j)
#                    ngrams = set()
#                    fail = False
#                    gram = []
#                    for i in range(le - 1):
#                        # Last n tokens, n = block_ngram_repeat
#                        gram = (gram + [hyp[i].item()])[-self.block_ngram_repeat:]
#                        # Skip the blocking if it is in the exclusion list
#                        if set(gram) & self.exclusion_tokens: continue
#                        if tuple(gram) in ngrams: fail = True
#                        ngrams.add(tuple(gram))
#                    if fail:
#                        beam_scores[j] = -10e20
        else:
            beam_scores = word_probs[0]

        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_scores_id = flat_beam_scores.topk(self.b, 0, True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * num_words))
        self.attn.append(attn_out.index_select(0, prev_k))
        self.global_scorer.update_global_state(self)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == self._eos:
            self.all_scores.append(self.scores)
            self.eos_top = True



