""" Generator module """
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda

import onmt.inputters as inputters
from onmt.utils import loss

from pdb import set_trace

from rdkit import RDLogger

lg = RDLogger.logger()

lg.setLevel(RDLogger.CRITICAL)


class CanonicalAccuracy(object):
    def __init__(self, tgt_vocab):
        self.padding_idx = tgt_vocab.stoi[inputters.PAD_WORD]
        self.stop_idx = tgt_vocab.stoi[inputters.EOS_WORD]
        self.tgt_vocab = tgt_vocab
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _score(self, preds, gtruth):
        """
        both list of strings
        """
        from rdkit.Chem import MolFromSmiles, MolToSmiles

        scores = []

        for pred, gt in zip(preds, gtruth):
            # canonical prediction
            if pred == gt:
                scores.append(1)
                continue
            # valid smiles ?
            mol = MolFromSmiles(pred)
            if mol is None:
                scores.append(-1) # penalize invalidity
                continue
            try:
                can_pred = MolToSmiles(mol, isomericSmiles=True)
            except RuntimeError:
                scores.append(-1)
                continue
            if can_pred == gt:
                scores.append(1)
                continue

            scores.append(0)
        return scores

    def score(self, sampled_preds, greedy_preds, tgt):
        """
            sample_pred: LongTensor [bs x len]
            greedy_pred: LongTensor [bs x len]
            tgt: LongTensor [bs x len]
        """
        def pred2trans(batch_preds):
            translations = []
            for seq in batch_preds:
                translation = ''
                for token in seq:
                    if token in [self.stop_idx]: # should we also stop translation on pad_idx?
                        break
                    translation += self.tgt_vocab.itos[token]
                translations.append(translation)
            return translations

        sampled_hyps = pred2trans(sampled_preds)
        greedy_hyps = pred2trans(greedy_preds)
        gtruth = pred2trans(tgt)
        sampled_scores = self._score(sampled_hyps, gtruth)
        greedy_scores = self._score(greedy_hyps, gtruth)

        ts = torch.Tensor(sampled_scores).to(self.device)
        gs = torch.Tensor(greedy_scores).to(self.device)

        return (gs - ts)


class MixedLossCompute(loss.LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, generator, tgt_vocab, normalization="sents",
                 label_smoothing=0.0, gamma=0.9984, train=True):
        super(MixedLossCompute, self).__init__(generator, tgt_vocab)
        assert 0.0 <= label_smoothing <= 1.0
        weight = torch.ones(len(tgt_vocab))
        weight[self.padding_idx] = 0
        if label_smoothing > 0:
            # When label smoothing is turned on,
            # KL-divergence between q_{smoothed ground truth prob.}(w)
            # and p_{prob. computed by model}(w) is minimized.
            # If label smoothing value is set to zero, the loss
            # is equivalent to NLLLoss or CrossEntropyLoss.
            # All non-true labels are uniformly set to low-confidence.
            self.ml_criterion = nn.KLDivLoss(size_average=False)
            one_hot = torch.randn(1, len(tgt_vocab))
            one_hot.fill_(label_smoothing / (len(tgt_vocab) - 2))
            one_hot[0][self.padding_idx] = 0
            self.register_buffer('one_hot', one_hot)

        else:
            self.ml_criterion = nn.NLLLoss(weight, size_average=False)
        # negative log likely-hood loss for rl 
        self.rl_criterion = nn.NLLLoss(weight, size_average=False)
        self.confidence = 1.0 - label_smoothing
        self.gamma = gamma
        self.scoring_function = CanonicalAccuracy(tgt_vocab)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def _make_shard_state(self, batch, output, range_, attns=None):
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
        }

    def _compute_loss(self, batch, output, target):
        scores = self.generator(self._bottle(output))
        gtruth = target.view(-1)
        if self.confidence < 1:
            tdata = gtruth.data
            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze(-1)
            # log_likelihood = torch.gather(scores.data, 1, tdata.unsqueeze(1))
            tmp_ = self.one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            if mask.size(0) > 0:
                tmp_.index_fill_(0, mask, 0)
            gtruth = tmp_
        ml_loss = self.ml_criterion(scores, gtruth)

        if self.gamma > 0:
            _, ml_pred = scores.max(1)

            dist = torch.distributions.Categorical(scores)
            rl_pred = dist.sample() # * target.neq(self.padding_idx)

            rl_loss = self.rl_criterion(scores, rl_pred)

            metric = self.scoring_function.score(
                                    rl_pred.view(target.size()).t(),
                                     ml_pred.view(target.size()).t(), 
                                     target.t()
                                 )

            metric = metric.to(self.device)

            rl_loss = (rl_loss * metric).sum()
            loss = (self.gamma * rl_loss) + ((1 - self.gamma) * ml_loss)
        else:
            loss = ml_loss


        if self.confidence < 1:
            loss_data = loss.data.clone()
        else:
            loss_data = loss.data.clone()

        stats = self._stats(loss_data, scores.data, target.view(-1).data)

        return loss, stats





