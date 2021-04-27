"""
reimplement of 《MixMatch: A Holistic Approach to Semi-Supervised Learning》
    https://arxiv.org/abs/1905.02249
"""
if __name__ == '__main__':
    import os
    import sys

    chdir = os.path.dirname(os.path.abspath(__file__))
    chdir = os.path.dirname(chdir)
    chdir = os.path.dirname(chdir)
    sys.path.append(chdir)

import torch

from thexp import Trainer, Meter
from trainers import SemiSupervisedParams

from trainers.mixin import *


class MixMatchParams(SemiSupervisedParams):

    def __init__(self):
        super().__init__()
        self.epoch = 1024
        self.batch_size = 64
        self.optim = self.create_optim('Adam',
                                       lr=0.002,
                                       weight_decay=0.02 * 0.002)  # wd = flags.wd * lr = 0.02 * 0.002 = 0.00004)
        self.K = 2
        self.T = 0.5
        self.w_watch = 75

    def initial(self):
        super().initial()
        self.w_sche = self.SCHE.Linear(1, self.w_watch, right=self.epoch)


class MixMatchTrainer(callbacks.BaseCBMixin,
                      datasets.MixMatchDatasetMixin,
                      models.BaseModelMixin,
                      acc.ClassifyAccMixin,
                      losses.CELoss, losses.MixMatchLoss,
                      Trainer):

    def train_batch(self, eidx, idx, global_step, batch_data, params: MixMatchParams, device: torch.device):
        super().train_batch(eidx, idx, global_step, batch_data, params, device)
        meter = Meter()
        sup, unsup = batch_data
        xs, ys = sup
        un_imgs, un_labels = unsup[:params.K], unsup[params.K]

        targets = tricks.onehot(ys, params.n_classes)

        un_logits = self.to_logits(torch.cat(un_imgs))
        un_targets = self.label_guesses_(*un_logits.chunk(params.K))
        un_targets = self.sharpen_(un_targets, params.T)

        mixed_input, mixed_target = self.mixmatch_up_(xs, un_imgs, targets, un_targets)

        sup_mixed_target, unsup_mixed_target = mixed_target.split_with_sizes(
            [xs.shape[0], mixed_input.shape[0] - xs.shape[0]])

        sup_mixed_logits, unsup_mixed_logits = self.to_logits(mixed_input).split_with_sizes(
            [xs.shape[0], mixed_input.shape[0] - xs.shape[0]])

        meter.Lall = meter.Lall + self.loss_ce_with_targets_(sup_mixed_logits, sup_mixed_target,
                                                             meter=meter, name='Lx')
        meter.Lall = meter.Lall + self.loss_ce_with_targets_(unsup_mixed_logits, unsup_mixed_target,
                                                             meter=meter, name='Lu') * params.w_sche(eidx)

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        self.acc_precise_(self.predict(xs).argmax(dim=1), ys, meter)
        self.acc_precise_(un_logits[0].argmax(dim=1), un_labels, meter, name='unacc')

        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    params = MixMatchParams()
    params.device = 'cuda:1'
    params.from_args()
    trainer = MixMatchTrainer(params)
    trainer.train()
