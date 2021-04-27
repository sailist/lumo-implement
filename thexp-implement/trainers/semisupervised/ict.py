"""
reimplement of 《Interpolation Consistency Training for Semi-Supervised Learning》
    https://arxiv.org/abs/1903.03825
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


class ICTParams(SemiSupervisedParams):

    def __init__(self):
        super().__init__()
        self.epoch = 400
        self.batch_size = 128
        self.optim = self.create_optim('SGD',
                                       lr=0.1,
                                       weight_decay=0.0001,
                                       momentum=0.9,
                                       nesterov=False)

    def initial(self):
        super().initial()
        self.mixup_consistency_sche = self.SCHE.Cos(0, 100, right=100)


class ICTTrainer(callbacks.BaseCBMixin,
                 datasets.FixMatchDatasetMixin,
                 models.BaseModelMixin,
                 acc.ClassifyAccMixin,
                 losses.CELoss, losses.ICTLoss,
                 Trainer):

    def train_batch(self, eidx, idx, global_step, batch_data, params: ICTParams, device: torch.device):
        super().train_batch(eidx, idx, global_step, batch_data, params, device)
        meter = Meter()

        sup, unsup = batch_data
        xs, ys = sup
        _, un_xs, _, un_ys = unsup

        logits_list = self.to_logits(torch.cat([xs, un_xs])).split_with_sizes(
            [xs.shape[0], un_xs.shape[0]])
        logits, un_logits = logits_list  # type:torch.Tensor

        mixed_xs, ys_a, ys_b, lam_sup = self.ict_mixup_(xs, ys)

        logits = self.to_logits(xs)
        mixed_logits = self.to_logits(mixed_xs)

        ema_un_logits = self.predict(un_xs)

        mixed_un_xs, ema_mixed_un_logits, _ = self.mixup_unsup_(un_xs, ema_un_logits, mix=True)

        mixed_un_logits = self.to_logits(mixed_un_xs)

        meter.Lall = meter.Lall + self.loss_mixup_sup_ce_(mixed_logits, ys_a, ys_b, lam_sup, meter=meter)
        meter.Lall = meter.Lall + self.loss_mixup_unsup_mse_(mixed_un_logits, ema_un_logits,
                                                             decay=params.mixup_consistency_sche(eidx),
                                                             meter=meter)

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        self.acc_precise_(logits.argmax(dim=1), ys, meter)
        self.acc_precise_(un_logits.argmax(dim=1), un_ys, meter, name='unacc')

        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    params = ICTParams()
    params.device = 'cuda:3'
    params.from_args()
    trainer = ICTTrainer(params)
    trainer.train()
