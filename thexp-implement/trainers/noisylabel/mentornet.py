"""
reimplement of 《MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks》
    https://arxiv.org/abs/1712.05055
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
from trainers import NoisyParams

from trainers.mixin import *
from torch.nn import functional as F


class MentorNetParams(NoisyParams):

    def __init__(self):
        super().__init__()
        self.epoch = 400
        self.batch_size = 128
        self.loss_p_percentile = 0.7
        self.optim = self.create_optim('SGD',
                                       lr=0.3,
                                       momentum=0.9,
                                       weight_decay=1e-4,
                                       nesterov=True)

    def initial(self):
        self.lr_sche = self.SCHE.List(
            [
                self.SCHE.Linear(start=0.06, end=0.06, right=params.stage2),
                self.SCHE.Cos(start=0.2, end=0.002, left=params.stage2, right=params.epoch),
            ]
        )
        self.epoch_step = self.SCHE.Linear(end=100, right=self.epoch)


class MentorTrainer(datasets.IEGSyntheticNoisyMixin,
                    models.BaseModelMixin,
                    acc.ClassifyAccMixin,
                    losses.CELoss, losses.MentorLoss,
                    Trainer):

    def initial(self):
        super().initial()
        self.target_mem = torch.zeros(50000, params.n_classes, device=self.device, dtype=torch.float)

    def train_batch(self, eidx, idx, global_step, batch_data, params: MentorNetParams, device: torch.device):
        meter = Meter()
        ids, xs, axs, ys, nys = batch_data  # type:torch.Tensor

        # Basic parameter for mentornet
        _epoch_step = params.epoch_step(eidx)
        _zero_labels = torch.zeros_like(nys)
        _loss_p_percentile = torch.ones(100, dtype=torch.float) * params.loss_p_percentile
        _dropout_rates = self.parse_dropout_rate_list_()

        logits = self.to_logits(xs)
        _basic_losses = F.cross_entropy(logits, ys, reduction='none').detach()

        weight = torch.rand_like(nys).detach()  # TODO replace to mentornet

        meter.Lall = torch.mean(_basic_losses * weight)

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        self.acc_precise_(logits.argmax(dim=1), ys, meter, name='true_acc')
        self.acc_precise_(logits.argmax(dim=1), nys, meter, name='noisy_acc')

        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    params = MentorNetParams()
    params.from_args()
    trainer = MentorTrainer(params)

    trainer.train()
