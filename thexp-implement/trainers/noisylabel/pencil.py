"""
reimplement of 《Probabilistic End-to-end Noise Correction for Learning with Noisy Labels pencil》
    https://arxiv.org/abs/1903.07788
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


class PencilParams(NoisyParams):

    def __init__(self):
        super().__init__()
        self.epoch = 400

        self.stage1 = 70
        self.stage2 = 200

        self.alpha = 0.4
        self.beta = 0.1
        self.lambda1 = 600  # the value of lambda
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


class PencilTrainer(datasets.IEGSyntheticNoisyMixin,
                    models.BaseModelMixin,
                    acc.ClassifyAccMixin,
                    losses.CELoss, losses.PencilLoss,
                    Trainer):

    def initial(self):
        super().initial()
        self.target_mem = torch.zeros(50000, params.n_classes, device=self.device, dtype=torch.float)

    def train_batch(self, eidx, idx, global_step, batch_data, params: PencilParams, device: torch.device):
        meter = Meter()
        ids, xs, axs, ys, nys = batch_data  # type:torch.Tensor

        logits = self.to_logits(xs)
        if eidx < params.stage1:
            # lc is classification loss
            Lce = self.loss_ce_(logits, nys, meter=meter, name='Lce')

            # init y_tilde, let softmax(y_tilde) is noisy labels
            noisy_targets = tricks.onehot(nys, params.n_classes)
            self.target_mem[ids] = noisy_targets
        else:
            yy = self.target_mem[ids]
            yy = torch.autograd.Variable(yy, requires_grad=True)

            # obtain label distributions (y_hat)
            last_y_var = torch.softmax(yy, dim=1)

            Lce = self.loss_ce_with_lc_targets_(logits, last_y_var, meter=meter, name='Lce')
            Lco = self.loss_ce_(last_y_var, nys, meter=meter, name='Lco')

        # le is entropy loss
        Lent = self.loss_ent_(logits, meter=meter, name='Lent')

        if eidx < params.stage1:
            meter.Lall = Lce
        elif eidx < params.stage2:
            meter.Lall = Lce + params.alpha * Lco + params.beta * Lent
        else:
            meter.Lall = Lce

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        self.acc_precise_(logits.argmax(dim=1), ys, meter, name='true_acc')
        self.acc_precise_(logits.argmax(dim=1), nys, meter, name='noisy_acc')

        if eidx >= params.stage1 and eidx < params.stage2:
            self.target_mem[ids] = self.target_mem[ids] - params.lambda1 * yy.grad.data
            self.acc_precise_(self.target_mem[ids].argmax(dim=1), ys, meter, name='check_acc')

        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    params = PencilParams()
    params.from_args()
    trainer = PencilTrainer(params)

    trainer.train()
