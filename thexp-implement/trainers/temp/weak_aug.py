"""
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
from trainers import SupervisedParams
from trainers.mixin import *
from torch import autograd


class SupervisedTrainer(callbacks.BaseCBMixin,
                        datasets.Base32Mixin,
                        models.BaseModelMixin,
                        acc.ClassifyAccMixin,
                        losses.CELoss, callbacks.callbacks.TrainCallback,
                        Trainer):

    def train_batch(self, eidx, idx, global_step, batch_data, params: SupervisedParams, device: torch.device):
        super().train_batch(eidx, idx, global_step, batch_data, params, device)
        meter = Meter()
        ids, xs, axs, ys = batch_data  # type:torch.Tensor
        logits = self.to_logits(xs)

        if eidx < 10:
            # mask = (ids % 1000) < eidx
            targets = torch.ones(xs.shape[0], params.n_classes, device=device) / params.n_classes
            # targets[mask] = tricks.onehot(ys[mask], params.n_classes)

            meter.Lall = meter.Lall + self.loss_ce_with_targets_(logits, targets, meter=meter)
        else:
            meter.Lall = meter.Lall + self.loss_ce_(logits, ys, meter=meter)

        self.acc_precise_(logits.argmax(dim=1), ys, meter, name='acc')

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    params = SupervisedParams()
    params.device = 'cuda:1'
    params.from_args()
    for _p in params.iter_baseline():
        for pp in _p.grid_range(1):  # try n times
            trainer = SupervisedTrainer(params)
            trainer.train()
