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


class SupervisedTrainer(callbacks.BaseCBMixin,
                        datasets.Base32Mixin,
                        models.BaseModelMixin,
                        acc.ClassifyAccMixin,
                        losses.CELoss, losses.MixMatchLoss,
                        callbacks.callbacks.TrainCallback,
                        Trainer):

    def train_batch(self, eidx, idx, global_step, batch_data, params: SupervisedParams, device: torch.device):
        super().train_batch(eidx, idx, global_step, batch_data, params, device)
        meter = Meter()
        ids, xs, axs, ys = batch_data  # type:torch.Tensor

        targets = tricks.onehot(ys, params.n_classes)
        mixed_xs, mixed_targets = self.mixup_(xs, targets)

        mixed_logits = self.to_logits(mixed_xs)
        meter.Lall = meter.Lall + self.loss_ce_with_targets_(mixed_logits, mixed_targets,
                                                             meter=meter)

        with torch.no_grad():
            self.acc_precise_(self.to_logits(xs).argmax(dim=1), ys,
                              meter=meter, name='acc')

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    params = SupervisedParams()
    params.device = 'cuda:0'
    params.from_args()
    # for _p in params.iter_baseline():
    #     for pp in _p.grid_range(1):  # try n times
    #         trainer = SupervisedTrainer(params)
    #         trainer.train()

    params.widenet282()
    trainer = SupervisedTrainer(params)
    trainer.train()
