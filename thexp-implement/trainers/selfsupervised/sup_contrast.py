"""
Reimplement of  Supervised Contrastive Learning
    https://arxiv.org/pdf/2004.11362.pdf

official repo：https://github.com/HobbitLong/SupContrast
"""

if __name__ == '__main__':
    import os
    import sys

    chdir = os.path.dirname(os.path.abspath(__file__))
    chdir = os.path.dirname(chdir)
    chdir = os.path.dirname(chdir)
    sys.path.append(chdir)

import torch
from thextra import memory_bank
from thexp import Trainer, Meter, Params, AvgMeter
from trainers import SupConParams
from trainers.mixin import *


class SimclrTrainer(callbacks.BaseCBMixin,
                    datasets.Base32Mixin,
                    models.SimCLRMixin,
                    losses.CELoss, losses.SupConLoss,
                    callbacks.callbacks.TrainCallback,
                    Trainer):

    def callbacks(self, params: SupConParams):
        super().callbacks(params)
        self.hook(self)

    def on_train_epoch_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        super().on_train_epoch_end(trainer, func, params, meter, *args, **kwargs)

    def train_batch(self, eidx, idx, global_step, batch_data, params: SupConParams, device: torch.device):
        super().train_batch(eidx, idx, global_step, batch_data, params, device)
        meter = Meter()
        _, xs, axs, ys = batch_data  # type:torch.Tensor

        b, c, h, w = xs.size()
        input_ = torch.cat([xs.unsqueeze(1), axs.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w)

        features = self.to_logits(input_).view(b, 2, -1)

        meter.Lall = meter.Lall + self.loss_supcon_(features, ys,
                                                    temperature=params.temperature,
                                                    meter=meter)

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    params = SupConParams()
    params.preresnet18()
    params.mid_dim = 512
    params.device = 'cuda:2'
    params.from_args()

    trainer = SimclrTrainer(params)
    trainer.train()
    trainer.saver.save_model(0, trainer.model.backbone.state_dict())
