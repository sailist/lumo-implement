"""
噪音的梯度测试

"""
from collections import defaultdict
from itertools import cycle

if __name__ == '__main__':
    import os
    import sys

    chdir = os.path.dirname(os.path.abspath(__file__))
    chdir = os.path.dirname(chdir)
    chdir = os.path.dirname(chdir)
    sys.path.append(chdir)

import numpy as np
import torch
from torch import autograd
from thexp import Trainer, Meter, Params, AvgMeter
from torch.nn import functional as F

from trainers import NoisyParams, GlobalParams
from trainers.mixin import *


class FitEvalParams(NoisyParams):

    def __init__(self):
        super().__init__()
        self.train_size = 2000
        # clean 表示直接拟合干净的后再去拟合噪音
        # noisy 表示先拟合一遍噪音，然后再拟合一遍干净的，然后再拟合一遍噪音
        self.eval_mode = self.choice('eval_mode', 'clean', 'noisy')

    def initial(self):
        super().initial()
        self.lr_sche = None


class MixupEvalTrainer(datasets.SyntheticNoisyMixin,
                       callbacks.BaseCBMixin, callbacks.callbacks.TrainCallback,
                       models.BaseModelMixin,
                       acc.ClassifyAccMixin,
                       losses.CELoss, losses.MixMatchLoss, losses.IEGLoss,
                       Trainer):
    priority = -1

    def callbacks(self, params: GlobalParams):
        super().callbacks(params)
        self.remove_callback(callbacks.callbacks.LRSchedule)
        self.hook(self)

    def train_batch(self, eidx, idx, global_step, batch_data, params: NoisyParams, device: torch.device):
        meter = Meter()
        ids, xs, axs, ys, nys = batch_data  # type:torch.Tensor

        logits = self.to_logits(xs)
        meter.Lall = meter.Lall + self.loss_ce_(logits, nys, meter=meter)

        self.acc_precise_(logits.argmax(dim=1), nys, meter, name='acc')
        self.acc_precise_(logits.argmax(dim=1), ys, meter, name='tacc')

        # _err = (ys != nys)
        # ids, xs, axs, ys, nys = ids[_err], xs[_err], axs[_err], ys[_err], nys[_err]

        for i in range(params.n_classes):
            _cls_mask = (ys == 0)
            _fcls_mask = (nys == 1)
            _tloss = F.cross_entropy(logits[_cls_mask], ys[_cls_mask], reduction='none')
            _floss = F.cross_entropy(logits[_fcls_mask], nys[_fcls_mask], reduction='none')

            tcls_grads = [i for i in
                          autograd.grad(_tloss, self.model.parameters(), grad_outputs=torch.ones_like(_tloss),
                                        retain_graph=True,
                                        allow_unused=True) if i is not None]

            fcls_grads = [i for i in
                          autograd.grad(_floss, self.model.parameters(), grad_outputs=torch.ones_like(_floss),
                                        retain_graph=True,
                                        allow_unused=True) if i is not None]
            res = 0
            for tgrad, fgrad in zip(tcls_grads, fcls_grads):
                # res += (tgrad - fgrad).abs().pow(2).sum()
                # res = max(tricks.cos_similarity(tgrad, fgrad).mean(), res)
                res += tricks.cos_similarity(tgrad, fgrad).mean()

            meter.res = res
            break

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)

    def on_train_epoch_end(self, trainer: Trainer, func, params: Params, meter: AvgMeter, *args, **kwargs):
        super().on_train_epoch_end(trainer, func, params, meter, *args, **kwargs)


if __name__ == '__main__':
    params = FitEvalParams()
    params.wideresnet282()
    params.optim.args.lr = 0.02
    params.epoch = 1000
    params.batch_size = 128
    params.device = 'cuda:2'
    params.ema = False
    params.large_model = False
    params.from_args()
    trainer = MixupEvalTrainer(params)
    trainer.train()
