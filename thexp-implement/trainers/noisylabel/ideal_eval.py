"""
该Trainer用于验证理想情况下准确率的变化，操作如下：
1. 每一个 epoch 按 loss 排序将噪声中最大的一部分权重置零
2. 每一个 epoch 将剩余的噪音权重调低一点点

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

from thexp import Trainer, Meter, Params
from torch.nn import functional as F

from trainers import NoisyParams, GlobalParams
from trainers.mixin import *


class MixupEvalTrainer(datasets.SyntheticNoisyMixin,
                       callbacks.BaseCBMixin, callbacks.callbacks.TrainCallback,
                       models.BaseModelMixin,
                       acc.ClassifyAccMixin,
                       losses.CELoss, losses.MixMatchLoss, losses.IEGLoss,
                       Trainer):

    def callbacks(self, params: GlobalParams):
        super().callbacks(params)
        self.hook(self)
        self.neg_dict = [[i for i in range(params.n_classes) if i != j] for j in range(params.n_classes)]

    def train_batch(self, eidx, idx, global_step, batch_data, params: NoisyParams, device: torch.device):
        meter = Meter()
        ids, xs, axs, ys, nys = batch_data  # type:torch.Tensor

        logits = self.to_logits(axs)

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        self.acc_precise_(logits.argmax(dim=1), ys, meter, name='true_acc')
        n_mask = nys != ys
        if n_mask.any():
            self.acc_precise_(logits.argmax(dim=1)[n_mask], nys[n_mask], meter, name='noisy_acc')

        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    params = NoisyParams()
    params.optim.args.lr = 0.06
    params.epoch = 400
    params.device = 'cuda:0'
    params.filter_ema = 0.999

    params.mixup = True
    params.ideal_mixup = True
    params.worst_mixup = False
    params.noisy_ratio = 0.8
    params.from_args()
    params.initial()
    trainer = MixupEvalTrainer(params)
    trainer.train()
