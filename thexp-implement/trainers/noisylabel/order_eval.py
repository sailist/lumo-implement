"""
该Trainer 验证噪音混合方式对结果的影响。

模型是以 batch size 为单位更新的，那么，如果将噪音标签按照不同粒度，不同比例混入干净标签，其余训练方式不变，那么模型的表现是否会有不同？

比如说，将噪音标签集中出现的一段batch中；比如说，均匀的将噪音混合在batch中；比如说，将噪音标签尽可能的集中在batch中，但混合着和干净batch一起训练。

如果某种方式更有效，那么就可以更改数据集的序号排列方式，然后按照某种方式将噪音标签和干净标签区分开来二次重组，然后进行训练。

随机分布的 baseline：0018.d9661979（来自 mixup_eval.py）

目前看来，似乎区别不大
0001.6f53c3a1


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

    def train_batch(self, eidx, idx, global_step, batch_data, params: NoisyParams, device: torch.device):
        meter = Meter()
        ids, xs, axs, ys, nys = batch_data  # type:torch.Tensor

        logits = self.to_logits(xs)
        meter.Lall = meter.Lall = self.loss_ce_(logits, nys, meter=meter)

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
    params.order_sampler = True
    params.from_args()
    params.initial()
    trainer = MixupEvalTrainer(params)
    trainer.train()
