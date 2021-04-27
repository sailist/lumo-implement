"""
一个新的 noisy label 的解决方法

    label smoothing 后评估概率变化，随后决定下一次 label smoothing 的各概率的情况，以此来做 label correction

    怎么确保 label 收敛呢？

    nys 对应的初始的权重为 0.9，其余为 0.1

    在训练完后，对比差距，概率越大的，变化权重越小，概率越小的，变化权重越大
    （目前暂时决定为线性变化，或许根据一般的训练曲线，考虑 log/exp ）

    根据差距更新标签，先直接把变化相加，然后归一化到 0-1 ？

    然后设计一个度量指标，评价每次是否都向正确标签移动了（是否）


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
        self.smooth_cls_mem = torch.zeros(self.train_size, params.n_classes, device=self.device)

    def train_batch(self, eidx, idx, global_step, batch_data, params: NoisyParams, device: torch.device):
        meter = Meter()
        ids, xs, axs, ys, nys = batch_data  # type:torch.Tensor
        if eidx == 1:
            self.smooth_cls_mem[ids] = self.smooth_cls_mem[ids].scatter(1, nys.unsqueeze(1), 0.8) + 0.1
            # self.smooth_cls_mem[ids] += 0.1

        logits = self.to_logits(xs)
        ns_targets = self.smooth_cls_mem[ids]
        meter.Lall = meter.Lall + self.loss_ce_with_targets_(logits, ns_targets, meter=meter)

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        with torch.no_grad():
            nlogits = self.to_logits(xs)
            res = (torch.softmax(nlogits, dim=-1) - torch.softmax(logits, dim=-1)) / ns_targets
            # res = res.scatter(1, res.argsort(dim=-1, descending=True)[:, 2:], 0)

            ns_targets = torch.clamp(ns_targets + res * 0.1, 0, 1)
            self.smooth_cls_mem[ids] = ns_targets

            # ns_max, _ = ns_targets.max(dim=-1, keepdim=True)
            # ns_min, _ = ns_targets.min(dim=-1, keepdim=True)
            # ns_targets = (ns_targets - ns_min)
            # ns_targets = ns_targets / ns_targets.sum(dim=1, keepdims=True)
            targets = tricks.onehot(ys, params.n_classes)

            meter.cm = ((targets * res) > 0).any(dim=-1).float().mean()
            self.smooth_cls_mem[ids] = ns_targets

        self.acc_precise_(logits.argmax(dim=1), ys, meter, name='tacc')
        self.acc_precise_(ns_targets.argmax(dim=1), ys, meter, name='cacc')
        n_mask = nys != ys
        if n_mask.any():
            self.acc_precise_(logits.argmax(dim=1)[n_mask], nys[n_mask], meter, name='nacc')

        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    params = NoisyParams()
    params.optim.args.lr = 0.06
    params.epoch = 400
    params.device = 'cuda:2'
    params.filter_ema = 0.999
    params.smooth = True
    params.mixup = True
    params.ideal_mixup = True
    params.worst_mixup = False
    params.noisy_ratio = 0.8
    params.from_args()
    params.initial()
    trainer = MixupEvalTrainer(params)
    trainer.train()
