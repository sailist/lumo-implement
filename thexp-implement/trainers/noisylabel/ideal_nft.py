"""
尽可能的模拟正常的噪音筛选过程，找出准确率上界

0002.6cc3a7d6：94.17%，模拟的是一开始就能全部区分开时，采用高置信度伪标签的准确率。
0005.c644f497：93.35%，模拟的是一开始没有办法全部分开，在中途全部分开时的准确率
0012.33f90c0d：94.17%，模拟的是一开始没有办法全部分开，在中途全部分开时的准确率，为了公平，在区分的差不多后，以开始时候的学习率重新学习一遍
0011.23fa475b：94.15%，剩 0.03% 没有区分出来

"""

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


class IdealParams(NoisyParams):

    def initial(self):
        super().initial()
        self.lr_sche = self.SCHE.List([
            self.SCHE.Cos(self.optim.args.lr, 0.0001, left=0, right=200),
            self.SCHE.Cos(self.optim.args.lr, 0.0001, left=200, right=params.epoch),
        ])

        self.ideal_nyw_sche = self.SCHE.Linear(1, params.ideal_end, right=200)


class MultiHeadTrainer(datasets.SyntheticNoisyMixin,
                       callbacks.BaseCBMixin, callbacks.callbacks.TrainCallback,
                       models.BaseModelMixin,
                       acc.ClassifyAccMixin,
                       losses.CELoss, losses.MixMatchLoss, losses.IEGLoss, tricks.Mixture,
                       Trainer):
    priority = -1

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)

    def callbacks(self, params: GlobalParams):
        super().callbacks(params)
        self.hook(self)
        callbacks.callbacks.CUDAErrorHold().hook(self)

    def train_batch(self, eidx, idx, global_step, batch_data, params: NoisyParams, device: torch.device):
        meter = Meter()
        ids, xs, axs, ys, nys = batch_data  # type:torch.Tensor
        logits = self.to_logits(axs)

        w_logits = self.to_logits(xs)

        preds = torch.softmax(logits, dim=1).detach()
        label_pred = preds.gather(1, nys.unsqueeze(dim=1)).squeeze()
        # weight = torch.softmax(label_pred - self.target_mem[ids], dim=0)
        fweight = (nys == ys).float()
        fweight[nys != ys] = params.ideal_nyw_sche(eidx)

        raw_targets = torch.softmax(w_logits, dim=1)

        with torch.no_grad():
            targets = self.plabel_mem[ids] * params.targets_ema + raw_targets * (1 - params.targets_ema)
            self.plabel_mem[ids] = targets

        top_values, top_indices = targets.topk(2, dim=-1)
        p_labels = top_indices[:, 0]
        values = top_values[:, 0]

        p_targets = (
            torch.zeros_like(targets)
                .scatter(-1, top_indices[:, 0:1], 0.9)
                .scatter(-1, top_indices[:, 1:2], 0.1)
        )

        mask = values > params.pred_thresh
        if mask.any():
            self.acc_precise_(p_labels[mask], ys[mask], meter, name='pacc')
        mask = mask.float()
        meter.pm = mask.mean()

        n_targets = tricks.onehot(nys, params.n_classes)
        n_targets = n_targets * 0.9 + p_targets * 0.1

        meter.Lall = meter.Lall + self.loss_ce_with_targets_masked_(logits, n_targets,
                                                                    fweight,
                                                                    meter=meter)
        meter.Lall = meter.Lall + self.loss_ce_with_targets_masked_(logits, p_targets,
                                                                    (1 - fweight) * values * mask,
                                                                    meter=meter,
                                                                    name='Lpce') * params.plabel_sche(eidx)

        meter.tw = fweight[ys == nys].mean()
        meter.fw = fweight[ys != nys].mean()

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        self.acc_precise_(logits.argmax(dim=1), ys, meter, name='tacc')
        n_mask = nys != ys
        if n_mask.any():
            self.acc_precise_(logits.argmax(dim=1)[n_mask], nys[n_mask], meter, name='nacc')

        false_pred = targets.gather(1, nys.unsqueeze(dim=1)).squeeze()  # [ys != nys]
        true_pred = targets.gather(1, ys.unsqueeze(dim=1)).squeeze()  # [ys != nys]
        with torch.no_grad():
            self.true_pred_mem[ids, eidx - 1] = true_pred
            self.false_pred_mem[ids, eidx - 1] = false_pred
            self.loss_mem[ids, eidx - 1] = F.cross_entropy(w_logits, nys, reduction='none')
            # mem_mask = ids < self.pred_mem_size - 1
            # self.pred_mem[ids[mem_mask], :, eidx - 1] = targets[ids[mem_mask]]

        if eidx == 1:
            self.cls_mem[ids, 0] = ys
        elif eidx == 2:
            self.cls_mem[ids, 1] = nys
        else:
            self.cls_mem[ids, eidx - 1] = p_labels

        return meter

    def on_initial_end(self, trainer: Trainer, func, params: NoisyParams, meter: Meter, *args, **kwargs):
        self.target_mem = torch.zeros(self.train_size, device=self.device, dtype=torch.float)
        self.plabel_mem = torch.zeros(self.train_size, params.n_classes, device=self.device, dtype=torch.float)
        self.noisy_cls_mem = torch.zeros(self.train_size, dtype=torch.float, device=self.device)
        self.noisy_cls = torch.zeros(self.train_size, dtype=torch.float, device=self.device)
        self.false_pred_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)

        # self.pred_mem_size = self.train_size // params.n_classes
        # self.pred_mem = torch.zeros(self.pred_mem_size, params.n_classes, params.epoch,
        #                             dtype=torch.float, device=self.device)
        self.clean_mean_prob = 0
        self.gmm_model = None

        # meter
        self.true_pred_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.loss_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.cls_mem = torch.zeros(self.train_size, params.epoch, device=self.device, dtype=torch.long)


if __name__ == '__main__':
    # for retry,params in enumerate(params.grid_range(5)):
    # for retry in range(5):
    #     retry = retry + 1
    params = IdealParams()
    params.large_model = False
    params.right_n = 10
    params.use_right_label = False
    params.optim.args.lr = 0.03
    params.epoch = 600
    params.device = 'cuda:2'
    params.filter_ema = 0.999
    params.burnin = 2
    params.mix_burnin = 20
    params.targets_ema = 0.3
    params.pred_thresh = 0.85
    params.feature_mean = False
    params.local_filter = True  # 局部的筛选方法
    params.mixt_ema = True  # 是否对 BMM 的预测结果用 EMA 做平滑
    params.ideal_end = 0
    params.from_args()

    params.initial()
    trainer = MultiHeadTrainer(params)
    if params.ss_pretrain:
        ckpt = torch.load(params.ss_pretrain_fn)
        trainer.model.load_state_dict(ckpt)
    trainer.train()

    trainer.save_checkpoint()
    trainer.save_model()
