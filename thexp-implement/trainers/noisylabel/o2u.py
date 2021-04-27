"""
reimplement of 'O2U-Net: A Simple Noisy Label Detection Approach for Deep Neural Networks'
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_O2U-Net_A_Simple_Noisy_Label_Detection_Approach_for_Deep_Neural_ICCV_2019_paper.pdf
    original repository : https://github.com/hjimce/O2U-Net

    Currently can't reproduce the results.
"""

if __name__ == '__main__':
    import os
    import sys

    chdir = os.path.dirname(os.path.abspath(__file__))
    chdir = os.path.dirname(chdir)
    chdir = os.path.dirname(chdir)
    sys.path.append(chdir)

from arch.meta import MetaWideResNet, MetaSGD
import torch
from torch import autograd
from typing import List, Tuple
from thexp import Trainer, Meter, Params, AvgMeter
from trainers import NoisyParams, GlobalParams
from torch.nn import functional as F
from trainers.mixin import *
from arch.meta import MetaModule

from itertools import accumulate


class O2UParams(NoisyParams):

    def __init__(self):
        super().__init__()
        self.stage = [250, 250, 250]
        self.acc_stage = list(accumulate(self.stage))
        self.epoch = sum(self.stage)
        self.batch_size = 100
        self.K = 1
        self.mix_beta = 0.5
        self.T = 0.5
        self.burn_in_epoch = 0
        self.loss_p_percentile = 0.7
        self.optim = self.create_optim('SGD',
                                       lr=0.1,
                                       momentum=0.9,
                                       weight_decay=1e-4,
                                       nesterov=True)
        self.meta_optim = {
            'lr': 0.1,
            'momentum': 0.9,
        }
        self.noisy_ratio = 0.8
        self.forget_rate = None
        self.ema_alpha = 0.999
        self.consistency_factor = 20

        self.widen_factor = 2  # 10 needs multi-gpu

    def initial(self):
        super(O2UParams, self).initial()
        self.lr_sche = self.SCHE.List(
            [
                self.SCHE.Cos(start=0.01, end=0.001, left=0, right=self.stage[0]),
                self.SCHE.PeriodLinear(start=0.01, end=0.001, left=self.stage[1], period=10),
                self.SCHE.Cos(start=0.01, end=0.001, left=self.acc_stage[1], right=self.acc_stage[2]),
            ]
        )
        # self.lr_sche = self.SCHE.Cos(start=0.1, end=0.002, left=0, right=params.epoch)
        if self.forget_rate is None:
            self.forget_rate = self.noisy_ratio


class O2UTrainer(datasets.SyntheticNoisyMixin,
                 callbacks.callbacks.TrainCallback,
                 callbacks.BaseCBMixin,
                 models.BaseModelMixin,
                 acc.ClassifyAccMixin,
                 losses.CELoss, losses.MixMatchLoss, losses.IEGLoss,
                 Trainer):

    def models(self, params: GlobalParams):
        super().models(params)

    def callbacks(self, params: O2UParams):
        super(O2UTrainer, self).callbacks(params)
        self.hook(self)

    def initial(self):
        super().initial()
        self.moving_loss_dic = torch.zeros(50000, device=self.device, dtype=torch.float)

    def on_train_epoch_begin(self, trainer: Trainer, func, params: O2UParams, *args, **kwargs):
        self.globals_loss = 0
        self.example_loss = torch.zeros(50000, device=self.device, dtype=torch.float)
        self.mask = torch.ones(50000, device=self.device, dtype=torch.float)

    def on_train_epoch_end(self, trainer: Trainer, func, params: O2UParams, meter: Meter, *args, **kwargs):
        super().on_train_epoch_end(trainer, func, params, meter, *args, **kwargs)

        # second stage
        if params.eidx > params.acc_stage[1] and params.eidx < params.acc_stage[2]:
            self.example_loss = self.example_loss - self.example_loss.mean()
            self.moving_loss_dic = self.moving_loss_dic + self.example_loss

            ind_1_sorted = torch.argsort(self.moving_loss_dic)
            loss_1_sorted = self.moving_loss_dic[ind_1_sorted]

            remember_rate = 1 - params.forget_rate
            num_remember = int(remember_rate * len(loss_1_sorted))
            self.mask[ind_1_sorted[num_remember:]] = 0

    def train_batch(self, eidx, idx, global_step, batch_data, params: O2UParams, device: torch.device):
        meter = Meter()
        (ids, xs, axs, ys, nys) = batch_data  # type:torch.Tensor
        # ids, xs, nys = batch_data  # type:(torch.Tensor,torch.Tensor,torch.Tensor)
        if eidx > params.acc_stage[1] and eidx < params.acc_stage[2]:
            mask = self.mask[ids] > 0.5
            if not mask.any():
                return meter
            else:
                xs = xs[mask]
                nys = nys[mask]

            meter.tw = mask[nys == ys].float().mean()
            meter.fw = mask[nys != ys].float().mean()

        logits = self.to_logits(xs)
        Lces = F.cross_entropy(logits, nys, reduction='none')

        self.example_loss[ids] = Lces

        self.acc_precise_(logits.argmax(dim=1), ys, meter=meter, name='true_acc')
        self.acc_precise_(logits.argmax(dim=1), nys, meter=meter, name='noisy_acc')
        self.globals_loss = self.globals_loss + Lces.sum().cpu().data.item()
        meter.Lce = Lces.mean()
        self.optim.zero_grad()
        meter.Lce.backward()
        self.optim.step()
        return meter

    def create_metanet(self):
        metanet = MetaWideResNet(num_classes=params.n_classes,
                                 depth=params.depth,
                                 widen_factor=params.widen_factor).to(self.device)
        metanet.load_state_dict(self.model.state_dict())

        meta_sgd = MetaSGD(metanet, **self.params.meta_optim)
        return metanet, meta_sgd

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    params = O2UParams()
    params.device = 'cuda:0'
    params.from_args()
    trainer = O2UTrainer(params)

    trainer.train()
