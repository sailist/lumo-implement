"""
通过 init 正则来尝试是否能忘掉噪音?

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
from torch import nn
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
    priority = -1

    def datasets(self, params: NoisyParams):
        self.rnd.mark('kk')
        params.noisy_type = params.default('symmetric', True)
        params.noisy_ratio = params.default(0.2, True)

        from data.constant import norm_val
        mean, std = norm_val.get(params.dataset, [None, None])
        from data.transforms import ToNormTensor
        toTensor = ToNormTensor(mean, std)
        from data.transforms import Weak
        weak = Weak(mean, std)
        from data.transforms import Strong

        dataset_fn = datasets.datasets[params.dataset]
        train_x, train_y = dataset_fn(True)
        train_y = np.array(train_y)
        from thexp import DatasetBuilder

        from data.noisy import symmetric_noisy
        noisy_y = symmetric_noisy(train_y, params.noisy_ratio, n_classes=params.n_classes)
        clean_mask = (train_y == noisy_y)

        noisy_mask = np.logical_not(clean_mask)
        noisy_mask = np.where(noisy_mask)[0]

        nmask_a = noisy_mask[:len(noisy_mask) // 2]
        nmask_b = noisy_mask[len(noisy_mask) // 2:]

        clean_x, clean_y = train_x[clean_mask], noisy_y[clean_mask]
        clean_true_y = train_y[clean_mask]

        raw_x, raw_true_y = train_x[nmask_a], train_y[nmask_a]
        raw_y = noisy_y[nmask_a]

        change_x, change_true_y, change_y = train_x[nmask_b], train_y[nmask_b], noisy_y[nmask_b]

        first_x, first_y, first_true_y = (
            clean_x + raw_x,
            np.concatenate([clean_y, raw_y]),
            np.concatenate([clean_true_y, raw_true_y]),
        )

        second_x, second_y, second_true_y = (
            clean_x + change_x,
            np.concatenate([clean_y, change_y]),
            np.concatenate([clean_true_y, change_true_y]),
        )

        first_set = (
            DatasetBuilder(first_x, first_true_y)
                .add_labels(first_y, 'noisy_y')
                .toggle_id()
                .add_x(transform=weak)
                .add_y()
                .add_y(source='noisy_y')
        )
        second_set = (
            DatasetBuilder(second_x, second_true_y)
                .add_labels(second_y, 'noisy_y')
                .toggle_id()
                .add_x(transform=weak)
                .add_y()
                .add_y(source='noisy_y')
        )

        self.first_dataloader = first_set.DataLoader(batch_size=params.batch_size,
                                                     num_workers=params.num_workers,
                                                     drop_last=True,
                                                     shuffle=True)

        self.second_dataloader = second_set.DataLoader(batch_size=params.batch_size,
                                                       num_workers=params.num_workers,
                                                       drop_last=True,
                                                       shuffle=True)

        self.second_dataloader = second_set.DataLoader(batch_size=params.batch_size,
                                                       num_workers=params.num_workers,
                                                       drop_last=True,
                                                       shuffle=True)
        self.second = False
        self.regist_databundler(train=self.first_dataloader)
        self.cur_set = 0
        self.to(self.device)

    def change_dataset(self):
        if self.cur_set == 0:
            self.regist_databundler(train=self.second_dataloader)
            self.cur_set = 1
        else:
            self.regist_databundler(train=self.second_dataloader)
            self.cur_set = 0
        self.to(self.device)

    def on_train_epoch_end(self, trainer: Trainer, func, params: NoisyParams, meter: Meter, *args, **kwargs):
        super().on_train_epoch_end(trainer, func, params, meter, *args, **kwargs)

        def init_(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
        if params.eidx == 50:
            self.model.apply(init_)

    def callbacks(self, params: GlobalParams):
        super().callbacks(params)
        self.hook(self)

    def train_batch(self, eidx, idx, global_step, batch_data, params: NoisyParams, device: torch.device):
        meter = Meter()
        ids, xs, ys, nys = batch_data  # type:torch.Tensor

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
    params.echange = False  # 每一个 epoch 都换
    params.change = True
    params.noisy_ratio = 0.8
    params.from_args()
    params.initial()
    trainer = MixupEvalTrainer(params)
    trainer.train()
