"""

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

from thexp import Trainer, Meter, Params, AvgMeter
from torch.nn import functional as F

from trainers import NoisyParams, GlobalParams
from trainers.mixin import *


class EvalParams(NoisyParams):

    def __init__(self):
        super().__init__()
        self.eval_mode = self.choice('eval_mode',
                                     'full',  # 纯干净标签 99% 后更换数据集，观察噪音标签的学习速度。
                                     'same_epoch',  # 干净标签训练到相应 epoch 后更换数据集，观察噪音标签的学习速度
                                     'same_acc',  # 干净标签训练到相应 acc(75%) 后更换数据集，观察噪音标签的学习速度
                                     'mix',  # 混合标签训练到 115 个epoch 后只训练噪音标签，观察噪音标签的学习速度
                                     'raw',  # 啥都不干，一直用 mix 训练，观察噪音上限
                                     'direct'  # 直接训练噪音，观察其上限
                                     )


class MixupEvalTrainer(datasets.SyntheticNoisyMixin,
                       callbacks.BaseCBMixin, callbacks.callbacks.TrainCallback,
                       models.BaseModelMixin,
                       acc.ClassifyAccMixin,
                       losses.CELoss, losses.MixMatchLoss, losses.IEGLoss,
                       Trainer):
    priority = -1

    def datasets(self, params: EvalParams):
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

        if params.eval_mode in ['full', 'same_epoch', 'same_acc']:
            first_x, first_y = train_x[clean_mask], noisy_y[clean_mask]
            first_true_y = train_y[clean_mask]
        elif params.eval_mode in ['mix', 'raw', 'direct']:
            first_x, first_y = train_x, noisy_y
            first_true_y = train_y
        else:
            assert False

        second_x, second_true_y = train_x[noisy_mask], train_y[noisy_mask]
        second_y = noisy_y[noisy_mask]

        self.logger.info('noisy acc = {}'.format((first_true_y == first_y).mean()))
        self.logger.info('noisy acc = {}'.format((second_true_y == second_y).mean()))
        self.rnd.shuffle()

        first_set = (
            DatasetBuilder(first_x, first_true_y)
                .add_labels(first_y, 'noisy_y')
                .toggle_id()
                .add_x(transform=weak)
                .add_y()
                .add_y(source='noisy_y')
        )
        noisy_set = (
            DatasetBuilder(second_x, second_true_y)
                .add_labels(second_y, 'noisy_y')
                .toggle_id()
                .add_x(transform=weak)
                .add_y()
                .add_y(source='noisy_y')
        )

        first_dataloader = first_set.DataLoader(batch_size=params.batch_size,
                                                num_workers=params.num_workers,
                                                drop_last=True,
                                                shuffle=True)

        self.second_dataloader = noisy_set.DataLoader(batch_size=params.batch_size,
                                                      num_workers=params.num_workers,
                                                      drop_last=True,
                                                      shuffle=True)
        self.second = False
        self.regist_databundler(train=first_dataloader)
        self.to(self.device)
        if params.eval_mode == 'direct':
            self.change_dataset()

    def change_dataset(self):
        self.regist_databundler(train=self.second_dataloader)
        self.to(self.device)

    def callbacks(self, params: GlobalParams):
        super().callbacks(params)
        self.hook(self)
        self.remove_callback(callbacks.callbacks.LRSchedule)
        self.remove_callback(callbacks.callbacks.EvalCallback)

    def train_batch(self, eidx, idx, global_step, batch_data, params: EvalParams, device: torch.device):
        meter = Meter()
        ids, xs, ys, nys = batch_data  # type:torch.Tensor

        logits = self.to_logits(xs)
        meter.Lall = meter.Lall = self.loss_ce_(logits, nys, meter=meter)

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        self.acc_precise_(logits.argmax(dim=1), ys, meter, name='tacc')
        n_mask = nys != ys
        if n_mask.any():
            self.acc_precise_(logits.argmax(dim=1)[n_mask], nys[n_mask], meter, name='nacc')

        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)

    def on_train_epoch_end(self, trainer: Trainer, func, params: EvalParams, meter: AvgMeter, *args, **kwargs):
        super().on_train_epoch_end(trainer, func, params, meter, *args, **kwargs)
        if self.second and meter.avg.nacc > 0.99:
            self.stop_train()
            meter.change = params.eidx

        if params.eval_mode == 'full':
            if meter.avg.tacc > 0.99:
                self.change_dataset()
                self.second = True
                meter.change = params.eidx
        elif params.eval_mode == 'same_epoch':
            if params.eidx == 115:
                self.change_dataset()
                self.second = True
                meter.change = params.eidx
        elif params.eval_mode == 'same_acc':
            if meter.avg.tacc > 0.75:
                self.change_dataset()
                self.second = True
                meter.change = params.eidx
        elif params.eval_mode == 'mix':
            if params.eidx == 115:
                self.change_dataset()
                self.second = True
                meter.change = params.eidx
        elif params.eval_mode == 'raw':
            pass
        elif params.eval_mode == 'direct':
            pass
        else:
            assert False


if __name__ == '__main__':
    params = EvalParams()
    params.wideresnet282()
    params.optim.args.lr = 0.01
    params.epoch = 1000
    params.device = 'cuda:0'
    params.ema = False
    params.noisy_ratio = 0.8

    params.from_args()

    trainer = MixupEvalTrainer(params)
    trainer.train()
