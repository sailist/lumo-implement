"""
上一个实验验证了，无论是学习噪音还是干净标签，模型总能够学到一些有利的特征，从而便于拟合之后的特征。
    - 这或许有些像迁移学习/增量学习。

之后验证，拟合一批噪音样本，对拟合另外一批噪音样本是否有帮助？
    从目前的感觉来看，应该是有帮助的，


此外，还需要试验混合干净标签的噪音样本，在拟合程度达到最大时候，虽然此时噪音样本准确率很低，但实际上模型已经记住了这些噪音样本。


此外，还要证明

要充分的认识到，这样的一个有百万级别参数的模型，是一个复杂系统。

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

    def datasets(self, params: FitEvalParams):
        from thexp import DatasetBuilder
        from data.constant import norm_val
        from data.transforms import ToNormTensor, Weak, Strong

        mean, std = norm_val.get(params.dataset, [None, None])
        weak = Weak(mean, std)

        dataset_fn = datasets.datasets[params.dataset]
        train_x, train_y = dataset_fn(True)
        train_y = np.array(train_y)
        from data.noisy import symmetric_noisy

        train_x, train_y = train_x[:params.train_size], train_y[:params.train_size]
        part_size = params.train_size // 2
        noisy_x1, noisy_true_y1 = train_x[:part_size], train_y[:part_size]
        noisy_x2, noisy_true_y2 = train_x[part_size:], train_y[part_size:]

        noisy_ratio = 0.9
        noisy_y1 = symmetric_noisy(noisy_true_y1, noisy_ratio, n_classes=params.n_classes)
        noisy_y2 = symmetric_noisy(noisy_true_y2, noisy_ratio, n_classes=params.n_classes)

        self.logger.info('noisy dataset ratio: ',
                         (noisy_true_y1 == noisy_y1).mean(),
                         (noisy_true_y2 == noisy_y2).mean())

        noisy_set1 = (
            DatasetBuilder(noisy_x1, noisy_true_y1)
                .add_labels(noisy_y2, source_name='noisy_y')
                .toggle_id()
                .add_x(transform=weak)
                .add_y()
                .add_y(source='noisy_y')
        )
        noisy_set2 = (
            DatasetBuilder(noisy_x2, noisy_true_y2)
                .add_labels(noisy_y2, source_name='noisy_y')
                .toggle_id()
                .add_x(transform=weak)
                .add_y()
                .add_y(source='noisy_y')
        )

        self.noisy_loader1 = noisy_set1.DataLoader(batch_size=params.batch_size,
                                                   num_workers=params.num_workers,
                                                   drop_last=True,
                                                   shuffle=True)
        self.noisy_loader2 = noisy_set2.DataLoader(batch_size=params.batch_size,
                                                   num_workers=params.num_workers,
                                                   drop_last=True,
                                                   shuffle=True)
        self.eval_state = 1 - int(params.eval_mode == 'clean')
        self.toggle_dataset((self.eval_state % 2) == 0)
        self.to(self.device)

    def toggle_dataset(self, clean: bool):
        if clean:
            self.logger.info('toggle noisy1 dataset')
            self.regist_databundler(train=self.noisy_loader1)
        else:
            self.logger.info('toggle noisy2 dataset')
            self.regist_databundler(train=self.noisy_loader2)
        self.to(self.device)

    def callbacks(self, params: GlobalParams):
        super().callbacks(params)
        self.remove_callback(callbacks.callbacks.LRSchedule)
        self.hook(self)

    def train_batch(self, eidx, idx, global_step, batch_data, params: NoisyParams, device: torch.device):
        meter = Meter()
        ids, xs, ys, nys = batch_data  # type:torch.Tensor

        logits = self.to_logits(xs)
        meter.Lall = meter.Lall + self.loss_ce_(logits, nys, meter=meter)
        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        self.acc_precise_(logits.argmax(dim=1), nys, meter, name='acc')
        self.acc_precise_(logits.argmax(dim=1), ys, meter, name='tacc')

        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)

    def on_train_epoch_end(self, trainer: Trainer, func, params: Params, meter: AvgMeter, *args, **kwargs):
        super().on_train_epoch_end(trainer, func, params, meter, *args, **kwargs)
        if meter.avg.acc > 0.99:
            self.logger.info('change_dataset')
            self.eval_state += 1
            self.toggle_dataset((self.eval_state % 2) == 0)
            meter.change = params.eidx

        if self.eval_state == 4:
            self.stop_train()


if __name__ == '__main__':
    params = FitEvalParams()
    params.wideresnet282()
    params.optim.args.lr = 0.02
    params.epoch = 1000
    params.batch_size = 128
    params.eval_mode = 'noisy'
    params.device = 'cuda:3'
    params.ema = False
    params.large_model = False
    params.from_args()
    for p in params.grid_range(20):
        trainer = MixupEvalTrainer(p)
        trainer.train()
