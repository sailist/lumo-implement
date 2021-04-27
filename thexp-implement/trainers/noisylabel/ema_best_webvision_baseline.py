"""
ema_bfw_gma_nft.py 的  webvisoin 版本


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
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


class GmaParams(NoisyParams):

    def __init__(self):
        super().__init__()
        self.optim = self.create_optim('SGD',
                                       lr=0.01,
                                       momentum=0.9,
                                       weight_decay=5e-4,
                                       nesterov=True)
        self.mixture_offset = 0
        self.pretrain = True
        self.dataset = 'webvision'
        self.num_workers = 8
        # self.cut_size = self.batch_size * 60 * 2
        self.cut_size = 3584  # 7168，10752  bs * 64
        self.n_classes = 14

    def initial(self):
        super().initial()
        self.filter_ema_sche = self.SCHE.Log(start=0.99, end=0.99, right=self.epoch)
        self.lr_sche = self.SCHE.Cos(start=self.optim.args.lr,
                                     end=0.00005,
                                     right=self.epoch)
        # self.lr_sche = self.SCHE.Power(self.optim.args.lr, decay_rate=0.2, decay_steps=5)

        # lr1 = self.optim.args.lr
        # lr2 = lr1 / 10
        # self.lr_sche = self.SCHE.List([
        #     self.SCHE.Cos(lr1, lr2, right=25),
        #     self.SCHE.Linear(lr2, lr2 / 10, left=25, right=50),
        #     self.SCHE.Linear(lr2 / 10, lr2 / 100, left=50, right=80),
        # ])
        self.offset_sche = self.SCHE.Cos(start=self.mixture_offset,
                                         end=self.mixture_offset,
                                         right=self.epoch // 2)

        self.n_classes = 50


class MultiHeadTrainer(datasets.WebVisionDatasetMixin,
                       callbacks.BaseCBMixin, callbacks.callbacks.TrainCallback,
                       models.BaseModelMixin,
                       acc.ClassifyAccMixin,
                       losses.CELoss, losses.MixMatchLoss,
                       losses.IEGLoss, tricks.Mixture,
                       Trainer):
    priority = -1

    def to_logits(self, xs):
        return self.model(xs)

    def callbacks(self, params: GlobalParams):
        super().callbacks(params)
        self.hook(self)

    def train_batch(self, eidx, idx, global_step, batch_data, params: GmaParams, device: torch.device):
        meter = Meter()
        ids, xs, axs, nys = batch_data  # type:torch.Tensor

        logits = self.to_logits(xs)
        meter.Lall = meter.Lall + self.loss_ce_(logits, nys, meter=meter)

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        self.acc_precise_(logits.argmax(dim=-1), nys)

        return meter


if __name__ == '__main__':
    # for retry,params in enumerate(params.grid_range(5)):
    # for retry in range(5):
    #     retry = retry + 1
    params = GmaParams()
    params.inception50()
    params.pretrain = False
    params.use_right_label = False
    params.epoch = 80
    params.device = 'cuda:0'
    params.filter_ema = 0.999
    params.burnin = 5
    params.mix_burnin = 1
    params.with_fc = True
    params.ema = True
    params.smooth_ratio_sche = params.SCHE.Exp(0.1, 0, right=200)
    params.val_size = 0
    params.targets_ema = 0.3
    params.pred_thresh = 0.85
    params.feature_mean = False
    params.local_filter = True  # 局部的筛选方法
    params.mixt_ema = True  # 是否对 BMM 的预测结果用 EMA 做平滑
    params.batch_size = 36
    params.warm_epoch = 0
    params.warm_size = 100000
    params.eval_test_per_epoch = (0, 1)
    params.from_args()

    # for p in params.grid_search('noisy_ratio', [0.2, 0.4, 0.6]):
    #     p.initial()
    #     trainer = MultiHeadTrainer(p)
    #     trainer.train()
    from thextra.hold_memory import memory

    memory(10273, device=params.device, hold=False).start()

    trainer = MultiHeadTrainer(params)
    trainer.train()
    #
    memory.hold_current()
