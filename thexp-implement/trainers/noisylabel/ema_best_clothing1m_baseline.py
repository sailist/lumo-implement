"""
ema_bfw_gma_nft.py 的 clothing1m 版本


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
                                       lr=0.002,
                                       momentum=0.9,
                                       weight_decay=5e-4,
                                       nesterov=True)
        self.pretrain = True

    def initial(self):
        super().initial()
        self.lr_sche = self.SCHE.Log(start=self.optim.args.lr,
                                     end=0.00005,
                                     right=self.epoch)


class MultiHeadTrainer(datasets.CleanClothing1mDatasetMixin,
                       callbacks.BaseCBMixin, callbacks.callbacks.TrainCallback,
                       models.BaseModelMixin,
                       acc.ClassifyAccMixin,
                       losses.CELoss, losses.MixMatchLoss, losses.IEGLoss, tricks.Mixture,
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
        logits = self.to_logits(axs)
        w_logits = self.to_logits(xs)


        meter.Lall = meter.Lall + self.loss_ce_(logits, nys, meter=meter)
        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()
        self.acc_precise_(w_logits.argmax(dim=1), nys, meter, name='acc')

        return meter


if __name__ == '__main__':
    # for retry,params in enumerate(params.grid_range(5)):
    # for retry in range(5):
    #     retry = retry + 1
    params = GmaParams()
    params.resnet50()
    params.batch_size = 64
    params.epoch = 40
    params.device = 'cuda:0'
    params.val_size = 0
    params.targets_ema = 0.3
    params.pred_thresh = 0.85
    params.sub_size = 200000

    params.feature_mean = False
    params.local_filter = True  # 局部的筛选方法
    params.mixt_ema = True  # 是否对 BMM 的预测结果用 EMA 做平滑
    params.dataset = 'clothing1m'
    params.batch_size = 48

    params.eval_test_per_epoch = (0, 1)
    params.from_args()

    trainer = MultiHeadTrainer(params)
    trainer.train()
