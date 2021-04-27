"""
1. 0010.1baf5c34, 双头，训练干净标签和干净标签+1，效果几乎等同于训练干净标签。
2. 双头，训练噪音标签和噪音标签+1，

"""

if __name__ == '__main__':
    import os
    import sys

    chdir = os.path.dirname(os.path.abspath(__file__))
    chdir = os.path.dirname(chdir)
    chdir = os.path.dirname(chdir)
    sys.path.append(chdir)

import gc
import torch

from data.constant import norm_val
from thextra.hold_memory import memory
from typing import List, Tuple
from thexp import Trainer, Meter, Params, AvgMeter, DataBundler
from trainers import NoisyParams, GlobalParams
from torch.nn import functional as F
from trainers.mixin import *
from thexp import DatasetBuilder
from data.transforms import ToNormTensor
from data.transforms import Weak
from data.transforms import Strong
import numpy as np

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


class CoRandomParams(NoisyParams):

    def __init__(self):
        super().__init__()
        self.epoch = 300
        self.batch_size = 128
        self.K = 1
        self.mix_beta = 0.5
        self.T = 0.5  # sharpening temperature
        self.burn_in_epoch = 0
        self.loss_p_percentile = 0.7
        self.optim = self.create_optim('SGD',
                                       lr=0.02,
                                       momentum=0.9,
                                       weight_decay=5e-4)
        self.lambda_u = 0  # weight for unsupervised loss
        self.noisy_ratio = 0.8
        self.ema = False
        self.p_threshold = 0.5  # clean probability threshold
        self.noisy_type = 'symmetric'
        self.widen_factor = 2  # 10 needs multi-gpu
        self.with_fc = False
        self.head = 3

    def initial(self):
        super(CoRandomParams, self).initial()
        # self.lr_sche = self.SCHE.Cos(start=self.optim.args.lr, end=0.002,
        #                              left=0, right=self.epoch)
        self.lr_sche = self.SCHE.Cos(start=self.optim.args.lr,
                                     end=0.00005,
                                     right=self.epoch)

        self.nys_sche = self.SCHE.Cos(start=1, end=0.1, right=self.epoch // 2)

        if self.dataset == 'cifar10':
            self.warm_up = 10
        elif self.dataset == 'cifar100':
            self.warm_up = 30 * 2

        if self.architecture == 'PreResnet':
            self.feature_dim = 512

        self.rampup_sche = self.SCHE.Linear(start=0, end=1, left=self.warm_up, right=self.warm_up + 16)


class CoRandomTrainer(datasets.SyntheticNoisyMixin,
                      callbacks.callbacks.TrainCallback,
                      callbacks.BaseCBMixin,
                      models.BaseModelMixin,
                      acc.ClassifyAccMixin,
                      losses.CELoss, losses.MixMatchLoss, losses.MSELoss, losses.MinENTLoss,
                      Trainer):
    priority = -1

    def predict(self, xs) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.to_logits(xs, only_one=True)
            return outputs

    def models(self, params: CoRandomParams):
        from trainers.mixin.models import load_backbone
        from arch.multihead import MultiHead
        self.model = load_backbone(params)
        from copy import deepcopy

        self.head = MultiHead(params.feature_dim, params.n_classes, params.head)

        from thexp.contrib import ParamGrouper
        noptim = params.optim.args.copy()
        noptim['weight_decay'] = 0

        grouper = ParamGrouper(self.model, self.head)
        param_groups = [
            grouper.create_param_group(grouper.kernel_params(with_norm=False), **params.optim.args),
            grouper.create_param_group(grouper.bias_params(with_norm=False), **noptim),
            grouper.create_param_group(grouper.norm_params(), **noptim),
        ]
        self.optim = params.optim.build(param_groups)
        self.to(self.device)

    def callbacks(self, params: CoRandomParams):
        super(CoRandomTrainer, self).callbacks(params)
        self.hook(self)

    def _regist_databundler(self, key, val):
        from torch.utils.data import DataLoader
        assert isinstance(val, (DataBundler, DataLoader))
        if isinstance(val, DataLoader):
            val = DataBundler().add(val)
        if key in self._databundler_dict:
            del self._databundler_dict[key]
        self._databundler_dict[key] = val

    def initial(self):
        super().initial()
        # 用于平滑概率
        self.probs = torch.ones(self.train_size, params.n_classes, device=self.device) / params.n_classes

        # 用于存放随机游走后的标签
        self.tys = torch.zeros(self.train_size, device=self.device, dtype=torch.long)
        self.fys = torch.zeros(self.train_size, device=self.device, dtype=torch.long)
        self.pys = torch.zeros(self.train_size, device=self.device, dtype=torch.long)
        self.pyss = torch.zeros(self.train_size, params.head - 1, device=self.device, dtype=torch.long)

    def warmup_model(self, batch_data, model, head, optim, probs, meter: Meter):
        """最初的测试，观察模型能够从 ys 和 nys 中学到正确的部份"""
        (ids, xs, axs, ys, nys) = batch_data  # type:torch.Tensor
        optim.zero_grad()

        features = model(xs)  # type:torch.Tensor

        n_mask = (ys != nys)

        outs = head(features)[:3]

        meter.Lall = meter.Lall + self.loss_ce_(outs[0], nys, meter=meter, name='Lce')
        meter.Lall.backward()
        optim.step()

        with torch.no_grad():
            # EMA
            probs[ids] = probs[ids] * 0.3 + torch.softmax(outs[0], dim=-1) * 0.7
        #
        # n_mask = (nys != ys)
        self.acc_precise_(outs[0].argmax(dim=-1), ys, meter=meter, name='acc')
        self.acc_precise_(outs[0][n_mask].argmax(dim=-1), nys[n_mask], meter=meter, name='nacc')

    def train_model(self, batch_data, model, head, optim, probs, pys, meter: Meter):
        (ids, xs, axs, ys, nys) = batch_data  # type:torch.Tensor
        optim.zero_grad()
        pys = pys[ids]
        features = model(xs)  # type:torch.Tensor

        n_mask = (ys != nys)

        outs = head(features)[:params.head]

        meter.Lall = meter.Lall + self.loss_ce_(outs[0], nys, meter=meter, name='Lce')

        n_targets = tricks.onehot(nys, params.n_classes)
        p_targets = tricks.onehot(pys, params.n_classes)

        mixed_xs1, mixed_targets1 = self.mixup_(xs, n_targets, target_b=p_targets)
        mixed_xs2, mixed_targets2 = self.mixup_(xs, p_targets, target_b=n_targets)



        for i in range(1, params.head):
            meter.Lall = meter.Lall + self.loss_ce_(outs[i], pys[:, i - 1], meter=meter, name='Lfce{}'.format(i))

        # EMA
        meter.Lall.backward()
        optim.step()

        with torch.no_grad():
            # EMA
            probs[ids] = probs[ids] * 0.3 + torch.softmax(outs[0], dim=-1) * 0.7
        #
        # n_mask = (nys != ys)
        self.acc_precise_(outs[0].argmax(dim=-1), ys, meter=meter, name='acc')
        self.acc_precise_(outs[0][n_mask].argmax(dim=-1), nys[n_mask], meter=meter, name='nacc')
        self.acc_precise_(outs[1].argmax(dim=-1), ys, meter=meter, name='pacc')

    def train_batch(self, eidx, idx, global_step, batch_data, params: CoRandomParams, device: torch.device):
        meter = Meter()
        if eidx <= params.warm_up:
            self.warmup_model(batch_data, self.model, self.head, self.optim, self.probs, meter)
        else:
            self.train_model(batch_data, self.model, self.head, self.optim, self.probs, self.pyss, meter)
        return meter

    def to_mid(self, xs) -> torch.Tensor:
        return self.model(xs)

    def mid_to_logits(self, mid, only_one=False) -> torch.Tensor:
        if only_one:
            return self.head.fc0(mid)
        return self.head(mid)

    def to_logits(self, xs, only_one=False) -> torch.Tensor:
        return self.mid_to_logits(self.to_mid(xs), only_one=only_one)

    def on_train_epoch_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        super().on_train_epoch_begin(trainer, func, params, *args, **kwargs)
        if params.eidx == 1:
            for (ids, _, _, ys, nys) in self.train_dataloader:
                self.tys[ids] = ys
                self.fys[ids] = nys
                self.pys[ids] = nys

    def on_train_epoch_end(self, trainer: Trainer, func, params: CoRandomParams, meter: AvgMeter, *args, **kwargs):
        super().on_train_epoch_end(trainer, func, params, meter, *args, **kwargs)

        if params.eidx < params.warm_up:
            return
        else:
            with torch.no_grad():
                clss = np.arange(10)
                # 遍历一遍数据集，随机游走
                probs = self.probs
                pys = self.pyss
                # probs = probs.cpu().numpy()  # type: np.ndarray
                topk = probs.topk(5, dim=-1)[1]
                # topk[:, 1] = self.fys
                probs = probs.gather(1, topk).clone()

                # mask low confidence sample have equal prob
                # eqp_mask = probs[:, 0] > 0.85
                # probs[eqp_mask] = torch.tensor([0.5, 0.5], device=eqp_mask.device)

                np_prob = (probs / probs.sum(dim=1, keepdim=True)).cpu().numpy()

                np_topk = topk.cpu().numpy()

                for (ids, xs, axs, ys, nys) in self.train_dataloader:

                    for _id in ids:
                        pys[_id] = torch.tensor(np.random.choice(np_topk[_id], params.head - 1, p=np_prob[_id]),
                                                dtype=torch.long,
                                                device=self.device)
                    # fmask = (ys != nys)
                    # tmask = fmask.logical_not()
                    # meter.pacc = (pys[ids] == ys).float().mean()
                    # meter.ptacc = (pys[ids[tmask]] == ys[tmask]).float().mean()
                tmask = (self.tys == self.fys)
                fmask = (self.tys != self.fys)
                meter.ppacc = (pys[:, 0] == self.tys).float().mean()
                meter.ptacc = (pys[:, 0][tmask] == self.tys[tmask]).float().mean()
                meter.pfacc = (pys[:, 0][fmask] == self.tys[fmask]).float().mean()
                meter.ffacc = (pys[:, 0][fmask] == self.fys[fmask]).float().mean()
                self.logger.info(meter.ppacc, meter.ptacc, meter.pfacc, meter.ffacc)

        if params.eidx == params.warm_up:
            self.save_checkpoint()


if __name__ == '__main__':
    params = CoRandomParams()
    params.num_workers = 4
    params.device = 'cuda:2'
    params.noisy_ratio = 0.4
    params.preresnet18()
    params.from_args()

    trainer = CoRandomTrainer(params)
    # trainer.load_checkpoint(
    #     '/home/share/yanghaozhe/experiments/thexp-implement2.19/random_multihead.noisylabel/0024.523b9741/modules/0000010.ckpt')
    trainer.train()
