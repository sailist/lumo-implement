"""
优势：
    对单个样本的噪音分布进行了建模
    使得噪音的标签更加的随机？
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
        self.epoch = 300 * 2
        self.batch_size = 64
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

    def initial(self):
        super(CoRandomParams, self).initial()
        # self.lr_sche = self.SCHE.Cos(start=self.optim.args.lr, end=0.002,
        #                              left=0, right=self.epoch)
        self.lr_sche = self.SCHE.Cos(start=self.optim.args.lr,
                                     end=0.00005,
                                     right=self.epoch)

        if self.dataset == 'cifar10':
            self.warm_up = 10 * 2
        elif self.dataset == 'cifar100':
            self.warm_up = 30 * 2

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
            outputs = self.model(xs) + self.model2(xs)
            return outputs

    def models(self, params: CoRandomParams):
        from trainers.mixin.models import load_backbone
        self.model = load_backbone(params)
        from copy import deepcopy
        self.model2 = deepcopy(self.model)

        from thexp.contrib import ParamGrouper
        noptim = params.optim.args.copy()
        noptim['weight_decay'] = 0

        grouper = ParamGrouper(self.model)
        param_groups = [
            grouper.create_param_group(grouper.kernel_params(with_norm=False), **params.optim.args),
            grouper.create_param_group(grouper.bias_params(with_norm=False), **noptim),
            grouper.create_param_group(grouper.norm_params(), **noptim),
        ]
        self.optim = params.optim.build(param_groups)

        grouper = ParamGrouper(self.model2)
        param_groups = [
            grouper.create_param_group(grouper.kernel_params(with_norm=False), **params.optim.args),
            grouper.create_param_group(grouper.bias_params(with_norm=False), **noptim),
            grouper.create_param_group(grouper.norm_params(), **noptim),
        ]

        self.optim2 = params.optim.build(param_groups)
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
        self.moving_loss_dic = torch.zeros(50000, device=self.device, dtype=torch.float)

        # 用于平滑概率
        self.probs1 = torch.ones(self.train_size, params.n_classes, device=self.device) / params.n_classes
        self.probs2 = torch.ones(self.train_size, params.n_classes, device=self.device) / params.n_classes

        # 用于存放随机游走后的标签
        self.tys = torch.zeros(self.train_size, device=self.device, dtype=torch.long)
        self.fys = torch.zeros(self.train_size, device=self.device, dtype=torch.long)
        self.pys1 = torch.zeros(self.train_size, device=self.device, dtype=torch.long)
        self.pys2 = torch.zeros(self.train_size, device=self.device, dtype=torch.long)

        self.all_loss = [[], []]

    def warmup_model(self, batch_data, model, optim, probs, meter: Meter):
        (ids, xs, axs, ys, nys) = batch_data  # type:torch.Tensor
        optim.zero_grad()

        logits = model(xs)  # type:torch.Tensor
        meter.Lall = meter.Lall + self.loss_ce_(logits, nys, meter=meter, name='Lce')

        # EMA
        meter.Lall.backward()
        optim.step()

        with torch.no_grad():
            # EMA
            logits = model(xs)
            probs[ids] = probs[ids] * 0.3 + torch.softmax(logits, dim=-1) * 0.7

        n_mask = (nys != ys)
        self.acc_precise_(logits.argmax(dim=-1), ys, meter=meter, name='acc')
        self.acc_precise_(logits[n_mask].argmax(dim=-1), nys[n_mask], meter=meter, name='nacc')
        self.acc_precise_(probs[ids].argmax(dim=-1), ys, meter=meter, name='pacc')

    def train_model(self, batch_data, model, optim, pys, probs, meter: Meter):
        (ids, xs, axs, ys, _) = batch_data  # type:torch.Tensor
        nys = pys[ids]

        optim.zero_grad()

        logits = model(xs)  # type:torch.Tensor
        meter.Lall = meter.Lall + self.loss_ce_(logits, nys, meter=meter, name='Lce')
        meter.Lall.backward()
        optim.step()

        with torch.no_grad():
            # EMA
            logits = model(xs)
            probs[ids] = probs[ids] * 0.3 + torch.softmax(logits, dim=-1) * 0.7

        n_mask = (nys != ys)
        self.acc_precise_(logits.argmax(dim=-1), ys, meter=meter, name='acc')
        self.acc_precise_(logits[n_mask].argmax(dim=-1), nys[n_mask], meter=meter, name='nacc')
        self.acc_precise_(probs[ids].argmax(dim=-1), ys, meter=meter, name='pacc')

    def train_batch(self, eidx, idx, global_step, batch_data, params: CoRandomParams, device: torch.device):
        meter = Meter()
        if eidx < params.warm_up:
            if eidx % 2 == 0:
                self.warmup_model(batch_data, self.model, self.optim, self.probs1, meter)
            else:
                self.warmup_model(batch_data, self.model2, self.optim2, self.probs2, meter)
        else:
            if eidx % 2 == 0:
                self.train_model(batch_data, self.model, self.optim, self.pys2, self.probs1, meter)
            else:
                self.train_model(batch_data, self.model2, self.optim2, self.pys1, self.probs2, meter)
        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)

    def on_train_epoch_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        super().on_train_epoch_begin(trainer, func, params, *args, **kwargs)
        if params.eidx == 1:
            for (ids, _, _, ys, nys) in self.train_dataloader:
                self.tys[ids] = ys
                self.fys[ids] = nys
                self.pys1[ids] = nys
                self.pys2[ids] = nys

    def on_train_epoch_end(self, trainer: Trainer, func, params: CoRandomParams, meter: AvgMeter, *args, **kwargs):
        super().on_train_epoch_end(trainer, func, params, meter, *args, **kwargs)

        if params.eidx < params.warm_up:
            return
        else:
            with torch.no_grad():
                clss = np.arange(10)
                # 遍历一遍数据集，随机游走
                if params.eidx % 2 == 0:
                    probs = self.probs1
                    pys = self.pys1
                else:
                    probs = self.probs2
                    pys = self.pys2
                # probs = probs.cpu().numpy()  # type: np.ndarray
                topk = probs.topk(2, dim=-1)[1]
                topk[:, 1] = self.fys
                probs = probs.gather(1, topk).clone()

                # mask low confidence sample have equal prob
                eqp_mask = probs[:, 0] > 0.85
                # probs[eqp_mask] = torch.tensor([0.5, 0.5], device=eqp_mask.device)

                np_prob = (probs / probs.sum(dim=1, keepdim=True)).cpu().numpy()

                np_topk = topk.cpu().numpy()

                for (ids, xs, axs, ys, nys) in self.train_dataloader:
                    # neq_mask = (nys != probs[ids].argmax(dim=-1))
                    # pys[ids] = probs[ids].argmax(dim=-1)

                    for _id in ids[eqp_mask[ids]]:
                        pys[_id] = np.random.choice(np_topk[_id], 1, p=np_prob[_id]).item()
                    # fmask = (ys != nys)
                    # tmask = fmask.logical_not()
                    # meter.pacc = (pys[ids] == ys).float().mean()
                    # meter.ptacc = (pys[ids[tmask]] == ys[tmask]).float().mean()
                tmask = (self.tys == self.fys)
                fmask = (self.tys != self.fys)
                meter.ppacc = (pys == self.tys).float().mean()
                meter.ptacc = (pys[tmask] == self.tys[tmask]).float().mean()
                meter.pfacc = (pys[fmask] == self.tys[fmask]).float().mean()
                meter.ffacc = (pys[fmask] == self.fys[fmask]).float().mean()
                self.logger.info('walk eq', (self.pys1 == self.pys2).float().mean())
                self.logger.info(meter.ppacc, meter.ptacc, meter.pfacc, meter.ffacc)

        if params.eidx == params.warm_up:
            self.save_checkpoint()


if __name__ == '__main__':
    params = CoRandomParams()
    params.num_workers = 4
    params.device = 'cuda:2'
    params.preresnet18()
    params.from_args()

    trainer = CoRandomTrainer(params)
    # 0.8
    # trainer.load_checkpoint(
    #     '/home/share/yanghaozhe/experiments/thexp-implement2.19/co_random.noisylabel/0012.f19d17d2/modules/0000020.ckpt')

    # 0.4
    trainer.load_checkpoint(
        '/home/share/yanghaozhe/experiments/thexp-implement2.19/co_random.noisylabel/0014.db000620/modules/0000020.ckpt')

    trainer.train()
