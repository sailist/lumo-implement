"""
使用 GMM 区分噪音标签

对得到的标记为噪音标签的，分两步削弱权重

0012.04b89d98, 0.8, 79.73%

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


class MultiHeadTrainer(datasets.SyntheticNoisyMixin,
                       callbacks.BaseCBMixin, callbacks.callbacks.TrainCallback,
                       models.BaseModelMixin,
                       acc.ClassifyAccMixin,
                       losses.CELoss, losses.MixMatchLoss, losses.IEGLoss, tricks.Mixture,
                       Trainer):

    def callbacks(self, params: GlobalParams):
        super().callbacks(params)
        self.hook(self)

    def train_batch(self, eidx, idx, global_step, batch_data, params: NoisyParams, device: torch.device):
        meter = Meter()
        ids, xs, axs, ys, nys = batch_data  # type:torch.Tensor
        logits = self.to_logits(axs)

        w_logits = self.to_logits(xs)

        preds = torch.softmax(logits, dim=1).detach()
        label_pred = preds.gather(1, nys.unsqueeze(dim=1)).squeeze()

        weight = label_pred - self.target_mem[ids]
        if params.tolerance_type == 'linear':
            weight = weight + label_pred * 0.5 / params.n_classes - 0.375 / params.n_classes

        elif params.tolerance_type == 'exp':
            exp_ratio = (torch.exp((self.target_mem[ids] - 1) * 5) - np.exp(-5) * (1 - self.target_mem[ids]))
            weight = weight + (params.tol_start * (1 - exp_ratio) + params.tol_end * exp_ratio)

        elif params.tolerance_type == 'log':
            log_ratio = 1 - torch.exp(-self.target_mem[ids] * 5) + np.exp(-5) * self.target_mem[ids]
            weight = weight + (params.tol_start * (1 - log_ratio) + params.tol_end * log_ratio)

        # weight[self.noisy_cls[ids] == 0] -= params.gmm_sche(eidx)
        weight[self.noisy_cls[ids] == 0] = 0

        raw_targets = torch.softmax(w_logits, dim=1)

        with torch.no_grad():
            targets = self.plabel_mem[ids] * params.targets_ema + raw_targets * (1 - params.targets_ema)
            self.plabel_mem[ids] = targets
        values, p_labels = targets.max(dim=1)

        mask = values > params.pred_thresh
        if mask.any():
            self.acc_precise_(p_labels[mask], ys[mask], meter, name='pacc')

        weight[weight > 0] = 1

        if eidx < params.burnin:
            weight = torch.ones_like(weight)

        meter.m0 = (weight == 0).float().mean()
        meter.pm = mask.float().mean()

        weight_mask = weight.bool()
        pweight_mask = weight_mask.logical_not() & mask
        if weight_mask.any():
            meter.Lall = meter.Lall + self.loss_ce_(logits[weight_mask], nys[weight_mask],
                                                    meter=meter)

        if pweight_mask.any():
            meter.Lall = meter.Lall + self.loss_ce_(logits[pweight_mask], p_labels[pweight_mask],
                                                    meter=meter,
                                                    name='Lpce') * params.plabel_sche(eidx)

        meter.tw = weight[ys == nys].mean()
        meter.fw = weight[ys != nys].mean()

        with torch.no_grad():
            ids_mask = weight.bool()
            alpha = params.filter_ema
            if eidx < params.burnin:
                alpha = 0.99
            self.target_mem[ids[ids_mask]] = self.target_mem[ids[ids_mask]] * alpha + label_pred[ids_mask] * (1 - alpha)

        if 'Lall' in meter:
            self.optim.zero_grad()
            meter.Lall.backward()
            self.optim.step()
        self.acc_precise_(logits.argmax(dim=1), ys, meter, name='true_acc')
        n_mask = nys != ys
        if n_mask.any():
            self.acc_precise_(logits.argmax(dim=1)[n_mask], nys[n_mask], meter, name='noisy_acc')

        false_pred = targets.gather(1, nys.unsqueeze(dim=1)).squeeze()  # [ys != nys]
        true_pred = targets.gather(1, ys.unsqueeze(dim=1)).squeeze()  # [ys != nys]
        with torch.no_grad():
            self.true_pred_mem[ids] = true_pred
            self.false_pred_mem[ids, eidx - 1] = false_pred
            self.loss_mem[ids, eidx - 1] = F.cross_entropy(w_logits, nys, reduction='none')

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
        self.noisy_cls = torch.ones(self.train_size, dtype=torch.float, device=self.device)
        self.false_pred_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.clean_mean_prob = 0
        # meter
        self.true_pred_mem = torch.zeros(self.train_size, dtype=torch.float, device=self.device)
        self.loss_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.cls_mem = torch.zeros(self.train_size, params.epoch, device=self.device, dtype=torch.long)

    def on_train_epoch_end(self, trainer: 'NoisyTrainer', func, params: NoisyParams, meter: Meter, *args, **kwargs):
        true_f = os.path.join(self.experiment.test_dir, 'true.pth')
        false_f = os.path.join(self.experiment.test_dir, 'false.pth')
        loss_f = os.path.join(self.experiment.test_dir, 'loss.pth')
        cls_f = os.path.join(self.experiment.test_dir, 'cls.pth')

        if params.eidx % 10 == 0 or params.eidx == 3:
            torch.save(self.true_pred_mem, true_f)
            torch.save(self.false_pred_mem, false_f)
            torch.save(self.loss_mem, loss_f)
            torch.save(self.cls_mem, cls_f)

        if params.eidx > params.gmm_burnin:
            f_mean = self.false_pred_mem[:, :params.eidx].mean(dim=1).cpu().numpy()
            f_cur = self.false_pred_mem[:, params.eidx - 1].cpu().numpy()
            feature = self.create_feature(f_mean, f_cur)

            noisy_cls = self.gmm_predict(feature, with_prob=False)

            true_cls = (self.true_pred_mem == self.false_pred_mem[:, params.eidx - 1])
            m = self.acc_mixture_(true_cls, noisy_cls)
            meter.update(m)
            self.logger.info(m)

            if params.eidx > params.gmm_burnin:
                self.noisy_cls = torch.tensor(noisy_cls, device=self.device)

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    # for retry,params in enumerate(params.grid_range(5)):
    # for retry in range(5):
    #     retry = retry + 1
    params = NoisyParams()
    params.right_n = 10
    params.optim.args.lr = 0.06
    params.epoch = 400
    params.device = 'cuda:0'
    params.filter_ema = 0.999
    params.burnin = 2
    params.gmm_burnin = 10
    params.targets_ema = 0.3

    params.pred_thresh = 0.9
    params.from_args()
    params.initial()
    trainer = MultiHeadTrainer(params)

    if params.ss_pretrain:
        ckpt = torch.load(params.ss_pretrain_fn)

    trainer.train()
