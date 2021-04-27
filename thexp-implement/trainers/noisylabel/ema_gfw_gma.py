"""
使用 GMM 区分噪音标签

对得到的标记为噪音标签的，分两步削弱权重

然后对 GMM 的参数用 EMA 更新

如何做EMA呢？一开始是想着对参数做EMA，后来发现直接对结果做EMA是一样的，而 <0.5 的，会被视作是正确标签，一开始是预测结果直接处理，
后来变成了直接做EMA，然后再做一次阈值筛选，感觉会好一些（前者会导致有一部分后期被视作是正确标签的，其权重没有办法归于零。）

添加了 right_n


cifar10,
0029.f6b40ce2, 0.6, 94.16%
0013.a149a358，0.8，


cifar100
0011.dcd79029，0.4,
0012.f68977b4，0.8，
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

from trainers import NoisyParams
from trainers.mixin import *


class MultiHeadTrainer(datasets.SyntheticNoisyMixin,
                       callbacks.BaseCBMixin, callbacks.callbacks.TrainCallback,
                       models.BaseModelMixin,
                       acc.ClassifyAccMixin,
                       losses.CELoss, losses.MixMatchLoss, losses.IEGLoss, tricks.Mixture,
                       Trainer):

    def train_batch(self, eidx, idx, global_step, batch_data, params: NoisyParams, device: torch.device):
        meter = Meter()
        ids, xs, axs, ys, nys = batch_data  # type:torch.Tensor
        logits = self.to_logits(axs)

        w_logits = self.to_logits(xs)

        preds = torch.softmax(logits, dim=1).detach()
        label_pred = preds.gather(1, nys.unsqueeze(dim=1)).squeeze()
        # weight = torch.softmax(label_pred - self.target_mem[ids], dim=0)
        weight = label_pred - self.target_mem[ids]
        weight = weight + label_pred * 0.5 / params.n_classes - 0.375 / params.n_classes

        fweight = torch.ones_like(weight)
        if eidx >= params.burnin:
            fweight -= self.noisy_cls[ids]
            fweight[weight <= 0] -= params.gmm_w_sche(eidx)
            fweight = torch.relu(fweight)

        if params.right_n > 0:
            right_mask = ids < params.right_n
            if right_mask.any():
                fweight[right_mask] = 1

        raw_targets = torch.softmax(w_logits, dim=1)

        with torch.no_grad():
            targets = self.plabel_mem[ids] * params.targets_ema + raw_targets * (1 - params.targets_ema)
            self.plabel_mem[ids] = targets
        values, p_labels = targets.max(dim=1)

        mask = values > params.pred_thresh
        if mask.any():
            self.acc_precise_(p_labels[mask], ys[mask], meter, name='pacc')
        mask = mask.float()
        # uda_mask = label_pred > params.uda_sche(eidx)
        # meter.uda = uda_mask.float().mean()

        meter.m0 = (fweight == 0).float().mean()
        meter.m1 = (fweight == 1).float().mean()
        meter.pm = mask.mean()

        meter.Lall = meter.Lall + self.loss_ce_with_masked_(logits, nys, fweight, meter=meter)

        meter.Lall = meter.Lall + self.loss_ce_with_masked_(logits, p_labels,
                                                            (1 - fweight) * mask,
                                                            meter=meter,
                                                            name='Lpce') * params.plabel_sche(eidx)

        meter.tw = fweight[ys == nys].mean()
        meter.fw = fweight[ys != nys].mean()

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
            self.true_pred_mem[ids, eidx - 1] = true_pred
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
        self.noisy_cls_mem = torch.zeros(self.train_size, dtype=torch.float, device=self.device)
        self.noisy_cls = torch.zeros(self.train_size, dtype=torch.float, device=self.device)
        self.false_pred_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.clean_mean_prob = 0
        self.gmm_model = None

        # meter
        self.true_pred_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.loss_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.cls_mem = torch.zeros(self.train_size, params.epoch, device=self.device, dtype=torch.long)

    def on_train_epoch_end(self, trainer: 'NoisyTrainer', func, params: NoisyParams, meter: Meter, *args, **kwargs):
        true_f = os.path.join(self.experiment.test_dir, 'true.pth')
        false_f = os.path.join(self.experiment.test_dir, 'false.pth')
        loss_f = os.path.join(self.experiment.test_dir, 'loss.pth')
        cls_f = os.path.join(self.experiment.test_dir, 'cls.pth')
        if params.eidx % 10 == 0:
            torch.save(self.true_pred_mem, true_f)
            torch.save(self.false_pred_mem, false_f)
            torch.save(self.loss_mem, loss_f)
            torch.save(self.cls_mem, cls_f)

        with torch.no_grad():
            f_mean = self.false_pred_mem[:, max(params.eidx - params.gmm_burnin, 0):params.eidx].mean(
                dim=1).cpu().numpy()
            f_cur = self.false_pred_mem[:, params.eidx - 1].cpu().numpy()
            feature = self.create_feature(f_mean, f_cur)

            noisy_cls = self.gmm_predict(feature)

            true_cls = (self.true_pred_mem == self.false_pred_mem).all(dim=1).cpu().numpy()
            m = self.acc_mixture_(true_cls, noisy_cls)
            meter.update(m)
            self.logger.info(m)

            if params.eidx > params.gmm_burnin:
                self.noisy_cls_mem = torch.tensor(noisy_cls, device=self.device) * 0.1 + self.noisy_cls_mem * 0.9
                self.noisy_cls = self.noisy_cls_mem.clone()
                self.noisy_cls[self.noisy_cls < 0.5] = 0

                # 随时间推移，越难以区分的样本越应该直接挂掉，而不是模糊来模糊去的加权（或许）
                self.noisy_cls[self.noisy_cls >= 0.5].clamp_min_(params.gmm_w_sche(params.eidx))


if __name__ == '__main__':
    # for retry,params in enumerate(params.grid_range(5)):
    # for retry in range(5):
    #     retry = retry + 1
    params = NoisyParams()
    params.right_n = 10
    params.use_right_label = False
    params.optim.args.lr = 0.06
    params.epoch = 300
    params.device = 'cuda:2'
    params.filter_ema = 0.999
    params.burnin = 2
    params.gmm_burnin = 20
    params.targets_ema = 0.3
    # params.tolerance_type = 'exp'
    params.pred_thresh = 0.9
    # params.widen_factor = 10
    params.from_args()
    params.initial()
    trainer = MultiHeadTrainer(params)

    trainer.train()
