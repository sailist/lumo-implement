"""
reimplement of 'Distilling Effective Supervision from Severe Label Noise'
other name of this paper(submission withdraw in ICLR2020) is 'IEG: Robust neural net training with severe label noises'
    https://arxiv.org/abs/1911.09781

0001.71d0d7f0，强增广，似乎不太行

0008.79157ed4, 未知原因 崩了
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
import numpy as np


class IEGParams(NoisyParams):

    def __init__(self):
        super().__init__()
        self.epoch = 500
        self.batch_size = 100
        self.K = 1
        self.mix_beta = 0.5
        self.T = 0.5
        self.burn_in_epoch = 0
        self.loss_p_percentile = 0.7
        self.optim = self.create_optim('SGD',
                                       lr=0.03,
                                       momentum=0.9,
                                       weight_decay=1e-4,
                                       nesterov=True)
        self.meta_optim = {
            'lr': 0.1,
            'momentum': 0.9,
        }
        self.noisy_ratio = 0.8
        self.ema_alpha = 0.999
        self.consistency_factor = 20

        self.widen_factor = 2  # 10 needs multi-gpu

    def initial(self):
        super(IEGParams, self).initial()
        self.lr_sche = self.SCHE.Cos(start=self.optim.args.lr, end=0.0001, left=0, right=params.epoch)

        self.epoch_step = self.SCHE.Linear(end=100, right=self.epoch)
        self.init_eps_val = 1. / self.batch_size
        self.grad_eps_init = 0.9  # eps for meta learning init value

        self.plabel_sche = self.SCHE.Cos(1 / self.batch_size, 1, right=self.epoch // 2)
        self.gmm_sche = self.SCHE.Cos(1 / self.n_classes, 0.9, right=self.epoch)
        self.gmm_w_sche = self.SCHE.Cos(0.5, 1, right=self.epoch // 2)

        self.semi_sche = self.SCHE.Cos(1, 0, right=self.epoch // 4)

        self.val_size = 5000
        if self.dataset == 'cifar100':
            self.query_size = 1000
            self.wideresnet28_10()
        elif self.dataset == 'cifar10':
            self.wideresnet282()
            self.query_size = 100

class IEGTrainer(datasets.IEGSyntheticNoisyMixin,
                 callbacks.BaseCBMixin, callbacks.callbacks.TrainCallback,
                 models.BaseModelMixin,
                 acc.ClassifyAccMixin,
                 losses.CELoss, losses.MixMatchLoss, losses.IEGLoss,
                 Trainer):

    def callbacks(self, params: IEGParams):
        super(IEGTrainer, self).callbacks(params)
        self.hook(self)

    def on_initial_end(self, trainer: Trainer, func, params: NoisyParams, meter: Meter, *args, **kwargs):
        self.target_mem = torch.zeros(50000, device=self.device, dtype=torch.float)
        self.plabel_mem = torch.zeros(50000, params.n_classes, device=self.device, dtype=torch.float)
        self.noisy_cls_mem = torch.zeros(50000, dtype=torch.float, device=self.device)
        self.noisy_cls = torch.zeros(50000, dtype=torch.float, device=self.device)
        self.true_pred_mem = torch.zeros(50000, params.epoch, dtype=torch.float, device=self.device)
        self.false_pred_mem = torch.zeros(50000, params.epoch, dtype=torch.float, device=self.device)
        self.clean_mean_prob = 0
        self.gmm_model = None

    def on_train_epoch_end(self, trainer: 'IEGTrainer', func, params: NoisyParams, meter: Meter, *args, **kwargs):
        with torch.no_grad():
            from sklearn import metrics
            f_mean = self.false_pred_mem[:, max(params.eidx - params.gmm_burnin, 0):params.eidx].mean(
                dim=1).cpu().numpy()
            f_cur = self.false_pred_mem[:, params.eidx - 1].cpu().numpy()
            feature = np.stack([f_mean, f_cur], axis=1)

            model = tricks.group_fit(feature)
            noisy_cls = model.predict_proba(feature)[:, 0]  # type:np.ndarray
            if params.eidx == params.gmm_burnin:
                self.noisy_cls_mem = torch.tensor(noisy_cls, device=self.device)

            if params.eidx > params.gmm_burnin:
                self.noisy_cls_mem = torch.tensor(noisy_cls, device=self.device) * 0.1 + self.noisy_cls_mem * 0.9
                self.noisy_cls = self.noisy_cls_mem.clone()
                self.noisy_cls[self.noisy_cls < 0.5] = 0

                # 随时间推移，越难以区分的样本越应该直接挂掉，而不是模糊来模糊去的加权（或许）
                self.noisy_cls[self.noisy_cls >= 0.5].clamp_min_(params.gmm_w_sche(params.eidx))

                true_ncls = (self.true_pred_mem == self.false_pred_mem).all(dim=1).cpu().numpy()
                self.logger.info('gmm accuracy',
                                 metrics.confusion_matrix(true_ncls, self.noisy_cls.cpu().numpy() == 0,
                                                          labels=None, sample_weight=None))

                error_idx = set(np.where(true_ncls == 0)[0])
                self.logger.info('err set', len(set(np.where(noisy_cls != 0)[0]) & error_idx) / len(error_idx))

    def models(self, params: GlobalParams):
        super().models(params)

    def initial(self):
        super().initial()

    def unsupervised_loss(self,
                          xs: torch.Tensor, axs: torch.Tensor,
                          vxs: torch.Tensor, vys: torch.Tensor,
                          logits_lis: List[torch.Tensor],
                          meter: Meter):
        '''create Lub, Lpb, Lkl'''

        logits_lis = [self.logit_norm_(logits) for logits in logits_lis]

        p_target = self.label_guesses_(*logits_lis)
        p_target = self.sharpen_(p_target, params.T)

        re_v_targets = tricks.onehot(vys, params.n_classes)
        mixed_input, mixed_target = self.mixmatch_up_(vxs, [axs], re_v_targets, p_target,
                                                      beta=params.mix_beta)

        mixed_logits = self.to_logits(mixed_input)
        mixed_logits_lis = mixed_logits.split_with_sizes([vxs.shape[0], axs.shape[0]])
        (mixed_v_logits, mixed_nn_logits) = [self.logit_norm_(l) for l in mixed_logits_lis]  # type:torch.Tensor

        # mixed_nn_logits = torch.cat([mixed_n_logits, mixed_an_logits], dim=0)
        mixed_v_targets, mixed_nn_targets = mixed_target.split_with_sizes(
            [mixed_v_logits.shape[0], mixed_nn_logits.shape[0]])

        # Lpβ，验证集作为半监督中的有标签数据集
        meter.Lall = meter.Lall + self.loss_ce_with_targets_(mixed_v_logits, mixed_v_targets,
                                                             meter=meter, name='Lpb') * params.semi_sche(params.eidx)
        # p * Luβ，训练集作为半监督中的无标签数据集
        # meter.Lall = meter.Lall + self.loss_ce_with_targets_(mixed_nn_logits, mixed_nn_targets,
        #                                                      meter=meter, name='Lub') * params.semi_sche(params.eidx)

        # Lkl，对多次增广的一致性损失

        return p_target

    def train_batch(self, eidx, idx, global_step, batch_data, params: IEGParams, device: torch.device):
        meter = Meter()
        meter = Meter()
        train_data, (vxs, vys) = batch_data  # type:List[torch.Tensor],(torch.Tensor,torch.Tensor)

        ids = train_data[0]
        axs = train_data[1]
        xs = torch.cat(train_data[2:2 + params.K])
        ys, nys = train_data[-2:]  # type:torch.Tensor

        w_logits = self.to_logits(xs)
        aug_logits = self.to_logits(axs)

        logits = w_logits.chunk(params.K)[0]
        # logits = aug_logits  # .chunk(params.K)[0]

        w_targets = torch.softmax(w_logits.chunk(params.K)[0], dim=1).detach()
        guess_targets = self.unsupervised_loss(xs, axs, vxs, vys,
                                               logits_lis=[*w_logits.chunk(params.K)],
                                               meter=meter)

        label_pred = guess_targets.gather(1, nys.unsqueeze(dim=1)).squeeze()
        # label_pred = .gather(1, nys.unsqueeze(dim=1)).squeeze()
        weight = label_pred - self.target_mem[ids]
        weight = weight + label_pred * 0.5 / params.n_classes - 0.25 / params.n_classes

        fweight = torch.ones_like(weight)
        if eidx >= params.burnin:
            fweight -= self.noisy_cls[ids]
            fweight[weight <= 0] -= params.gmm_w_sche(eidx)
            fweight = torch.relu(fweight)

        raw_targets = w_targets

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

        n_targets = tricks.onehot(nys, params.n_classes)
        mixed_targets = tricks.elementwise_mul(fweight, n_targets) + tricks.elementwise_mul(1 - fweight, guess_targets)

        # init_eps = torch.ones([guess_targets.shape[0]],
        #                       dtype=torch.float,
        #                       device=self.device) * params.grad_eps_init
        # init_mixed_labels = tricks.elementwise_mul(init_eps,
        #                                            n_targets) + tricks.elementwise_mul(1 - init_eps, guess_targets)

        # loss with initial weight
        meter.Lall = meter.Lall + self.loss_ce_with_targets_(logits, mixed_targets,
                                                             meter=meter, name='Lws')  # Lw*
        # meter.Lall = meter.Lall + self.loss_ce_with_targets_masked_(logits, init_mixed_labels,
        #                                                             mask=fweight * 0.01,
        #                                                             meter=meter, name='Llamda')
        # meter.Lall = meter.Lall + (meter.Lws + meter.Llamda) / 2

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

        self.acc_precise_(w_targets.argmax(dim=1), ys, meter, name='true_acc')
        n_mask = nys != ys
        if n_mask.any():
            self.acc_precise_(w_targets.argmax(dim=1)[n_mask], nys[n_mask], meter, name='noisy_acc')

        false_pred = targets.gather(1, nys.unsqueeze(dim=1)).squeeze()  # [ys != nys]
        true_pred = targets.gather(1, ys.unsqueeze(dim=1)).squeeze()  # [ys != nys]
        with torch.no_grad():
            self.true_pred_mem[ids, eidx - 1] = true_pred
            self.false_pred_mem[ids, eidx - 1] = false_pred

        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    params = IEGParams()
    params.device = 'cuda:0'
    params.from_args()
    params.val_size = 100
    params.burnin = 2
    params.gmm_burnin = 10

    params.targets_ema = 0.3
    # params.tolerance_type = 'exp'
    params.pred_thresh = 0.9
    params.filter_ema = 0.999

    trainer = IEGTrainer(params)

    trainer.train()
