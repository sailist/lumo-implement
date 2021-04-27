"""
综合利用样本的全部特征

暂时考虑非标记类别的概率累加值作为区分特征，目前看来不太行，用的可能不是很好

cifar10
0005.8b12fae5, 0.8, f_mean, 57.67% 不行的
0007.da3f14e1, 0.8, f_cur

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
    priority = -1

    def callbacks(self, params: GlobalParams):
        super().callbacks(params)
        self.hook(self)
        callbacks.callbacks.CUDAErrorHold().hook(self)

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)

    def train_batch(self, eidx, idx, global_step, batch_data, params: NoisyParams, device: torch.device):
        meter = Meter()
        ids, xs, axs, ys, nys = batch_data  # type:torch.Tensor
        logits = self.to_logits(axs)

        w_logits = self.to_logits(xs)

        preds = torch.softmax(logits, dim=1).detach()
        label_pred = preds.gather(1, nys.unsqueeze(dim=1)).squeeze()
        # weight = torch.softmax(label_pred - self.target_mem[ids], dim=0)
        if params.local_filter:
            weight = label_pred - self.target_mem[ids]
            weight = weight + label_pred * 0.5 / params.n_classes - 0.375 / params.n_classes
            weight_mask = weight < 0
            meter.tl = weight_mask[ys == nys].float().mean()
            meter.fl = weight_mask[ys != nys].float().mean()

        fweight = torch.ones(w_logits.shape[0], dtype=torch.float, device=device)
        if eidx >= params.burnin:
            if params.local_filter:
                fweight[weight_mask] -= params.gmm_w_sche(eidx)
            fweight -= self.noisy_cls[ids]
            fweight = torch.relu(fweight)

        if params.right_n > 0:
            right_mask = ids < params.right_n
            if right_mask.any():
                fweight[right_mask] = 1

        raw_targets = torch.softmax(w_logits, dim=1)

        with torch.no_grad():
            targets = self.plabel_mem[ids] * params.targets_ema + raw_targets * (1 - params.targets_ema)
            self.plabel_mem[ids] = targets
            self.sum_mem[ids] += raw_targets.detach()
        values, p_labels = targets.max(dim=1)

        mask = values > params.pred_thresh
        if mask.any():
            self.acc_precise_(p_labels[mask], ys[mask], meter, name='pacc')
        mask = mask.float()
        # uda_mask = label_pred > params.uda_sche(eidx)
        # meter.uda = uda_mask.float().mean()

        meter.pm = mask.mean()

        meter.Lall = meter.Lall + self.loss_ce_with_masked_(logits, nys, fweight, meter=meter)

        meter.Lall = meter.Lall + self.loss_ce_with_masked_(logits, p_labels,
                                                            (1 - fweight) * mask,
                                                            meter=meter,
                                                            name='Lpce') * params.plabel_sche(eidx)

        meter.tw = fweight[ys == nys].mean()
        meter.fw = fweight[ys != nys].mean()

        if params.local_filter:
            with torch.no_grad():
                ids_mask = weight_mask.logical_not()
                alpha = params.filter_ema
                if eidx < params.burnin:
                    alpha = 0.99
                self.target_mem[ids[ids_mask]] = self.target_mem[ids[ids_mask]] * alpha + label_pred[ids_mask] * \
                                                 (1 - alpha)

        if 'Lall' in meter:
            self.optim.zero_grad()
            meter.Lall.backward()
            self.optim.step()
        self.acc_precise_(logits.argmax(dim=1), ys, meter, name='tacc')
        n_mask = nys != ys
        if n_mask.any():
            self.acc_precise_(logits.argmax(dim=1)[n_mask], nys[n_mask], meter, name='nacc')

        false_pred = targets.gather(1, nys.unsqueeze(dim=1)).squeeze()  # [ys != nys]
        true_pred = targets.gather(1, ys.unsqueeze(dim=1)).squeeze()  # [ys != nys]
        with torch.no_grad():
            self.true_pred_mem[ids, eidx - 1] = true_pred
            self.false_pred_mem[ids, eidx - 1] = false_pred
            self.loss_mem[ids, eidx - 1] = F.cross_entropy(w_logits, nys, reduction='none')
            # mem_mask = ids < self.pred_mem_size - 1
            # self.pred_mem[ids[mem_mask], :, eidx - 1] = targets[ids[mem_mask]]

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
        self.sum_mem = torch.zeros(self.train_size, params.n_classes, device=self.device, dtype=torch.float)
        self.noisy_cls_mem = torch.zeros(self.train_size, dtype=torch.float, device=self.device)
        self.noisy_cls = torch.zeros(self.train_size, dtype=torch.float, device=self.device)
        self.false_pred_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)

        # self.pred_mem_size = self.train_size // params.n_classes
        # self.pred_mem = torch.zeros(self.pred_mem_size, params.n_classes, params.epoch,
        #                             dtype=torch.float, device=self.device)
        self.clean_mean_prob = 0
        self.gmm_model = None

        # meter
        self.true_pred_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.loss_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.cls_mem = torch.zeros(self.train_size, params.epoch, device=self.device, dtype=torch.long)

    def on_train_epoch_end(self, trainer: 'NoisyTrainer', func, params: NoisyParams, meter: Meter, *args, **kwargs):

        with torch.no_grad():
            f_mean = self.false_pred_mem[:, :params.eidx].mean(
                dim=1).cpu().numpy()
            f_cur = self.false_pred_mem[:, params.eidx - 1].cpu().numpy()
            _mask = torch.arange(params.n_classes).unsqueeze(0).repeat([self.train_size, 1])
            mask = _mask[_mask == self.cls_mem[:, 0].cpu().unsqueeze(1)].view(self.train_size, -1).to(self.device)

            f_alf = self.sum_mem.gather(1, mask).mean(dim=-1).detach().cpu().numpy()
            feature = self.create_feature(f_cur, f_alf)

            noisy_cls = self.bmm_predict(feature)

            true_cls = (self.true_pred_mem == self.false_pred_mem).all(dim=1).cpu().numpy()
            m = self.acc_mixture_(true_cls, noisy_cls)
            meter.update(m)
            self.logger.info(m)

            if params.eidx > params.mix_burnin:
                if params.mixt_ema:
                    self.noisy_cls_mem = torch.tensor(noisy_cls, device=self.device) * 0.1 + self.noisy_cls_mem * 0.9
                    self.noisy_cls = self.noisy_cls_mem.clone()
                else:
                    self.noisy_cls = torch.tensor(noisy_cls, device=self.device)
                # self.noisy_cls[self.noisy_cls < 0.5] = 0

                # 随时间推移，越难以区分的样本越应该直接挂掉，而不是模糊来模糊去的加权（或许）
                # self.noisy_cls[self.noisy_cls >= 0.5].clamp_min_(params.gmm_w_sche(params.eidx))


if __name__ == '__main__':
    # for retry,params in enumerate(params.grid_range(5)):
    # for retry in range(5):
    #     retry = retry + 1
    params = NoisyParams()
    params.right_n = 10
    params.use_right_label = False
    params.optim.args.lr = 0.06
    params.epoch = 500
    params.device = 'cuda:2'
    params.filter_ema = 0.999
    params.burnin = 2
    params.mix_burnin = 20
    params.targets_ema = 0.3
    params.pred_thresh = 0.9
    params.local_filter = True  # 局部的筛选方法
    params.mixt_ema = True  # 是否对 BMM 的预测结果用 EMA 做平滑
    params.from_args()
    params.initial()
    trainer = MultiHeadTrainer(params)
    if params.ss_pretrain:
        ckpt = torch.load(params.ss_pretrain_fn)
        trainer.model.load_state_dict(ckpt)
    trainer.train()

    trainer.save_checkpoint()
    trainer.save_model()
