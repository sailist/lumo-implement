"""
改自最好的 ema_bfw_re_nft.py， 通过同时训练两个模型，其中一个模型会延后 offset 个epoch 开始训练，


目前来看有问题，准确率没有因为一开始而提上去，也就是说有 上限，这受噪音筛选率的影响。
0014.d286e9fe，

0003.37b0d6b9

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


class OffsetParams(NoisyParams):

    def __init__(self):
        super().__init__()
        self.optim = self.create_optim('SGD',
                                       lr=0.02,
                                       momentum=0.9,
                                       weight_decay=5e-4,
                                       nesterov=True)
        self.offset_epoch = 100

    def initial(self):
        super().initial()

        self.lr_sche = self.SCHE.Log(start=self.optim.args.lr,
                                     end=0.00005,
                                     right=self.epoch - self.offset_epoch)
        self.lr_sche2 = self.SCHE.Log(start=self.optim.args.lr,
                                      end=0.00005,
                                      left=self.offset_epoch,
                                      right=self.epoch)


class MultiHeadTrainer(datasets.SyntheticNoisyMixin,
                       callbacks.BaseCBMixin, callbacks.callbacks.TrainCallback,
                       models.BaseModelMixin,
                       acc.ClassifyAccMixin,
                       losses.CELoss, losses.MixMatchLoss, losses.IEGLoss, tricks.Mixture,
                       Trainer):
    priority = -1

    def to_logits(self, xs, with_mid=False):
        raise NotImplementedError("use self.model/self.model2")

    def models(self, params: OffsetParams):
        from trainers.mixin.models import load_backbone
        self.model = load_backbone(params)
        from copy import deepcopy
        self.model2 = deepcopy(self.model)

        from thexp.contrib import ParamGrouper
        from thexp.contrib import EMA

        if params.ema:
            self.ema_model = EMA(self.model2)

        self.optim = params.optim.build(self.model.parameters())
        self.optim2 = params.optim.build(self.model2.parameters())
        self.to(self.device)

    def callbacks(self, params: OffsetParams):
        super().callbacks(params)
        self.hook(self)
        self.remove_callback(callbacks.callbacks.LRSchedule)
        callbacks.EachLRSchedule([
            [self.optim, params.lr_sche],
            [self.optim2, params.lr_sche2],
        ]).hook(self)
        callbacks.callbacks.CUDAErrorHold().hook(self)

    def train_first(self, eidx, model, optim, batch_data, meter: Meter):
        ids, xs, axs, ys, nys = batch_data  # type:torch.Tensor
        logits = model(axs)

        w_logits = model(xs)

        preds = torch.softmax(logits, dim=1).detach()
        label_pred = preds.gather(1, nys.unsqueeze(dim=1)).squeeze()
        # weight = torch.softmax(label_pred - self.target_mem[ids], dim=0)
        if params.local_filter:
            weight = label_pred - self.target_mem[ids]
            weight = weight + label_pred * 0.5 / params.n_classes - 0.25 / params.n_classes
            weight_mask = weight < 0
            meter.tl = weight_mask[ys == nys].float().mean()
            meter.fl = weight_mask[ys != nys].float().mean()

        fweight = torch.ones(w_logits.shape[0], dtype=torch.float, device=self.device)
        if eidx >= params.mix_burnin:
            if params.local_filter:
                fweight[weight_mask] -= params.gmm_w_sche(eidx)
            fweight -= self.noisy_cls[ids]
            fweight = torch.relu(fweight)

        self.filter_mem[ids] = fweight

        raw_targets = torch.softmax(w_logits, dim=1)

        with torch.no_grad():
            targets = self.plabel_mem[ids] * params.targets_ema + raw_targets * (1 - params.targets_ema)
            self.plabel_mem[ids] = targets

        top_values, top_indices = targets.topk(2, dim=-1)
        p_labels = top_indices[:, 0]
        values = top_values[:, 0]

        # ratio = params.smooth_ratio_sche(eidx)

        mask = values > params.pred_thresh
        if mask.any():
            self.acc_precise_(p_labels[mask], ys[mask], meter, name='pacc')

        n_targets = tricks.onehot(nys, params.n_classes)
        p_targets = tricks.onehot(p_labels, params.n_classes)

        mask = mask.float()
        meter.pm = mask.mean()

        meter.Lall = meter.Lall + self.loss_ce_with_targets_masked_(logits, n_targets,
                                                                    fweight,
                                                                    meter=meter)
        meter.Lall = meter.Lall + self.loss_ce_with_targets_masked_(logits, p_targets,
                                                                    (1 - fweight) * values * mask,
                                                                    meter=meter,
                                                                    name='Lpce') * params.plabel_sche(eidx)

        meter.tw = fweight[ys == nys].mean()
        meter.fw = fweight[ys != nys].mean()

        if params.local_filter:
            with torch.no_grad():
                ids_mask = weight_mask.logical_not()
                alpha = params.filter_ema

                self.target_mem[ids[ids_mask]] = self.target_mem[ids[ids_mask]] * alpha + label_pred[ids_mask] * \
                                                 (1 - alpha)

                # 将没有参与的逐渐回归到
                # self.target_mem[ids[weight_mask]] = self.target_mem[ids[weight_mask]] * alpha + (1 / params.n_classes) * \
                #                                  (1 - alpha)

        if 'Lall' in meter:
            optim.zero_grad()
            meter.Lall.backward()
            optim.step()
        self.acc_precise_(w_logits.argmax(dim=1), ys, meter, name='tacc')
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

    def train_second(self, eidx, model, batch_data, meter: Meter):

        ids, xs, axs, ys, nys = batch_data  # type:torch.Tensor
        fweight = self.filter_mem[ids]
        logits = model(axs)

        w_logits = model(xs)

        preds = torch.softmax(logits, dim=1).detach()
        label_pred = preds.gather(1, nys.unsqueeze(dim=1)).squeeze()
        # weight = torch.softmax(label_pred - self.target_mem[ids], dim=0)

        raw_targets = torch.softmax(w_logits, dim=1)

        with torch.no_grad():
            targets = self.plabel_mem_2[ids] * params.targets_ema + raw_targets * (1 - params.targets_ema)
            self.plabel_mem_2[ids] = targets

        top_values, top_indices = targets.topk(2, dim=-1)
        p_labels = top_indices[:, 0]
        values = top_values[:, 0]

        mask = values > params.pred_thresh
        if mask.any():
            self.acc_precise_(p_labels[mask], ys[mask], meter, name='pacc2')

        n_targets = tricks.onehot(nys, params.n_classes)
        p_targets = tricks.onehot(p_labels, params.n_classes)

        mask = mask.float()
        meter.pm = mask.mean()

        p_sche = params.plabel_sche(eidx)

        meter.Lall2 = meter.Lall2 + self.loss_ce_with_targets_masked_(logits, n_targets,
                                                                      fweight)
        meter.Lall2 = meter.Lall2 + self.loss_ce_with_targets_masked_(logits, p_targets,
                                                                      (1 - fweight) * values * mask) * p_sche

        self.acc_precise_(w_logits.argmax(dim=1), ys, meter, name='tacc2')

        self.optim2.zero_grad()
        meter.Lall2.backward()
        self.optim2.step()

    def train_batch(self, eidx, idx, global_step, batch_data, params: OffsetParams, device: torch.device):
        meter = Meter()
        if params.epoch - eidx > params.offset_epoch:
            self.train_first(eidx, self.model, self.optim, batch_data, meter)
        if eidx - params.offset_epoch > 0:
            if params.epoch - eidx > params.offset_epoch:
                self.train_second(eidx, self.model2, batch_data, meter)
            else:
                self.train_first(eidx, self.model2, self.optim2, batch_data, meter)

        return meter

    def on_initial_end(self, trainer: Trainer, func, params: OffsetParams, meter: Meter, *args, **kwargs):
        self.target_mem = torch.zeros(self.train_size, device=self.device, dtype=torch.float)
        self.plabel_mem = torch.zeros(self.train_size, params.n_classes, device=self.device, dtype=torch.float)
        self.plabel_mem_2 = torch.zeros(self.train_size, params.n_classes, device=self.device, dtype=torch.float)
        self.noisy_cls_mem = torch.zeros(self.train_size, dtype=torch.float, device=self.device)
        self.noisy_cls = torch.zeros(self.train_size, dtype=torch.float, device=self.device)
        self.false_pred_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.filter_mem = torch.zeros(self.train_size, dtype=torch.float, device=self.device)
        self.filter_mem_bak = torch.zeros(self.train_size, dtype=torch.float, device=self.device)
        # self.pred_mem_size = self.train_size // params.n_classes
        # self.pred_mem = torch.zeros(self.pred_mem_size, params.n_classes, params.epoch,
        #                             dtype=torch.float, device=self.device)
        self.clean_mean_prob = 0
        self.gmm_model = None

        # meter
        self.true_pred_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.loss_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.cls_mem = torch.zeros(self.train_size, params.epoch, device=self.device, dtype=torch.long)

    def on_train_epoch_end(self, trainer: 'NoisyTrainer', func, params: OffsetParams, meter: Meter, *args, **kwargs):
        true_f = os.path.join(self.experiment.test_dir, 'true.pth')
        false_f = os.path.join(self.experiment.test_dir, 'false.pth')
        loss_f = os.path.join(self.experiment.test_dir, 'loss.pth')
        if params.eidx % 10 == 0:
            torch.save(self.true_pred_mem, true_f)
            torch.save(self.false_pred_mem, false_f)
            torch.save(self.loss_mem, loss_f)

        with torch.no_grad():
            f_mean = self.false_pred_mem[:, :params.eidx].mean(
                dim=1).cpu().numpy()
            f_cur = self.false_pred_mem[:, params.eidx - 1].cpu().numpy()
            feature = self.create_feature(f_mean, f_cur)

            noisy_cls = self.bmm_predict(feature, mean=params.feature_mean)
            if params.eidx > 1:
                self.noisy_cls_mem = torch.tensor(noisy_cls, device=self.device) * 0.1 + self.noisy_cls_mem * 0.9
                true_cls = (self.true_pred_mem == self.false_pred_mem).all(dim=1).cpu().numpy()
                m = self.acc_mixture_(true_cls, self.noisy_cls_mem.cpu().numpy())
                meter.update(m)
                self.logger.info(m)

                if params.eidx > params.mix_burnin:
                    self.noisy_cls = self.noisy_cls_mem.clone()


if __name__ == '__main__':
    # for retry,params in enumerate(params.grid_range(5)):
    # for retry in range(5):
    #     retry = retry + 1
    params = OffsetParams()
    params.large_model = False
    params.use_right_label = False
    params.epoch = 500
    params.device = 'cuda:2'
    params.filter_ema = 0.999

    params.mix_burnin = 10
    params.with_fc = True
    params.smooth_ratio_sche = params.SCHE.Exp(0.1, 0, right=200)
    params.val_size = 0
    params.targets_ema = 0.3
    params.pred_thresh = 0.75
    params.feature_mean = False
    params.local_filter = True  # 局部的筛选方法
    params.mixt_ema = True  # 是否对 BMM 的预测结果用 EMA 做平滑
    params.from_args()

    # for p in params.grid_search('noisy_ratio', [0.2, 0.4, 0.6]):
    trainer = MultiHeadTrainer(params)
    trainer.train()
    #
    # trainer.save_checkpoint()
    # trainer.save_model()
