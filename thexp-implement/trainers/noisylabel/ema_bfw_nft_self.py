"""
加了一部分 类内距离拉近，类间距离拉远的loss，理想情况下，把真实标签输入进去；真实情况下，把筛选后的干净标签和高置信度伪标签输入进去。

理想情况下（0021.84c70524，92.41%）准确率仍然还有差距，这体现在伪标签的准确率上，为什么伪标签的准确率会低，原因暂且不明

问题可能出在，对比损失函数求的是余弦相似度，而根据 []，在空间中噪音点的分布应该处于 outline，两者的约束不同。

但是欧式距离的尝试失败了，模型会直接 nan 掉，这可能是因为 交叉熵训练侧重的是角度信息，而欧式距离则会要求空间距离...
    nan 一定说明某次梯度让矩阵大到 nan 了，猜测原因，可能是由于某个 layer 的不同loss



"""

if __name__ == '__main__':
    import os
    import sys

    chdir = os.path.dirname(os.path.abspath(__file__))
    chdir = os.path.dirname(chdir)
    chdir = os.path.dirname(chdir)
    sys.path.append(chdir)

from torch import autograd

import numpy as np
import torch

from thexp import Trainer, Meter, Params
from torch.nn import functional as F

from trainers import NoisyParams, GlobalParams
from trainers.mixin import *


class SelfParams(NoisyParams):

    def __init__(self):
        super().__init__()
        self.optim = self.create_optim('SGD',
                                       lr=0.3,
                                       momentum=0.9,
                                       weight_decay=1e-3,
                                       nesterov=True)
        self.feature_dim = 128
        self.mid_dim = 128

    def initial(self):
        super().initial()
        self.lr_sche = self.SCHE.Cos(start=self.optim.args.lr,
                                     end=0.001,
                                     right=400)
        self.self_sche = self.SCHE.Cos(start=0.01, end=0.00001, right=200)
        if self.architecture == 'PreResnet':
            self.mid_dim = 512


class MultiHeadTrainer(datasets.SyntheticNoisyMixin,
                       callbacks.BaseCBMixin, callbacks.callbacks.TrainCallback,
                       models.SimCLRMixin,
                       acc.ClassifyAccMixin,
                       losses.ConsistenceLoss,
                       losses.CELoss, losses.MixMatchLoss, losses.IEGLoss, losses.ClsContrastLoss,
                       tricks.Mixture,
                       Trainer):
    priority = -1

    def callbacks(self, params: GlobalParams):
        super().callbacks(params)
        self.hook(self)
        callbacks.callbacks.CUDAErrorHold().hook(self)

    def to_mid(self, xs) -> torch.Tensor:
        return self.model(xs)

    def mid_to_logits(self, xs) -> torch.Tensor:
        return self.model.fc(xs)

    def to_logits(self, xs, with_mid=False):
        mid = self.to_mid(xs)
        logits = self.mid_to_logits(mid)
        if with_mid:
            return mid, logits
        else:
            return logits

    def train_batch(self, eidx, idx, global_step, batch_data, params: SelfParams, device: torch.device):
        meter = Meter()
        ids, xs, axs, ys, nys = batch_data  # type:torch.Tensor
        mid, logits = self.to_logits(axs, with_mid=True)

        w_mid, w_logits = self.to_logits(xs, with_mid=True)

        preds = torch.softmax(logits, dim=1).detach()
        label_pred = preds.gather(1, nys.unsqueeze(dim=1)).squeeze()
        # weight = torch.softmax(label_pred - self.target_mem[ids], dim=0)
        if params.local_filter:
            weight = label_pred - self.target_mem[ids]
            weight = weight + label_pred * 0.5 / params.n_classes - 0.25 / params.n_classes
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
        values, p_labels = targets.max(dim=1)

        mask = values > params.pred_thresh
        if mask.any():
            self.acc_precise_(p_labels[mask], ys[mask], meter, name='pacc')
        mask = mask.float()
        # uda_mask = label_pred > params.uda_sche(eidx)
        # meter.uda = uda_mask.float().mean()

        meter.pm = mask.mean()

        n_targets = tricks.onehot(nys, params.n_classes)
        p_targets = tricks.onehot(p_labels, params.n_classes)

        sys = (tricks.elementwise_mul(n_targets, fweight) +
               tricks.elementwise_mul(p_targets, 1 - fweight)).argmax(dim=-1)
        cls_mask, loss = self.loss_mt_contrast_(mid, pys=sys, meter=meter)
        meter.Lall = meter.Lall + loss * params.self_sche(eidx)

        meter.Lall = meter.Lall + self.loss_ce_with_targets_masked_(logits, n_targets,
                                                                    fweight,
                                                                    meter=meter)
        meter.Lall = meter.Lall + self.loss_ce_with_masked_(logits, p_labels,
                                                            (1 - fweight) * values * mask,
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

                # 将没有参与的逐渐回归到
                # self.target_mem[ids[weight_mask]] = self.target_mem[ids[weight_mask]] * alpha + (1 / params.n_classes) * \
                #                                  (1 - alpha)

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
        true_f = os.path.join(self.experiment.test_dir, 'true.pth')
        false_f = os.path.join(self.experiment.test_dir, 'false.pth')
        loss_f = os.path.join(self.experiment.test_dir, 'loss.pth')
        cls_f = os.path.join(self.experiment.test_dir, 'cls.pth')
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
            self.noisy_cls_mem = torch.tensor(noisy_cls, device=self.device) * 0.1 + self.noisy_cls_mem * 0.9
            true_cls = (self.true_pred_mem == self.false_pred_mem).all(dim=1).cpu().numpy()
            m = self.acc_mixture_(true_cls, self.noisy_cls_mem.cpu().numpy())
            meter.update(m)
            self.logger.info(m)

            if params.eidx > params.mix_burnin:
                if params.mixt_ema:
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
    params = SelfParams()
    params.large_model = False
    params.right_n = 10
    params.use_right_label = False
    params.optim.args.lr = 0.01
    params.epoch = 500
    params.device = 'cuda:2'
    params.filter_ema = 0.999
    params.burnin = 10
    params.mix_burnin = 20
    params.targets_ema = 0.3
    params.pred_thresh = 0.9

    params.with_fc = False
    params.feature_mean = False
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
