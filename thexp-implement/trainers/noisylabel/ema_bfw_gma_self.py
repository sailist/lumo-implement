"""
使用 GMM 区分噪音标签，伪标签部份不太一样了。

cifar10, wideresnet 282
0008.ae2bab25, 0.8, 88.36%


0036.cf5cfe96: mid kl
0038.4c2d15e0: logits kl

cifar10, preresnet11
0052.e9733fc9，0.8，92.72%
0062.75f737f4，0.8, 92.00%
0056.40e62f8b，
0067.653e23eb, 0.8, 90.34%
0088.18ec71da, 0.8, 91.55%
0095.7ce83e0e，0.8，79.08
0096.81459e31，0.8，92.43% 跟 filter_ema 很有关
0056.90046099，0.6, 93.28%
0057.d91f9ad9，0.6，94.68%
0071.3ecb0596, 0.2, 95.27%


0100.1ba37ee1 ?
0101.53d408ba ?


cifar100,
0019.87183d85, 0.8, 61.62% / 63.20%(best)
0032.744a4ea6, 0.8, 59.20% / 60.38%(best)
0038.16660538，0.8，61.01%
0035.c3bbeedf，0.8，62.04%
0040.0c71026d, 0.2, 73.74%，还是干净标签比例太少
0045.d0e3e2fc，0.2，77.81%
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
                                       lr=0.02,
                                       momentum=0.9,
                                       weight_decay=5e-4,
                                       nesterov=True)
        self.mixture_offset = 0

    def initial(self):
        super().initial()
        self.filter_ema_sche = self.SCHE.Log(start=0.99, end=0.999, right=self.epoch)
        self.lr_sche = self.SCHE.Cos(start=self.optim.args.lr,
                                     end=0.00005,
                                     right=self.epoch)
        self.offset_sche = self.SCHE.Cos(start=0,
                                         end=self.mixture_offset,
                                         right=self.epoch // 2)
        self.erl_sche = self.SCHE.Cos(start=1, end=0, right=self.epoch // 2)


class MultiHeadTrainer(datasets.SyntheticNoisyMixin,
                       callbacks.BaseCBMixin, callbacks.callbacks.TrainCallback,
                       models.BaseModelMixin,
                       acc.ClassifyAccMixin,
                       losses.CELoss, losses.MixMatchLoss,
                       losses.IEGLoss, losses.ELRLoss, losses.ClsContrastLoss, losses.SimCLRLoss,
                       tricks.Mixture,
                       Trainer):
    priority = -1

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

    def callbacks(self, params: GlobalParams):
        super().callbacks(params)
        self.hook(self)
        callbacks.callbacks.CUDAErrorHold().hook(self)

    def train_batch(self, eidx, idx, global_step, batch_data, params: GmaParams, device: torch.device):
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
        self.filter_mem[ids] = fweight

        raw_targets = torch.softmax(w_logits, dim=1)

        with torch.no_grad():
            # targets = raw_targets
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
        p_targets = targets
        # n_targets = n_targets * (1 - ratio) + p_targets * ratio
        #
        # p_targets[mask.logical_not()] = p_targets.scatter(-1, top_indices[:, 1:2], ratio)[mask.logical_not()]

        mask = mask.float()
        meter.pm = mask.mean()

        meter.Lall = meter.Lall + self.loss_ce_with_targets_masked_(logits, n_targets,
                                                                    fweight,
                                                                    meter=meter)

        contras_loss, _pos_mask, _neg_mask = self.loss_shade_contrast(torch.stack([w_logits, logits], dim=1),
                                                                      nys, torch.cat([p_targets, p_targets]),
                                                                      weight=torch.cat([fweight, fweight]),
                                                                      meter=meter)
        meter.Lall = meter.Lall + contras_loss * 0.1

        meter.pos_per = _pos_mask.float().sum(dim=-1).mean()
        meter.neg_per = _neg_mask.float().sum(dim=-1).mean()

        # meter.Lall = meter.Lall + self.loss_ce_with_targets_masked_(logits, p_targets,
        #                                                             (1 - fweight) * values * mask,
        #                                                             meter=meter,
        #                                                             name='Lpce') * params.plabel_sche(eidx)

        # lerl = self.loss_elr_(logits, self.plabel_mem[ids], meter=meter) * params.erl_sche(eidx)
        #
        # while lerl < -3:
        #     lerl = lerl / 2
        #
        # meter.Lall = meter.Lall + lerl

        meter.tw = fweight[ys == nys].mean()
        meter.fw = fweight[ys != nys].mean()

        if params.local_filter:
            with torch.no_grad():
                ids_mask = weight_mask.logical_not()
                alpha = params.filter_ema_sche(eidx)
                self.target_mem[ids[ids_mask]] = self.target_mem[ids[ids_mask]] * alpha + label_pred[ids_mask] * \
                                                 (1 - alpha)

                # 将没有参与的逐渐回归到
                # self.target_mem[ids[weight_mask]] = self.target_mem[ids[weight_mask]] * alpha + (1 / params.n_classes) * \
                #                                  (1 - alpha)

        if 'Lall' in meter:
            self.optim.zero_grad()
            meter.Lall.backward()
            self.optim.step()
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

        return meter

    def on_initial_end(self, trainer: Trainer, func, params: NoisyParams, meter: Meter, *args, **kwargs):
        self.target_mem = torch.zeros(self.train_size, device=self.device, dtype=torch.float)
        self.plabel_mem = torch.zeros(self.train_size, params.n_classes, device=self.device, dtype=torch.float)
        self.noisy_cls_mem = torch.zeros(self.train_size, dtype=torch.float, device=self.device)
        self.noisy_cls = torch.zeros(self.train_size, dtype=torch.float, device=self.device)
        self.false_pred_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.filter_mem = torch.zeros(self.train_size, dtype=torch.float, device=self.device)
        # self.pred_mem_size = self.train_size // params.n_classes
        # self.pred_mem = torch.zeros(self.pred_mem_size, params.n_classes, params.epoch,
        #                             dtype=torch.float, device=self.device)
        self.clean_mean_prob = 0
        self.gmm_model = None

        # meter
        self.true_pred_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.loss_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.cls_mem = torch.zeros(self.train_size, params.epoch, device=self.device, dtype=torch.long)

    def on_train_epoch_end(self, trainer: 'NoisyTrainer', func, params: GmaParams, meter: Meter, *args, **kwargs):
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

            noisy_cls = self.bmm_predict(feature, mean=params.feature_mean, offset=params.offset_sche(params.eidx))
            if params.eidx > 1:
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
    params = GmaParams()
    params.large_model = False
    params.use_right_label = False
    params.epoch = 400
    params.device = 'cuda:0'
    params.filter_ema = 0.999
    params.burnin = 2
    params.mix_burnin = 5
    params.with_fc = False
    params.smooth_ratio_sche = params.SCHE.Exp(0.1, 0, right=200)
    params.val_size = 0
    params.targets_ema = 0.7
    params.pred_thresh = 0.85
    params.feature_mean = False
    params.local_filter = True  # 局部的筛选方法
    params.mixt_ema = True  # 是否对 BMM 的预测结果用 EMA 做平滑
    params.lam = 1
    params.from_args()
    if params.dataset == 'cifar100':
        params.optim.args.lr = 0.06

    # for p in params.grid_search('noisy_ratio', [0.2, 0.4, 0.6]):
    #     p.initial()
    #     trainer = MultiHeadTrainer(p)
    #     trainer.train()

    from thextra.hold_memory import memory

    memory(3500, device=params.device).start()

    trainer = MultiHeadTrainer(params)
    trainer.train()

    memory.hold_current()
    # trainer.save_checkpoint()
    # trainer.save_model()
