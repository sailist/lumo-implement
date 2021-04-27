"""
ema_bfw_gma_nft.py 的 clothing1m 版本

0029.1cd605e5: 哪里失败了，结果需要和baseline对比
0033.5ae21dcb: 100k
0038.a2906cac：200k
0045.9e6ce6c7：200k，75.20


0062.710f1bd2：DivideMix 对照组

0101.2a02d7e7：72.5%，
0105.ec87f25c：73.46%
0108.a5f048e8：

0129.c1de7696: 73.82%

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
                                       weight_decay=0.001,
                                       nesterov=True)
        self.mixture_offset = 0
        self.pretrain = True
        self.dataset = 'clothing1m_balance'
        self.num_workers = 8
        # self.cut_size = self.batch_size * 60 * 2
        self.cut_size = 3584  # 7168，10752  bs * 64
        self.n_classes = 14

    def initial(self):
        super().initial()
        self.filter_ema_sche = self.SCHE.Log(start=0.99, end=0.99, right=self.epoch)
        self.lr_sche = self.SCHE.Log(start=self.optim.args.lr,
                                     end=0.00005,
                                     right=self.epoch)
        # self.lr_sche = self.SCHE.Power(self.optim.args.lr, decay_rate=0.2, decay_steps=5)

        # lr1 = self.optim.args.lr
        # lr2 = self.optim.args.lr / 10
        # self.lr_sche = self.SCHE.List([
        #     self.SCHE.Linear(lr1, lr1, right=40),
        #     self.SCHE.Linear(lr2, lr2, left=40, right=self.epoch),
        # ])
        self.offset_sche = self.SCHE.Cos(start=self.mixture_offset,
                                         end=self.mixture_offset,
                                         right=self.epoch // 2)


class MultiHeadTrainer(datasets.Clothing1mDatasetMixin,
                       callbacks.BaseCBMixin, callbacks.callbacks.TrainCallback,
                       models.BaseModelMixin,
                       acc.ClassifyAccMixin,
                       losses.CELoss, losses.MixMatchLoss,
                       losses.IEGLoss, tricks.Mixture,
                       Trainer):
    priority = -1

    # def models(self, params: GlobalParams):
    #     super().models(params)
    #     from torch import nn

    def to_logits(self, xs):
        return self.model(xs)

    def callbacks(self, params: GlobalParams):
        super().callbacks(params)
        self.hook(self)

    def train_batch(self, eidx, idx, global_step, batch_data, params: GmaParams, device: torch.device):
        meter = Meter()
        ids, xs, axs, nys = batch_data  # type:torch.Tensor
        logits = self.to_logits(axs)
        # w_logits = self.to_logits(xs).detach()
        w_logits = logits.detach()

        fweight = self.filter_mem[ids]
        # fweight -= self.noisy_cls[ids]
        fweight = torch.relu(fweight)

        raw_targets = torch.softmax(w_logits, dim=1)

        if eidx == 1:
            targets = raw_targets
        else:
            targets = self.plabel_mem[ids]

        values, p_labels = targets.max(dim=-1)
        mask = values > params.pred_thresh
        if mask.any():
            self.acc_precise_(p_labels[mask], nys[mask], meter, name='pacc')

        n_targets = tricks.onehot(nys, params.n_classes)
        p_targets = tricks.onehot(p_labels, params.n_classes)

        mask = mask.float()
        meter.pm = mask.mean()
        meter.nm = self.noisy_cls[ids].mean()
        meter.fm = fweight.float().mean()

        p_targets[mask.logical_not()] = n_targets[mask.logical_not()]

        # mixed_input, mixed_target = self.mixup_(axs, n_targets, beta=0.75, target_b=p_targets)
        # mixed_logits = self.to_logits(mixed_input)

        # meter.Lall = meter.Lall + self.loss_ce_with_targets_masked_(mixed_logits, mixed_target,
        #                                                             fweight,
        #                                                             meter=meter)

        fmask = fweight > 0.5
        # if eidx > 5:
        #     meter.Lall = meter.Lall + self.loss_ce_with_targets_masked_(logits, p_targets,
        #                                                                 (fmask.logical_not().float()),
        #                                                                 meter=meter,
        #                                                                 name='Lpce') * params.plabel_sche(eidx)

        if fmask.any():
            meter.Lall = meter.Lall + self.loss_ce_with_targets_(logits[fmask], n_targets[fmask],
                                                                 # fweight[fweight > 0.5].float(),
                                                                 meter=meter)

            # if fmask.any():
            self.acc_precise_(w_logits.argmax(dim=-1)[fmask], nys[fmask], meter, name='tacc')


        else:
            self.acc_precise_(w_logits.argmax(dim=-1)[fmask.logical_not()], nys[fmask.logical_not()], meter,
                              name='facc')

        # prior = torch.ones(params.n_classes, device=self.device) / params.n_classes
        # pred_mean = torch.softmax(logits, dim=1).mean(0)
        # meter.Lpen = torch.sum(prior * torch.log(prior / pred_mean))  # penalty
        # meter.Lall = meter.Lall + meter.Lpen

        self.optim.zero_grad()
        if 'Lall' in meter:
            meter.Lall.backward()
            self.optim.step()

        with torch.no_grad():
            preds = torch.softmax(w_logits, dim=1).detach()
            label_pred = preds.gather(1, nys.unsqueeze(dim=1)).squeeze()

            weight = label_pred - self.target_mem[ids]
            weight = weight + label_pred * 0.5 / params.n_classes - 0.25 / params.n_classes
            weight_mask = weight < 0
            meter.wm = weight_mask.float().mean()

            ids_mask = weight_mask.logical_not()
            alpha = params.filter_ema_sche(eidx)
            if eidx == 1:
                alpha = 0.2
            self.target_mem[ids[ids_mask]] = self.target_mem[ids[ids_mask]] * alpha + label_pred[ids_mask] * \
                                             (1 - alpha)

            self.acc_precise_(w_logits.argmax(dim=1), nys, meter, name='acc')

            fweight = torch.ones(nys.shape[0], dtype=torch.float, device=device)
            fweight[weight_mask] -= params.gmm_w_sche(eidx)

            self.local_noisy_cls[ids] = fweight

            raw_targets = torch.softmax(w_logits, dim=1)

            if eidx == 1:
                targets = raw_targets
            else:
                targets = self.plabel_mem[ids] * params.targets_ema + raw_targets * (1 - params.targets_ema)

            self.plabel_mem[ids] = targets

            false_pred = targets.gather(1, nys.unsqueeze(dim=1)).squeeze()  # [nys != nys]
            self.false_pred_mem[ids, eidx - 1] = false_pred

        return meter

    def on_initial_end(self, trainer: Trainer, func, params: NoisyParams, meter: Meter, *args, **kwargs):
        self.target_mem = torch.zeros(self.train_size, device=self.device, dtype=torch.float) / params.n_classes
        self.plabel_mem = torch.ones(self.train_size, params.n_classes, device=self.device,
                                     dtype=torch.float) / params.n_classes
        self.noisy_cls_mem = torch.zeros(self.train_size, dtype=torch.float, device=self.device)
        self.noisy_cls = torch.zeros(self.train_size, dtype=torch.float, device=self.device)
        self.false_pred_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.local_noisy_cls = torch.ones(self.train_size, dtype=torch.float, device=self.device)
        self.filter_mem = torch.ones(self.train_size, dtype=torch.float, device=self.device)

        self.clean_mean_prob = 0
        self.gmm_model = None
        self.logger.info('initial end')

        self.cls_mem = torch.zeros(self.train_size, device=self.device, dtype=torch.long)

    def on_train_epoch_end(self, trainer: 'NoisyTrainer', func, params: GmaParams, meter: Meter, *args, **kwargs):
        filter_mem = os.path.join(self.experiment.test_dir, 'filter_mem_{}.pth'.format(params.eidx))
        local_noisy_cls = os.path.join(self.experiment.test_dir, 'local_noisy_cls_{}.pth'.format(params.eidx))
        noisy_cls = os.path.join(self.experiment.test_dir, 'noisy_cls_{}.pth'.format(params.eidx))
        false_pred_mem = os.path.join(self.experiment.test_dir, 'false_pred_mem.pth')

        torch.save(self.filter_mem, filter_mem)
        torch.save(self.local_noisy_cls, local_noisy_cls)
        torch.save(self.noisy_cls, noisy_cls)
        torch.save(self.false_pred_mem, false_pred_mem)

        if params.eidx == 1:
            for ids, xs, axs, nys in self.train_dataloader:
                self.cls_mem[ids] = nys

            cls_mem = self.cls_mem.detach().cpu().numpy()
            self.nys_ids = []
            for i in range(params.n_classes):
                self.nys_ids.append(np.where(cls_mem == i)[0])

        with torch.no_grad():
            id_ = 0  # max(0, params.eidx - 5)
            self.logger.info('f_mean left id', id_)
            f_mean = self.false_pred_mem[:, id_:params.eidx].mean(
                dim=1).cpu().numpy()
            f_cur = self.false_pred_mem[:, params.eidx - 1].cpu().numpy()
            feature = self.create_feature(f_mean, f_mean)

            # noisy_cls = self.bmm_predict(feature, mean=params.feature_mean, offset=params.offset_sche(params.eidx))
            noisy_cls = np.zeros(len(feature))
            for cls_ids in self.nys_ids:
                noisy_cls[cls_ids] = self.bmm_predict(feature[cls_ids])

            # noisy_cls = self.gmm_predict(feature)
            self.noisy_cls_mem = torch.tensor(noisy_cls, device=self.device) * 0.8 + self.noisy_cls_mem * 0.2

            if params.eidx <= 40 or True:
                self.noisy_cls = torch.tensor(noisy_cls, device=self.device)
                self.noisy_cls_mem = torch.tensor(noisy_cls, device=self.device)
            else:
                self.noisy_cls = self.noisy_cls_mem.clone()

            meter.mm = (self.noisy_cls > 0.5).float().mean()
            self.logger.info('mm raw ratio', (noisy_cls > 0.5).mean(),
                             'mm ratio', meter.mm)
            self.filter_mem = self.local_noisy_cls - self.noisy_cls
            self.logger.info('filter', (self.filter_mem > 0.5).float().mean())
            values, _ = self.plabel_mem.max(dim=-1)
            self.logger.info('plabel filter', (values > params.pred_thresh).float().mean())

            # if (values > params.pred_thresh).float().mean() > 0.6:
            #     self.logger.info('use plabel mask to filter')
            #     tmask = values > params.pred_thresh
            #     self.filter_mem = tmask.float()
            # else:
            tmask = self.filter_mem > 0.5

            tids = torch.where(tmask)[0]
            tids = tids[torch.randperm(len(tids))]
            fids = torch.where(tmask.logical_not())[0]
            fids = fids[torch.randperm(len(fids))]
            sub_ids = torch.cat([tids, fids]).cpu().numpy()
            # sub_ids = torch.randperm(len(self.filter_mem)).numpy()
            self.train_set.subset(sub_ids)


if __name__ == '__main__':
    # for retry,params in enumerate(params.grid_range(5)):
    # for retry in range(5):
    #     retry = retry + 1
    params = GmaParams()
    params.resnet50()
    params.use_right_label = False
    params.epoch = 90
    params.device = 'cuda:0'
    params.filter_ema = 0.999
    params.burnin = 1
    params.mix_burnin = 1
    params.with_fc = True
    params.ema = True
    params.smooth_ratio_sche = params.SCHE.Exp(0.1, 0, right=200)
    params.val_size = 0
    params.targets_ema = 0.3
    params.pred_thresh = 0.85
    params.feature_mean = False
    params.local_filter = True  # 局部的筛选方法
    params.mixt_ema = True  # 是否对 BMM 的预测结果用 EMA 做平滑
    params.batch_size = 56
    params.warm_epoch = 0
    params.warm_size = 100000
    params.eval_test_per_epoch = (0, 1)
    params.from_args()

    # for p in params.grid_search('noisy_ratio', [0.2, 0.4, 0.6]):
    #     p.initial()
    #     trainer = MultiHeadTrainer(p)
    #     trainer.train()
    from thextra.hold_memory import memory

    # memory(10273, device=params.device, hold=False).start()

    trainer = MultiHeadTrainer(params)
    trainer.train()
    #
    memory.hold_current()
    # trainer.save_checkpoint()
    # trainer.save_model()
    # params.initial()
    # params.lr_sche.plot(right=params.epoch)
