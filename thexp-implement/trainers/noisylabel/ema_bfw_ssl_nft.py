"""
沿用 ema_bfw_gma_nft.py，但模仿 dividemix 的方式用半监督方式重新构建数据集
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

from thexp import Trainer, Meter, Params, DatasetBuilder, DataBundler
from torch.nn import functional as F

from trainers import NoisyParams, GlobalParams
from trainers.mixin import *


class GmaParams(NoisyParams):

    def __init__(self):
        super().__init__()
        self.optim = self.create_optim('SGD',
                                       lr=0.02,
                                       momentum=0.9,
                                       weight_decay=5e-4,
                                       nesterov=True)

    def initial(self):
        super().initial()
        self.filter_ema_sche = self.SCHE.Exp(start=0.999, end=0.99999, right=self.epoch)
        self.lr_sche = self.SCHE.Log(start=self.optim.args.lr,
                                     end=0.00005,
                                     right=400)
        self.w_sche = self.SCHE.Linear(1, 75, right=self.epoch)


class MultiHeadTrainer(datasets.SyntheticNoisyMixin,
                       callbacks.BaseCBMixin, callbacks.callbacks.TrainCallback,
                       models.BaseModelMixin,
                       acc.ClassifyAccMixin,
                       losses.CELoss, losses.MixMatchLoss, losses.IEGLoss, tricks.Mixture,
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

    def change_dataset(self):
        """
        根据当前的 filter_mem，按 thresh 将其分为有监督和无监督
        :return:
        """
        from data.constant import norm_val
        from data.transforms import ToNormTensor, Weak, Strong

        train_x, train_y, noisy_y = self.train_set

        filter_prob = self.filter_mem.cpu().numpy()
        clean_mask = filter_prob > 0.5
        self.logger.info('sup size', clean_mask.sum())
        if clean_mask.all() or not np.logical_not(clean_mask).any():
            return

        clean_ids = np.where(clean_mask)[0]
        noisy_ids = np.where(np.logical_not(clean_mask))[0]

        mean, std = norm_val.get(params.dataset, [None, None])
        weak = Weak(mean, std)
        strong = Strong(mean, std)

        supervised_dataloader = (
            DatasetBuilder(train_x, train_y)
                .add_labels(noisy_y, source_name='ny')
                .toggle_id()
                .add_x(strong)
                .add_y()
                .add_y(source='ny')
                .subset(clean_ids)
                .DataLoader(params.batch_size // 2,
                            shuffle=True,
                            num_workers=0,
                            drop_last=True)
        )

        unsupervised_dataloader = (
            DatasetBuilder(train_x, train_y)
                .add_labels(noisy_y, source_name='ny')
                .add_labels(filter_prob, source_name='nprob')
                .toggle_id()
                .add_x(strong)
                .add_x(strong)
                .add_y()
                .add_y(source='ny')
                .add_y(source='nprob')
                .subset(noisy_ids)
                .DataLoader(params.batch_size // 2,
                            shuffle=True,
                            num_workers=0,
                            drop_last=True)
        )
        if len(supervised_dataloader) > len(unsupervised_dataloader):
            train_dataloader = (
                DataBundler()
                    .add(supervised_dataloader)
                    .cycle(unsupervised_dataloader)
            )
        else:
            train_dataloader = (
                DataBundler()
                    .cycle(supervised_dataloader)
                    .add(unsupervised_dataloader)
            )
        if len(unsupervised_dataloader) == 0 or len(supervised_dataloader) == 0:
            self.ssl_dataloader = None
            return

        self.ssl_dataloader = train_dataloader.zip_mode().to(self.device)
        self.logger.info('ssl loader size', train_dataloader)
        self.ssl_loaderiter = iter(self.ssl_dataloader)

    def warn_up(self, eidx, batch_data, meter: Meter):
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

        fweight = torch.ones(w_logits.shape[0], dtype=torch.float, device=self.device)
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
        p_targets = tricks.onehot(p_labels, params.n_classes)
        # n_targets = n_targets * (1 - ratio) + p_targets * ratio
        # p_targets[mask.logical_not()] = p_targets.scatter(-1, top_indices[:, 1:2], ratio)[mask.logical_not()]

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

    def ssl(self, eidx, batch_data, meter: Meter):
        """"""
        sup, unsup = batch_data
        _ids, xs, _ys, nys = sup
        _uids, uxs1, uxs2, _uys, unys, unprob = unsup

        targets = tricks.onehot(nys, params.n_classes)

        logits = self.to_logits(xs).detach()
        un_logits = self.to_logits(torch.cat([uxs1, uxs2]))
        un_targets = self.label_guesses_(*un_logits.chunk(2))
        un_targets = self.sharpen_(un_targets, params.T)

        mixed_input, mixed_target = self.mixmatch_up_(xs, [uxs1, uxs2], targets, un_targets)

        sup_mixed_target, unsup_mixed_target = mixed_target.detach().split_with_sizes(
            [xs.shape[0], mixed_input.shape[0] - xs.shape[0]])

        sup_mixed_logits, unsup_mixed_logits = self.to_logits(mixed_input).split_with_sizes(
            [xs.shape[0], mixed_input.shape[0] - xs.shape[0]])

        meter.Lall = meter.Lall + self.loss_ce_with_targets_(sup_mixed_logits, sup_mixed_target,
                                                             meter=meter, name='Lx')
        meter.Lall = meter.Lall + self.loss_ce_with_targets_(unsup_mixed_logits, unsup_mixed_target,
                                                             meter=meter, name='Lu') * params.w_sche(eidx)
        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        preds = torch.softmax(torch.cat([logits, un_logits.chunk(2)[0]]), dim=1).detach()
        label_pred = preds.gather(1, torch.cat([nys, unys]).unsqueeze(dim=1)).squeeze()
        ids = torch.cat([_ids, _uids])
        if params.local_filter:
            weight = label_pred - self.target_mem[ids]
            weight = weight + label_pred * 0.5 / params.n_classes - 0.25 / params.n_classes
            weight_mask = weight < 0

        fweight = torch.ones(ids.shape[0], dtype=torch.float, device=self.device)
        if eidx >= params.burnin:
            if params.local_filter:
                fweight[weight_mask] -= params.gmm_w_sche(eidx)
            fweight -= self.noisy_cls[ids]
            fweight = torch.relu(fweight)
        self.filter_mem[ids] = fweight

        return meter

    def train_batch(self, eidx, idx, global_step, batch_data, params: GmaParams, device: torch.device):
        meter = Meter()
        if eidx <= 10 or self.ssl_dataloader is None:
            self.warn_up(eidx, batch_data, meter)
        else:
            try:
                batch_data = next(self.ssl_loaderiter)
            except:
                self.ssl_loaderiter = iter(self.ssl_dataloader)
                batch_data = next(self.ssl_loaderiter)
            self.ssl(eidx, batch_data, meter)
        return meter

    def on_initial_end(self, trainer: Trainer, func, params: NoisyParams, meter: Meter, *args, **kwargs):
        self.target_mem = torch.zeros(self.train_size, device=self.device, dtype=torch.float)
        self.plabel_mem = torch.zeros(self.train_size, params.n_classes, device=self.device, dtype=torch.float)
        self.noisy_cls_mem = torch.zeros(self.train_size, dtype=torch.float, device=self.device)
        self.noisy_cls = torch.zeros(self.train_size, dtype=torch.float, device=self.device)
        self.false_pred_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.filter_mem = torch.zeros(self.train_size, dtype=torch.float, device=self.device)

        self.clean_mean_prob = 0
        self.gmm_model = None

        # meter
        self.true_pred_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.loss_mem = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.cls_mem = torch.zeros(self.train_size, params.epoch, device=self.device, dtype=torch.long)

    def on_train_epoch_begin(self, trainer: Trainer, func, params: Params, *args, **kwargs):
        super().on_train_epoch_begin(trainer, func, params, *args, **kwargs)
        if params.eidx > 5:
            self.change_dataset()

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

            noisy_cls = self.bmm_predict(feature, mean=params.feature_mean, offset=params.mixture_offset)
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


if __name__ == '__main__':
    # for retry,params in enumerate(params.grid_range(5)):
    # for retry in range(5):
    #     retry = retry + 1
    params = GmaParams()
    params.large_model = False
    params.use_right_label = False
    params.epoch = 400
    params.device = 'cuda:2'
    params.filter_ema = 0.999
    params.burnin = 2
    params.mix_burnin = 5
    params.with_fc = False
    params.smooth_ratio_sche = params.SCHE.Exp(0.1, 0, right=200)
    params.val_size = 0
    params.targets_ema = 0.3
    params.pred_thresh = 0.85
    params.feature_mean = False
    params.local_filter = True  # 局部的筛选方法
    params.mixt_ema = True  # 是否对 BMM 的预测结果用 EMA 做平滑
    params.K = 2
    params.mixture_offset = 0
    params.T = 0.5

    params.from_args()
    if params.dataset == 'cifar100':
        params.optim.args.lr = 0.06

    # for p in params.grid_search('noisy_ratio', [0.2, 0.4, 0.6]):
    #     p.initial()
    #     trainer = MultiHeadTrainer(p)
    #     trainer.train()
    trainer = MultiHeadTrainer(params)
    trainer.train()

    # trainer.save_checkpoint()
    # trainer.save_model()
