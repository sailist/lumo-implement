"""
reimplement of  FixMatch: Simplifying Semi-Supervised Learning with Consistency and Conﬁdence
    https://arxiv.org/abs/2001.07685
"""
if __name__ == '__main__':
    import os
    import sys

    chdir = os.path.dirname(os.path.abspath(__file__))
    chdir = os.path.dirname(chdir)
    chdir = os.path.dirname(chdir)
    sys.path.append(chdir)

import torch

from thexp import Trainer, Meter
from trainers import SemiSupervisedParams

from trainers.mixin import *


class FixMatchParams(SemiSupervisedParams):

    def __init__(self):
        super().__init__()
        self.epoch = 16912  # 62 iter / epoch,
        self.batch_size = 64
        self.optim = self.create_optim('SGD',
                                       lr=0.03,
                                       weight_decay=0.0005,
                                       momentum=0.9,
                                       nesterov=True)
        self.pred_thresh = 0.95
        self.lambda_u = 1
        self.uratio = 7

    def initial(self):
        super().initial()
        if self.dataset == 'cifar100':
            self.optim.args.weight_decay = 0.001


class FixMatchTrainer(callbacks.BaseCBMixin,
                      callbacks.callbacks.TrainCallback,
                      datasets.FixMatchDatasetMixin,
                      models.BaseModelMixin,
                      acc.ClassifyAccMixin,
                      losses.CELoss, losses.FixMatchLoss,
                      Trainer):
    def callbacks(self, params: FixMatchParams):
        super().callbacks(params)
        callbacks.callbacks.ReportSche().hook(self)
        self.hook(self)
        self.logger.info('Model Params', sum(p.numel() for p in self.model.parameters()) / 1e6)

    def initial(self):
        super().initial()
        self.noisy_set = [85, 87, 90, 91, 92, 93, 94, 95, 96, 97, 98]

    def on_train_epoch_end(self, trainer: Trainer, func, params: FixMatchParams, meter: Meter, *args, **kwargs):
        super().on_train_epoch_end(trainer, func, params, meter, *args, **kwargs)

        if len(self.noisy_set) > 0:
            if meter.uwacc.avg > self.noisy_set[0] / 100:
                self.save_model({'noisy_ratio': self.noisy_set[0], 'real': meter.uwacc.avg})
                self.noisy_set = self.noisy_set[1:]

    def train_batch(self, eidx, idx, global_step, batch_data, params: FixMatchParams, device: torch.device):
        super().train_batch(eidx, idx, global_step, batch_data, params, device)
        meter = Meter()

        sup, unsup = batch_data
        xs, ys = sup
        _, un_xs, un_axs, un_ys = unsup

        logits_list = self.to_logits(torch.cat([xs, un_xs, un_axs])).split_with_sizes(
            [xs.shape[0], un_xs.shape[0], un_xs.shape[0]])
        logits, un_w_logits, un_s_logits = logits_list  # type:torch.Tensor

        pseudo_targets = torch.softmax(un_w_logits, dim=-1)
        max_probs, un_pseudo_labels = torch.max(pseudo_targets, dim=-1)
        mask = (max_probs > params.pred_thresh)

        if mask.any():
            self.acc_precise_(un_w_logits.argmax(dim=1)[mask], un_ys[mask], meter, name='umacc')

        meter.Lall = meter.Lall + self.loss_ce_(logits, ys,
                                                meter=meter, name='Lx')

        meter.Lall = meter.Lall + self.loss_ce_with_masked_(un_s_logits, un_pseudo_labels, mask.float(),
                                                            meter=meter, name='Lu') * params.lambda_u

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        meter.masked = mask.float().mean()
        self.acc_precise_(logits.argmax(dim=1), ys, meter)
        self.acc_precise_(un_w_logits.argmax(dim=1), un_ys, meter, name='uwacc')
        self.acc_precise_(un_s_logits.argmax(dim=1), un_ys, meter, name='usacc')

        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    params = FixMatchParams()
    params.n_percls = 4
    params.batch_size = 64
    params.device = 'cuda:2'
    params.from_args()
    trainer = FixMatchTrainer(params)
    trainer.train()
