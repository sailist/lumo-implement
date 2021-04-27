"""

该Trainer用于验证学习噪音标签的过程中，非噪音分类的概率值的变化规律。

在理想情况下，样本会在噪音分类和正确分类上模糊，徘徊，犹豫不决，这是一种特征。

对比结果：
0011.49476d58，相差不大，会有一些准确率的权衡，但差别不大
# Q.tests('0018.d9661979','0011.49476d58').boards().line('top1_test_')

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

from trainers import NoisyParams, GlobalParams
from trainers.mixin import *


class MixupEvalTrainer(datasets.SyntheticNoisyMixin,
                       callbacks.BaseCBMixin, callbacks.callbacks.TrainCallback,
                       models.BaseModelMixin,
                       acc.ClassifyAccMixin,
                       losses.CELoss, losses.MixMatchLoss, losses.IEGLoss,
                       Trainer):

    def callbacks(self, params: GlobalParams):
        super().callbacks(params)
        self.hook(self)
        self.neg_dict = [[i for i in range(params.n_classes) if i != j] for j in range(params.n_classes)]
        self.pred_mem = torch.zeros(self.train_size, params.n_classes, dtype=torch.float, device=self.device)
        self.sum_mem = torch.zeros(self.train_size, params.n_classes, dtype=torch.float, device=self.device)

    def train_batch(self, eidx, idx, global_step, batch_data, params: NoisyParams, device: torch.device):
        meter = Meter()
        ids, xs, axs, ys, nys = batch_data  # type:torch.Tensor

        logits = self.to_logits(xs)
        meter.Lall = meter.Lall = self.loss_ce_(logits, nys, meter=meter)

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        preds_ = logits.detach().clone()  # torch.softmax(logits, dim=-1)
        # old_preds = self.pred_mem[ids]
        # residual = preds - old_preds

        _mask = torch.arange(params.n_classes).unsqueeze(0).repeat([nys.shape[0], 1]).to(device)
        mask = _mask[_mask == nys.unsqueeze(1)].view(nys.shape[0], -1)

        # residual = residual.gather(1, mask)
        # old_preds = torch.softmax(old_preds.gather(1, mask), dim=-1)
        # preds = torch.softmax(preds_.gather(1, mask), dim=-1)
        # residual = preds - old_preds

        # residual = torch.pow(residual, 2)

        self.acc_precise_(logits.argmax(dim=1), ys, meter, name='tacc')
        n_mask = nys != ys
        if n_mask.any():
            self.acc_precise_(logits.argmax(dim=1)[n_mask], nys[n_mask], meter, name='nacc')

        self.pred_mem[ids] += preds_.detach()

        # self.sum_mem[ids] += residual.detach()
        # if eidx == 1:
        # else:
        #     self.pred_mem[ids] = self.pred_mem[ids] * 0.9 + logits.detach() * 0.1

        # meter.nres = residual[n_mask].mean()
        # meter.nsum = self.sum_mem[ids][n_mask].mean()
        # meter.tres = residual[n_mask.logical_not()].mean()
        # meter.tsum = self.sum_mem[ids][n_mask.logical_not()].mean()
        self.acc_precise_(self.pred_mem[ids].argmax(dim=1), ys, meter, name='sacc')

        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    params = NoisyParams()
    params.optim.args.lr = 0.06
    params.epoch = 400
    params.device = 'cuda:0'
    params.filter_ema = 0.999

    params.mixup = True
    params.ideal_mixup = True
    params.worst_mixup = False
    params.noisy_ratio = 0.8
    params.from_args()
    params.initial()
    trainer = MixupEvalTrainer(params)
    trainer.train()
