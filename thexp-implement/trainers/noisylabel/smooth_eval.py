"""
该Trainer用于通过 label smoothing 验证标签模糊化的作用。

根据 mixup 的测试结果，label smoothing 由于在标签中包含了正确标签，所以理论上会存在比直接训练更好的防止过拟合的特性。
在准确率上可能相差不大。

或许关键不在于某一类的概率取值是否是最大的，而在于某一类的概率在训练过程中是否有增长（尤其是对不属于标签的其他类）
    问题在于如何去评价/量化，设计一个累积/累加函数，和上一次的概率进行距离累加 √


分别验证：
 - 在每个类别上 smoothing
 - 在正确类别上 smoothing
 - 在随机类别上 smoothing
 - 以一定比率在正确类别上 smoothing

之后测试
 - 在 argmax 上以 value 的比例做 smoothing

[ ]测试当 smoothing 的标签随机时，模型最终的准确率是否有提升的现象
    猜测：当扰动很大时，模型只会选择更简单的 pattern 去记住

"""
from collections import defaultdict
from itertools import cycle

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


class SmoothParams(NoisyParams):

    def initial(self):
        super().initial()
        self.lr_sche = self.SCHE.Cos(start=self.optim.args.lr,
                                     end=0.03,
                                     right=self.epoch)


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

    def train_batch(self, eidx, idx, global_step, batch_data, params: NoisyParams, device: torch.device):
        meter = Meter()
        ids, xs, axs, ys, nys = batch_data  # type:torch.Tensor

        logits = self.to_logits(xs)
        if params.smooth:
            if params.smooth_ratio != -1:
                ns_targets = tricks.onehot(nys, params.n_classes)

                if params.smooth_argmax:
                    targets = tricks.onehot(logits.argmax(dim=-1), params.n_classes)
                else:
                    rys = ys.clone()
                    rids = torch.randperm(len(rys))
                    if params.smooth_ratio > 0:
                        rids = rids[:int((len(rids) * params.smooth_ratio))]
                        rys[rids] = torch.randint(0, params.n_classes, [len(rids)], device=device)
                    targets = tricks.onehot(rys, params.n_classes)

                if params.smooth_mixup:
                    l = np.random.beta(0.75, 0.75, size=targets.shape[0])
                    l = np.max([l, 1 - l], axis=0)
                    l = torch.tensor(l, device=device, dtype=torch.float).unsqueeze(1)
                else:
                    l = 0.9

                ns_targets = ns_targets * l + targets * (1 - l)


            else:
                ns_targets = tricks.label_smoothing(tricks.onehot(nys, params.n_classes))
            meter.Lall = meter.Lall + self.loss_ce_with_targets_(logits, ns_targets, meter=meter)
        else:
            meter.Lall = meter.Lall + self.loss_ce_(logits, nys, meter=meter)

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        with torch.no_grad():
            nlogits = self.to_logits(xs)
            res = torch.softmax(nlogits, dim=-1) - torch.softmax(logits, dim=-1)
            meter.nres = res.gather(1, nys.unsqueeze(1)).mean() * 10
            meter.tres = res.gather(1, ys.unsqueeze(1)).mean() * 100
            meter.rres = res.gather(1, torch.randint_like(ys, 0, params.n_classes).unsqueeze(1)).mean() * 100
        self.acc_precise_(logits.argmax(dim=1), ys, meter, name='true_acc')
        n_mask = nys != ys
        if n_mask.any():
            self.acc_precise_(logits.argmax(dim=1)[n_mask], nys[n_mask], meter, name='noisy_acc')

        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    params = SmoothParams()
    params.optim.args.lr = 0.1
    params.epoch = 400
    params.device = 'cuda:0'
    params.filter_ema = 0.999
    params.smooth = True
    params.smooth_ratio = -1  # -1, 或者 0-1 的值
    params.smooth_mixup = False
    params.smooth_argmax = False
    params.mixup = True
    params.ideal_mixup = True
    params.worst_mixup = False
    params.noisy_ratio = 0.8
    params.from_args()
    params.initial()
    trainer = MixupEvalTrainer(params)
    trainer.train()

"""
# 全标签 smoothing
python3 trainers/noisylabel/smooth_eval.py --device=cuda:0 --smooth_ratio=-1 --smooth_mixup=False --smooth_argmax=False

# 全使用 正确标签 smoothing
python3 trainers/noisylabel/smooth_eval.py --device=cuda:0 --smooth_ratio=0 --smooth_mixup=False --smooth_argmax=False

# 使用一半正确标签 smoothing
python3 trainers/noisylabel/smooth_eval.py --device=cuda:0 --smooth_ratio=0.5 --smooth_mixup=False --smooth_argmax=False

# 使用随机标签 smoothing
python3 trainers/noisylabel/smooth_eval.py --device=cuda:1 --smooth_ratio=-1 --smooth_mixup=False --smooth_argmax=False
"""
