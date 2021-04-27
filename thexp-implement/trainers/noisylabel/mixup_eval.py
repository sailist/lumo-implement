"""

该Trainer用于通过mixup验证标签模糊化的作用。

在极限情况下，每个噪音标签都可以混合其正确标签的样本，然后和随机mixup、不mixup进行比较

Mixup的结论还可以用来验证标签模糊化的作用。

目前的期望是：最理想情况下的mixup 比普通的 mixup 要好很多。然后希望比直接交叉熵训练干净标签要好（至少要在一个阶段内更优）
最差的情况下，模型应该要和直接交叉熵训练一样，甚至更差，但应该相对稳定性更高，不容易过拟合。因为其同时会始终有一个“混乱”的噪音在扰动。
    不容易过拟合的表现在于，在最高点维持的时间更长

这种方式可以建立 mixup 方法的上界和下界

这样就可以下一步，验证 label smoothing 在噪音标签学习上的作用。之后继续验证不同类别下的概率变化在合适的 scale 后，哪一个更高。
    退一步，如果验证所有类别的变化空间开销太大，那么就固定 mixup 的方式？ 然后每次寻找该类去 mixup，然后观察类别概率变化，
    如果有进步，那么就逐渐定下，如果没有变化，那么就换另一个类别。
    可以每次选择的mixup的标签 +1 ，这样就节省了一部分开销




# 验证大模型下的最高结果和过拟合能力
cifar10, preresnet18 baseline:


cifar10
0018.d9661979，75.x%，no mixup（baseline）
0016.c4239134，best mixup
0023.e842cc65，50% best mixup
0014.be592853，normal mixup
0017.5893c20c，worst mixup

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

        if params.mixup:
            id_dict = defaultdict(list)
            for i, (y, ny) in enumerate(zip(ys, nys)):
                y, ny = int(y), int(ny)
                if y != ny:
                    pass
                    id_dict[int(ny)].append(i)  # 保证添加进去的都是噪音标签
                else:
                    pass
                    # id_dict[int(ny)].append(i)  # 保证添加进去的都是干净标签
            for i in range(params.n_classes):
                id_dict[i] = cycle(id_dict[i])

            if params.ideal_mixup:  # 理想情况下
                re_id = []
                for i, (y, ny) in enumerate(zip(ys, nys)):
                    y, ny = int(y), int(ny)
                    try:
                        # 如果原本标签就是干净标签，那么无所谓混合什么标签（因为干净标签占主导）
                        # 否则，混合进去噪音标签
                        re_id.append(next(id_dict.get(y)))
                    except:
                        re_id.append(np.random.randint(0, len(nys)))

            elif params.worst_mixup:
                re_id = []
                for i, (y, ny) in enumerate(zip(ys, nys)):
                    y, ny = int(y), int(ny)
                    try:
                        # 混合进去非该类的标签，也即 mixup 最差的情况，如果最差情况下依然要优于非 mixup
                        # 那就验证了模型可以从噪音中学习
                        # 否则就说明 mixup 的有效性源自 label smoothing 中的正确部份的标签。
                        neg_cls = np.random.choice(self.neg_dict[y])
                        re_id.append(next(id_dict.get(neg_cls)))
                    except:
                        re_id.append(np.random.randint(0, len(nys)))
            else:
                # re_id = torch.randperm(len(nys)) # 随机

                # 安排固定的 50% 可能
                re_id = []
                rand = np.random.rand(len(nys))
                for i, (y, ny, rand) in enumerate(zip(ys, nys, rand)):
                    y, ny = int(y), int(ny)
                    try:
                        if rand < 0.1:
                            # 混合进去非该类的标签，也即 mixup 最差的情况，如果最差情况下依然要优于非 mixup
                            # 那就验证了模型可以从噪音中学习
                            # 否则就说明 mixup 的有效性源自 label smoothing 中的正确部份的标签。
                            neg_cls = np.random.choice(self.neg_dict[y])
                            re_id.append(next(id_dict.get(neg_cls)))
                        else:
                            re_id.append(next(id_dict.get(y)))
                    except:
                        re_id.append(np.random.randint(0, len(nys)))

            ntargets = tricks.onehot(nys, params.n_classes)
            mixed_input, mixed_target = self.mixup_(xs, ntargets, reids=re_id)

            mixed_logits = self.to_logits(mixed_input)
            meter.Lall = meter.Lall + self.loss_ce_with_targets_(mixed_logits, mixed_target, meter=meter)
            logits = self.predict(xs)
        else:
            logits = self.to_logits(xs)
            meter.Lall = meter.Lall = self.loss_ce_(logits, nys, meter=meter)

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        self.acc_precise_(logits.argmax(dim=1), ys, meter, name='true_acc')
        n_mask = nys != ys
        if n_mask.any():
            self.acc_precise_(logits.argmax(dim=1)[n_mask], nys[n_mask], meter, name='noisy_acc')

        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    params = NoisyParams()
    params.optim.args.lr = 0.06
    params.epoch = 400
    params.device = 'cuda:0'
    params.filter_ema = 0.999

    params.mixup = False
    params.ideal_mixup = True
    params.worst_mixup = False
    params.noisy_ratio = 0.8
    params.widenet282()
    params.from_args()
    params.initial()

    trainer = MixupEvalTrainer(params)
    trainer.train()
