"""
reimplement of 'Distilling Effective Supervision from Severe Label Noise'
other name of this paper(submission withdraw in ICLR2020) is 'IEG: Robust neural net training with severe label noises'
    https://arxiv.org/abs/1911.09781


"""

if __name__ == '__main__':
    import os
    import sys

    chdir = os.path.dirname(os.path.abspath(__file__))
    chdir = os.path.dirname(chdir)
    chdir = os.path.dirname(chdir)
    chdir = os.path.dirname(chdir)
    sys.path.append(chdir)

from arch.meta import MetaWideResNet, MetaSGD
import torch
from torch import autograd
from typing import List, Tuple
from thexp import Trainer, Meter, Params, AvgMeter
from trainers import SemiSupervisedParams
from torch.nn import functional as F
from trainers.mixin import *
from arch.meta import MetaModule


class MetaSSLParams(SemiSupervisedParams):

    def __init__(self):
        super().__init__()
        self.epoch = 400
        self.batch_size = 100
        self.K = 1
        self.mix_beta = 0.5
        self.T = 0.5
        self.burn_in_epoch = 0
        self.loss_p_percentile = 0.7
        self.optim = self.create_optim('SGD',
                                       lr=0.1,
                                       momentum=0.9,
                                       weight_decay=1e-4,
                                       nesterov=True)
        self.meta_optim = {
            'lr': 0.1,
            'momentum': 0.9,
        }
        self.noisy_ratio = 0.8
        self.ema_alpha = 0.999
        self.consistency_factor = 20

        self.widen_factor = 2  # 10 needs multi-gpu

    def initial(self):
        super(MetaSSLParams, self).initial()
        self.lr_sche = self.SCHE.Cos(start=0.1, end=0.002, left=0, right=params.epoch)

        self.epoch_step = self.SCHE.Linear(end=100, right=self.epoch)
        self.init_eps_val = 1. / self.batch_size
        self.grad_eps_init = 0.9  # eps for meta learning init value


class MetaSSLTrainer(datasets.FixMatchDatasetMixin,
                     callbacks.BaseCBMixin,
                     models.BaseModelMixin,
                     acc.ClassifyAccMixin,
                     losses.CELoss, losses.MixMatchLoss, losses.IEGLoss,
                     Trainer):

    def models(self, params: MetaSSLParams):
        super().models(params)

    def initial(self):
        super().initial()
        self.cls_center = torch.ones(params.n_classes, 128, device=self.device,
                                     dtype=torch.float) / params.n_classes

    def meta_optimizer(self,
                       xs: torch.Tensor,
                       vxs: torch.Tensor, vys: torch.Tensor,
                       meter: Meter):
        """
        先使用训练集数据和初始化权重更新一次，将权重封在 MetaNet 中，随后计算验证集梯度，然后求参数对初始化权重的梯度
        :param xs:
        :param guess_targets:
        :param n_targets:
        :param vxs:
        :param vys:
        :param meter:
        :return:
        """

        metanet, metasgd = self.create_metanet()
        metanet.zero_grad()

        mid_logits = metanet(xs)

        cls_center = autograd.Variable(self.cls_center, requires_grad=True)

        left, right = tricks.cartesian_product(mid_logits, cls_center)
        dist_ = F.pairwise_distance(left, right).reshape(mid_logits.shape[0], -1)
        dist_targets = torch.softmax(dist_, dim=-1)

        dist_loss = self.loss_ce_with_targets_(metanet.fc(mid_logits), dist_targets)

        var_grads = autograd.grad(dist_loss, metanet.params(), create_graph=True)

        metanet.update_params(0.1, var_grads)
        # metasgd.meta_step(var_grads)

        m_v_logits = metanet.fc(metanet(vxs))  # type:torch.Tensor
        meta_loss = self.loss_ce_(m_v_logits, vys)

        # method A
        # grad_meta_vars = autograd.grad(meta_loss, metanet.params(), create_graph=True)
        # grad_target, grad_eps = autograd.grad(
        #     metanet.params(), [cls_center, eps_k], grad_outputs=grad_meta_vars)

        # method B
        grad_target, = autograd.grad(
            meta_loss, [cls_center])

        self.cls_center - grad_target * 0.1

        self.acc_precise_(m_v_logits.argmax(dim=-1), vys, meter=meter, name='Macc')

        meter.LMce = meta_loss.detach()

    def train_batch(self, eidx, idx, global_step, batch_data, params: MetaSSLParams, device: torch.device):
        meter = Meter()

        sup, unsup = batch_data
        xs, ys = sup
        ids, un_xs, un_axs, un_ys = unsup

        mid_logits = self.to_mid(un_xs)

        self.meta_optimizer(un_xs, xs, ys, meter=meter)

        left, right = tricks.cartesian_product(mid_logits, self.cls_center)

        dist_ = F.pairwise_distance(left, right).reshape(mid_logits.shape[0], -1)
        dist_targets = torch.softmax(dist_, dim=-1)
        logits = self.to_logits(un_axs)

        meter.Lall = meter.Lall + self.loss_ce_with_targets_(logits, dist_targets, meter=meter, name='Ldce')  # Lw*

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        self.acc_precise_(logits.argmax(dim=1), un_ys, meter, name='un_acc')
        self.acc_precise_(logits.argmax(dim=1), un_ys, meter, name='un_acc')

        return meter

    def create_metanet(self):
        metanet = MetaWideResNet(num_classes=params.n_classes,
                                 depth=params.depth,
                                 with_fc=False,
                                 widen_factor=params.widen_factor).to(self.device)
        metanet.load_state_dict(self.model.state_dict())

        meta_sgd = MetaSGD(metanet, **self.params.meta_optim)
        return metanet, meta_sgd

    def to_logits(self, xs) -> torch.Tensor:
        return self.mid_to_logits(self.to_mid(xs))

    def mid_to_logits(self, mid) -> torch.Tensor:
        return self.model.fc(mid)

    def to_mid(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    params = MetaSSLParams()
    params.device = 'cuda:0'
    params.from_args()
    params.dataset = 'cifar10'
    params.with_fc = False
    params.uratio = 1
    # frame = inspect.currentframe()
    # gpu_tracker = MemTracker(frame)  # define a GPU tracker
    trainer = MetaSSLTrainer(params)

    trainer.train()
