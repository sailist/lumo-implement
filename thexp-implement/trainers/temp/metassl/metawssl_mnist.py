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

from arch.meta import MetaWideResNet, MetaSGD, MetaLeNet
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
                                       lr=0.03,
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


class MetaSSLTrainer(datasets.FixMatchMNISTDatasetMixin,
                     callbacks.BaseCBMixin,
                     models.BaseModelMixin,
                     acc.ClassifyAccMixin,
                     losses.CELoss, losses.MixMatchLoss, losses.IEGLoss,
                     Trainer):

    def models(self, params: MetaSSLParams):
        super().models(params)

    def initial(self):
        super().initial()

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

        logits = metanet(xs)

        weight_0 = torch.ones(xs.shape[0],
                              dtype=torch.float,
                              device=self.device) * params.init_eps_val
        weight_0 = autograd.Variable(weight_0, requires_grad=True)

        dist_loss = self.loss_ce_with_masked_(logits, logits.argmax(dim=1), weight_0)

        var_grads = autograd.grad(dist_loss, metanet.params(), create_graph=True)

        # metanet.update_params(0.1, var_grads)
        metasgd.meta_step(var_grads)

        m_v_logits = metanet(vxs)  # type:torch.Tensor
        meta_loss = self.loss_ce_(m_v_logits, vys)

        # method A
        # grad_meta_vars = autograd.grad(meta_loss, metanet.params(), create_graph=True)
        # grad_target, grad_eps = autograd.grad(
        #     metanet.params(), [cls_center, eps_k], grad_outputs=grad_meta_vars)

        # method B
        grad_target, = autograd.grad(
            meta_loss, [weight_0])

        raw_weight = weight_0 - grad_target
        raw_weight = raw_weight - params.init_eps_val
        unorm_weight = raw_weight.clamp_min(0)
        norm_c = unorm_weight.sum()
        weight = torch.div(unorm_weight, norm_c + 0.00001).detach()

        self.acc_precise_(m_v_logits.argmax(dim=-1), vys, meter=meter, name='Macc')

        meter.LMce = meta_loss.detach()

        return weight

    def train_batch(self, eidx, idx, global_step, batch_data, params: MetaSSLParams, device: torch.device):
        meter = Meter()

        sup, unsup = batch_data
        xs, ys = sup
        ids, un_xs, un_axs, un_ys = unsup

        if eidx < 2:
            logits = self.to_logits(xs)
            meter.Lall = meter.Lall + self.loss_ce_(logits, ys, meter=meter, name='Lce')
            self.acc_precise_(logits.argmax(dim=1), ys, meter, name='acc')
        else:
            un_logits = self.to_logits(un_xs)
            weight = self.meta_optimizer(un_xs, xs, ys, meter=meter)

            # meter.Lall = meter.Lall + self.loss_ce_(self.to_logits(xs), ys, meter=meter, name='Lce')
            meter.Lall = meter.Lall + self.loss_ce_with_masked_(un_logits, un_logits.argmax(dim=1), weight,
                                                                meter=meter, name='Ldce')  # Lw*
            self.acc_precise_(un_logits.argmax(dim=1), un_ys, meter, name='un_acc')

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        # self.acc_precise_(dist_targets.argmax(dim=1), un_ys, meter=meter, name='dist_acc')

        return meter

    def create_metanet(self):
        metanet = MetaLeNet(params.n_classes,
                            with_fc=True).to(self.device)
        metanet.load_state_dict(self.model.state_dict())

        meta_sgd = MetaSGD(metanet, **self.params.meta_optim)
        return metanet, meta_sgd

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    params = MetaSSLParams()
    params.device = 'cuda:0'
    params.from_args()
    params.dataset = 'cifar10'
    params.architecture = 'Lenet'
    params.with_fc = True
    params.uratio = 1
    params.n_percls = 4
    params.meta_optim = {
        'lr': 0.1,
        'momentum': 0.9,
    }
    # frame = inspect.currentframe()
    # gpu_tracker = MemTracker(frame)  # define a GPU tracker
    trainer = MetaSSLTrainer(params)

    trainer.train()
