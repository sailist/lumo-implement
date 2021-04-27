"""
reimplement of 'Distilling Effective Supervision from Severe Label Noise'
other name of this paper(submission withdraw in ICLR2020) is 'IEG: Robust neural net training with severe label noises'
    https://arxiv.org/abs/1911.09781

0002.209650e9, 似乎也就那样

"""

if __name__ == '__main__':
    import os
    import sys

    chdir = os.path.dirname(os.path.abspath(__file__))
    chdir = os.path.dirname(chdir)
    chdir = os.path.dirname(chdir)
    sys.path.append(chdir)

from arch.meta import MetaWideResNet, MetaSGD
import torch
from torch import autograd
from typing import List, Tuple
from thexp import Trainer, Meter, Params, AvgMeter
from trainers import NoisyParams, GlobalParams
from torch.nn import functional as F
from trainers.mixin import *
from arch.meta import MetaModule


class IEGParams(NoisyParams):

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
        super(IEGParams, self).initial()
        self.lr_sche = self.SCHE.Cos(start=0.1, end=0.002, left=0, right=params.epoch)

        self.epoch_step = self.SCHE.Linear(end=100, right=self.epoch)
        self.init_eps_val = 1. / self.batch_size
        self.grad_eps_init = 0.9  # eps for meta learning init value


class IEGTrainer(datasets.IEGSyntheticNoisyMixin,
                 callbacks.BaseCBMixin,
                 models.BaseModelMixin,
                 acc.ClassifyAccMixin,
                 losses.CELoss, losses.MixMatchLoss, losses.IEGLoss,
                 Trainer):

    def models(self, params: GlobalParams):
        super().models(params)

    def initial(self):
        super().initial()
        self.target_mem = torch.zeros(50000, params.n_classes, device=self.device, dtype=torch.float)

    def unsupervised_loss(self,
                          xs: torch.Tensor, axs: torch.Tensor,
                          vxs: torch.Tensor, vys: torch.Tensor,
                          logits: torch.Tensor, aug_logits: torch.Tensor,
                          meter: Meter):
        '''create Lub, Lpb, Lkl'''
        re_ids = torch.randperm(vxs.shape[0])
        re_vxs = vxs[re_ids]
        re_vys = vys[re_ids]

        p_target = self.label_guesses_(self.logit_norm_(logits), self.logit_norm_(aug_logits))
        p_target = self.sharpen_(p_target, params.T)

        re_v_targets = tricks.onehot(re_vys, params.n_classes)
        mixed_input, mixed_target = self.mixmatch_up_(re_vxs, [xs, axs], re_v_targets, p_target,
                                                      beta=params.mix_beta)

        mixed_logits = self.to_logits(mixed_input)
        mixed_logits_lis = mixed_logits.split_with_sizes([re_vxs.shape[0], xs.shape[0], axs.shape[0]])
        (mixed_v_logits,
         mixed_n_logits, mixed_an_logits) = [self.logit_norm_(l) for l in mixed_logits_lis]  # type:torch.Tensor

        mixed_nn_logits = torch.cat([mixed_n_logits, mixed_an_logits], dim=0)
        mixed_v_targets, mixed_nn_targets = mixed_target.split_with_sizes(
            [mixed_v_logits.shape[0], mixed_nn_logits.shape[0]])

        # # Lpβ，验证集作为半监督中的有标签数据集
        # meter.Lall = meter.Lall + self.loss_ce_with_targets_(mixed_v_logits, mixed_v_targets,
        #                                                      meter=meter, name='Lpb')
        # # p * Luβ，训练集作为半监督中的无标签数据集
        # meter.Lall = meter.Lall + self.loss_ce_with_targets_(mixed_nn_logits, mixed_nn_targets,
        #                                                      meter=meter, name='Lub')
        #
        # # Lkl，对多次增广的一致性损失
        # meter.Lall = meter.Lall + self.loss_kl_ieg_(logits, aug_logits,
        #                                             n_classes=params.n_classes,
        #                                             consistency_factor=params.consistency_factor,
        #                                             meter=meter)
        return p_target

    def meta_optimizer(self,
                       xs: torch.Tensor, guess_targets: torch.Tensor, n_targets: torch.Tensor,
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
        ## 0. create a metanet which can hold hyperparameter gradients.
        metanet, metasgd = self.create_metanet()

        ## 1. calculate loss of train sample with hyperparameters
        # the hyperparameter used to reweight, requires_grad must be True.
        weight_0 = torch.ones(guess_targets.shape[0],
                              dtype=torch.float,
                              device=self.device) * params.init_eps_val
        weight_0 = autograd.Variable(weight_0, requires_grad=True)

        eps = torch.ones([guess_targets.shape[0]],
                         dtype=torch.float,
                         device=self.device) * params.grad_eps_init
        eps = autograd.Variable(eps, requires_grad=True)
        eps_k = eps.unsqueeze(dim=-1)

        logits = metanet(xs)
        mixed_labels = eps_k * n_targets + (1 - eps_k) * guess_targets
        # ce loss with targets and none reduction
        _net_cost = -torch.mean(mixed_labels * torch.log_softmax(logits, dim=1), dim=1)
        lookahead_loss = torch.mul(weight_0, _net_cost).mean()

        ## 2. update gradient of train samples
        var_grads = autograd.grad(lookahead_loss, metanet.params(), create_graph=True)
        metanet.update_params(0.1, var_grads)
        # or metasgd.meta_step(var_grads)

        ## 3. calculate gradient of meta validate sample
        m_v_logits = metanet(vxs)  # type:torch.Tensor
        v_targets = tricks.onehot(vys, params.n_classes)
        meta_loss = self.loss_ce_with_targets_(m_v_logits, v_targets)

        # method A
        # grad_meta_vars = autograd.grad(meta_loss, metanet.params(), create_graph=True)
        # grad_target, grad_eps = autograd.grad(metanet.params(), [weight_0, eps_k],
        #                                       grad_outputs=grad_meta_vars)

        # equal method B
        grad_target, grad_eps = autograd.grad(
            meta_loss, [weight_0, eps_k])

        ## 4. build weight by meta vlidate gradient
        raw_weight = weight_0 - grad_target
        raw_weight = raw_weight - params.init_eps_val
        unorm_weight = raw_weight.clamp_min(0)
        norm_c = unorm_weight.sum()
        weight = torch.div(unorm_weight, norm_c + 0.00001).detach()
        new_eps = (grad_eps < 0).float().unsqueeze(dim=-1).detach()

        self.acc_precise_(m_v_logits.argmax(dim=-1), vys, meter=meter, name='Macc')

        meter.LMce = meta_loss.detach()
        return weight, new_eps

    def train_batch(self, eidx, idx, global_step, batch_data, params: IEGParams, device: torch.device):
        meter = Meter()
        train_data, (vxs, vys) = batch_data  # type:List[torch.Tensor],(torch.Tensor,torch.Tensor)

        ys, nys = train_data[-2:]
        xs = train_data[1]
        axs = torch.cat(train_data[2:2 + params.K])

        logits = self.to_logits(xs)
        aug_logits = self.to_logits(axs)
        n_targets = tricks.onehot(nys, params.n_classes)

        with torch.no_grad():
            guess_targets = self.unsupervised_loss(xs, axs, vxs, vys, logits, aug_logits, meter=meter)

        weight, eps_k = self.meta_optimizer(xs, guess_targets, n_targets,
                                            vxs, vys, meter=meter)

        meter.tw = eps_k[ys == nys].mean()
        meter.fw = eps_k[ys != nys].mean()

        mixed_targets = eps_k * n_targets + (1 - eps_k) * guess_targets

        # 表面上是 和 targets 做交叉熵，实际上 targets 都是 onehot 形式的

        # loss with initial eps
        init_eps = torch.ones([guess_targets.shape[0]],
                              dtype=torch.float,
                              device=self.device) * params.grad_eps_init
        init_mixed_labels = tricks.elementwise_mul(init_eps,
                                                   n_targets) + tricks.elementwise_mul(1 - init_eps, guess_targets)
        # loss with initial weight

        meter.Lws = self.loss_ce_with_targets_(logits, mixed_targets)  # Lw*
        meter.Llamda = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * init_mixed_labels, dim=1) * weight)  # Lλ*
        meter.Lall = meter.Lall + (meter.Lws + meter.Llamda) / 2

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        self.acc_precise_(logits.argmax(dim=1), ys, meter, name='true_acc')
        self.acc_precise_(logits.argmax(dim=1), nys, meter, name='noisy_acc')

        return meter

    def create_metanet(self):
        metanet = MetaWideResNet(num_classes=params.n_classes,
                                 depth=params.depth,
                                 widen_factor=params.widen_factor).to(self.device)
        metanet.load_state_dict(self.model.state_dict())
        metanet.zero_grad()

        meta_sgd = MetaSGD(metanet, **self.params.meta_optim)
        return metanet, meta_sgd

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    params = IEGParams()
    params.device = 'cuda:0'
    params.from_args()

    trainer = IEGTrainer(params)

    trainer.train()
