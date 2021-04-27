"""
reimplement of 《Learning to Reweight Examples for Robust Deep Learning》(L2R)，noisy label part
    https://arxiv.org/abs/1803.09050

Can't reach the paper results.
"""

if __name__ == '__main__':
    import os
    import sys

    chdir = os.path.dirname(os.path.abspath(__file__))
    chdir = os.path.dirname(chdir)
    chdir = os.path.dirname(chdir)
    sys.path.append(chdir)

import torch
from thexp import Trainer, Meter, AvgMeter
from torch.nn import functional as F

from arch.meta import MetaSGD, MetaWideResNet
from trainers import NoisyParams
from trainers.mixin import *


class L2RNoisyLoss(losses.Loss):
    """
    先noisy 计算一遍loss，计算梯度，然后再clean计算一遍loss，计算梯度，然后梯度对梯度计算权重

    noisy的时候，为交叉熵求和reduction+zero weight
    clean的时候，为交叉熵求和reduction+1/batch weight ，等价于mean CE
        所有的损失同时再加上 L2 损失

    """

    def regularization_loss(self, model, l2_decay, meter: Meter = None, name: str = 'L2'):
        from thexp.contrib import ParamGrouper

        params = ParamGrouper(model).kernel_params(with_norm=False)
        cost = 0
        for p in params:
            cost = cost + (p ** 2).sum()

        loss = cost * l2_decay
        if meter is not None:
            meter[name] = loss
        return loss

    def weighted_ce_loss(self, logits, labels, weights, meter: Meter = None, name: str = 'Lwce'):
        loss_ = F.cross_entropy(logits, labels, reduction='none')
        loss = torch.mean(loss_ * weights)

        if meter is not None:
            meter[name] = loss
        return loss


class L2RTrainer(callbacks.BaseCBMixin,
                 datasets.IEGSyntheticNoisyMixin,
                 models.BaseModelMixin,
                 acc.ClassifyAccMixin,
                 losses.CELoss, L2RNoisyLoss, losses.MixMatchLoss,
                 Trainer):

    def create_metanet(self):
        metanet = MetaWideResNet(num_classes=params.n_classes,
                                 depth=params.depth,
                                 widen_factor=params.widen_factor).to(self.device)
        metanet.load_state_dict(self.model.state_dict())

        meta_sgd = MetaSGD(metanet, **self.params.meta_optim)
        return metanet, meta_sgd

    def meta_optimizer(self,
                       xs: torch.Tensor, nys: torch.Tensor,
                       vxs: torch.Tensor, vys: torch.Tensor,
                       meter: Meter):
        ## 0. create a metanet which can hold hyperparameter gradients.
        metanet, meta_sgd = self.create_metanet()

        ## 1. calculate loss of train sample with hyperparameters
        # the hyperparameter used to reweight, requires_grad must be True.
        weight_0 = torch.ones_like(nys, device=self.device, dtype=torch.float32) * params.init_eps_val
        weight_0 = torch.autograd.Variable(weight_0, requires_grad=True)

        logits = metanet(xs)

        guess_targets = self.sharpen_(torch.softmax(logits, dim=1))
        n_targets = tricks.onehot(nys, params.n_classes)
        mixed_labels = n_targets * 0.1 + 0.9 * guess_targets

        # lookahead_loss = self.weighted_ce_loss(logits, nys, weight_0)  # meta noisy ce
        lookahead_loss = self.loss_ce_with_targets_masked_(logits, mixed_labels, weight_0)

        ## 2. update gradient of train samples
        metanet.zero_grad()
        grads = torch.autograd.grad(lookahead_loss, (metanet.params()), create_graph=True)
        # meta_sgd.meta_step(grads)
        metanet.update_params(0.1, grads=grads)

        ## 3. calculate gradient of meta validate sample
        m_v_logits = metanet(vxs)
        # eps_b = torch.ones_like(nys,
        #                         device=self.device,
        #                         requires_grad=True,
        #                         dtype=torch.float32) * (1 / params.batch_size)
        v_targets = tricks.onehot(vys, params.n_classes)
        v_meta_loss = (self.loss_ce_with_targets_(m_v_logits, v_targets))  # meta clean ce + meta clean l2

        ## 4. build weight by meta vlidate gradient

        # grads_b = torch.autograd.grad(v_meta_loss, metanet.params(), create_graph=True, retain_graph=True)
        # grad_weight = torch.autograd.grad(grads, weight_0, grad_outputs=grads_b)[0]
        grad_weight = torch.autograd.grad(v_meta_loss, [weight_0], only_inputs=True)[0]

        # - original weight generate method
        # w_tilde = torch.clamp_min(-grad_weight, min=0)
        # norm_c = torch.sum(w_tilde)
        #
        # if norm_c != 0:
        #     weight = w_tilde / norm_c
        # else:
        #     weight = w_tilde

        # - IEG weight generate method
        raw_weight = weight_0 - grad_weight
        raw_weight = raw_weight - params.init_eps_val
        weight = raw_weight.clamp_min(0)
        weight[weight > 0] = 1
        # norm_c = unorm_weight.mean()
        # weight = torch.div(unorm_weight, norm_c + 0.00001).detach()
        with torch.no_grad():
            meter.val_acc = (m_v_logits.argmax(dim=-1) == vys).float().mean()
        return weight.detach()

    def train_batch(self, eidx, idx, global_step, batch_data, params: NoisyParams, device: torch.device):
        meter = Meter()
        # 注意，ys 为训练集的真实label，只用于计算准确率，不用于训练过程
        (ids, xs, axs, ys, nys), (vxs, vys) = batch_data

        w_logits = self.model(xs)
        logits = self.model(axs)  # type:torch.Tensor

        p_targets = self.sharpen_(torch.softmax(w_logits, dim=1))

        weight = self.meta_optimizer(xs, nys, vxs, vys, meter=meter)

        meter.Lall = meter.Lall + self.loss_ce_with_masked_(logits, nys, weight, meter=meter, name='Lwce')
        meter.Lall = meter.Lall + self.loss_ce_with_targets_masked_(logits, p_targets, (1 - weight),
                                                                    meter=meter, name='Lwce')

        meter.tw = weight[ys == nys].mean()
        meter.fw = weight[ys != nys].mean()

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        self.acc_precise_(logits.argmax(dim=1), ys, meter=meter, name='true_acc')
        self.acc_precise_(logits.argmax(dim=1), nys, meter=meter, name='noisy_acc')

        return meter


if __name__ == '__main__':
    params = NoisyParams()
    params.ema = True  # l2r have no ema for model
    params.epoch = 120
    params.batch_size = 100
    params.device = 'cuda:3'
    params.optim.args.lr = 0.1
    params.meta_optim = {
        'lr': 0.1,
        'momentum': 0.9,
    }
    params.from_args()
    trainer = L2RTrainer(params)
    trainer.train()
