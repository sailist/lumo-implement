"""
reimplement of 《Learning to Reweight Examples for Robust Deep Learning》(L2R)，imblance part
    https://arxiv.org/abs/1803.09050

train 25 epoch is enough to see the result(can achieve 93%-95% accuracy), where basic methods can't train anything(about 50%)
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
from trainers import MnistImblanceParams
from trainers.mixin import *
from arch.meta import MetaLeNet
from torch.nn import functional as F


class L2RTrainer(datasets.MnistImblanceDataset,
                 callbacks.BaseCBMixin,
                 acc.ClassifyAccMixin,
                 losses.CELoss,
                 Trainer):

    def models(self, params: MnistImblanceParams):
        from arch.lenet import LeNet
        from thexp.contrib import ParamGrouper

        self.model = LeNet(1)  # type:nn.Module

        grouper = ParamGrouper(self.model)
        noptim = params.optim.args.copy()
        noptim['weight_decay'] = 0
        param_groups = [
            grouper.create_param_group(grouper.kernel_params(with_norm=False), **params.optim.args),
            grouper.create_param_group(grouper.bias_params(with_norm=False), **noptim),
            grouper.create_param_group(grouper.norm_params(), **noptim),
        ]

        self.optim = params.optim.build(param_groups)
        self.to(self.device)

    def meta_optimizer(self,
                       xs: torch.Tensor, ys: torch.Tensor,
                       vxs: torch.Tensor, vys: torch.Tensor,
                       meter: Meter):
        ## 0. create a metanet which can hold hyperparameter gradients.
        metanet = MetaLeNet(1).to(self.device)
        metanet.load_state_dict(self.model.state_dict())
        metanet.zero_grad()

        ## 1. calculate loss of train sample with hyperparameters
        # the hyperparameter used to reweight, requires_grad must be True.
        eps = torch.zeros_like(ys, device=self.device, requires_grad=True)

        logits = metanet(xs).squeeze()
        _net_costs = F.binary_cross_entropy_with_logits(logits, ys, reduction='none')
        pre_loss = torch.sum(_net_costs * eps)

        ## 2. update gradient of train samples
        meta_grads = torch.autograd.grad(pre_loss, (metanet.params()), create_graph=True)
        metanet.update_params(params.optim.args.lr, grads=meta_grads)

        ## 3. calculate gradient of meta validate sample
        m_v_logits = metanet(vxs).squeeze()
        m_v_loss = F.binary_cross_entropy_with_logits(m_v_logits, vys)
        grad_eps = torch.autograd.grad(m_v_loss, eps, only_inputs=True)[0]

        ## 4. build weight by meta vlidate gradient
        w_tilde = torch.clamp_min(-grad_eps, min=0)
        norm_c = torch.sum(w_tilde)

        if norm_c != 0:
            w = w_tilde / norm_c
        else:
            w = w_tilde

        return w

    def train_batch(self, eidx, idx, global_step, batch_data, params: MnistImblanceParams, device: torch.device):
        meter = Meter()

        (xs, ys), (vxs, vys) = batch_data  # type:torch.Tensor

        w = self.meta_optimizer(xs, ys, vxs, vys, meter=meter)

        logits = self.model(xs).squeeze()
        _net_costs = F.binary_cross_entropy_with_logits(logits, ys, reduction='none')
        l_f = torch.sum(_net_costs * w.detach())

        self.optim.zero_grad()
        l_f.backward()
        self.optim.step()

        meter.l_f = l_f
        if (ys == 1).any():
            meter.w1 = w[ys == 1].mean()  # minority will have larger weight
        if (ys == 0).any():
            meter.w0 = w[ys == 0].mean()

        return meter

    def test_eval_logic(self, dataloader, param: MnistImblanceParams):
        meter = AvgMeter()
        for itr, (images, labels) in enumerate(dataloader):
            output = self.model(images).squeeze()
            predicted = (torch.sigmoid(output) > 0.5).int()
            meter.acc = (predicted.int() == labels.int()).float().mean().detach()
            meter.percent(meter.acc_)
        return meter


if __name__ == '__main__':
    params = MnistImblanceParams()
    params.device = 'cpu'
    params.from_args()
    trainer = L2RTrainer(params)

    trainer.train()
