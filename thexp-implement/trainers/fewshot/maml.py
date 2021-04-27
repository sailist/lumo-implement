"""
"""
from arch.meta import MetaWideResNet

if __name__ == '__main__':
    import os
    import sys

    chdir = os.path.dirname(os.path.abspath(__file__))
    chdir = os.path.dirname(chdir)
    chdir = os.path.dirname(chdir)
    sys.path.append(chdir)

import torch
from thexp import Trainer, Meter
from trainers import FewshotParams
from trainers.mixin import *


class SupervisedTrainer(callbacks.BaseCBMixin,
                        datasets.Base32Mixin,
                        models.BaseModelMixin,
                        acc.ClassifyAccMixin,
                        losses.CELoss, losses.MixMatchLoss,
                        callbacks.callbacks.TrainCallback,
                        Trainer):

    def train_batch(self, eidx, idx, global_step, batch_data, params: FewshotParams, device: torch.device):
        super().train_batch(eidx, idx, global_step, batch_data, params, device)
        meter = Meter()
        losses_q = [0 for _ in range(params.update_step + 1)]
        corrects = [0 for _ in range(params.update_step + 1)]
        x_spt, y_spt, x_qry, y_qry = batch_data
        for i in range(params.task_num):
            metanet = self.create_metanet()
            logits = metanet(x_spt[i])
            loss = self.loss_ce_(logits, y_spt[i])

            grad = torch.autograd.grad(loss, metanet.params())
            metanet.update_params(params.meta_lr, grad)

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.model(x_qry[i])
                loss_q = self.loss_ce_(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = logits_q.argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = metanet(x_qry[i])
                loss_q = self.loss_ce_(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = logits_q.argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, params.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = metanet(x_spt[i])
                loss = self.loss_ce_(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, metanet.params())
                # 3. theta_pi = theta_pi - train_lr * grad
                metanet.update_params(params.meta_lr, grad)

                logits_q = metanet(x_qry[i])
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = self.loss_ce_(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = logits_q.argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        return meter

    def create_metanet(self):
        metanet = MetaWideResNet(num_classes=params.n_classes,
                                 depth=params.depth,
                                 widen_factor=params.widen_factor).to(self.device)
        metanet.load_state_dict(self.model.state_dict())

        return metanet

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    params = FewshotParams()
    params.from_args()
    for _p in params.iter_baseline():
        for pp in _p.grid_range(1):  # try n times
            trainer = SupervisedTrainer(params)
            trainer.train()
