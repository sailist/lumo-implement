"""
reimplement of 'DivideMix: Learning with Noisy Labels as Semi-supervised Learning'
    https://arxiv.org/abs/2002.07394
    original repository : https://github.com/LiJunnan1992/DivideMix

0026.55bde0f2，0.8, 90.54
"""

if __name__ == '__main__':
    import os
    import sys

    chdir = os.path.dirname(os.path.abspath(__file__))
    chdir = os.path.dirname(chdir)
    chdir = os.path.dirname(chdir)
    sys.path.append(chdir)

import gc
import torch
from data.constant import norm_val
from typing import List, Tuple
from thexp import Trainer, Meter, Params, AvgMeter, DataBundler
from trainers import NoisyParams, GlobalParams
from torch.nn import functional as F
from trainers.mixin import *
from thexp import DatasetBuilder
from data.transforms import ToNormTensor
from data.transforms import Weak
from data.transforms import Strong
import numpy as np

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


class DivideMixParams(NoisyParams):

    def __init__(self):
        super().__init__()
        self.epoch = 300 * 2
        self.batch_size = 64
        self.K = 1
        self.mix_beta = 0.5
        self.T = 0.5  # sharpening temperature
        self.burn_in_epoch = 0
        self.loss_p_percentile = 0.7
        self.optim = self.create_optim('SGD',
                                       lr=0.02,
                                       momentum=0.9,
                                       weight_decay=5e-4)
        self.lambda_u = 0  # weight for unsupervised loss
        self.noisy_ratio = 0.8
        self.ema = False
        self.p_threshold = 0.5  # clean probability threshold
        self.noisy_type = 'symmetric'
        self.widen_factor = 2  # 10 needs multi-gpu

    def initial(self):
        super(DivideMixParams, self).initial()
        # self.lr_sche = self.SCHE.Cos(start=self.optim.args.lr, end=0.002,
        #                              left=0, right=self.epoch)
        lr1 = self.optim.args.lr
        lr2 = self.optim.args.lr / 10
        self.lr_sche = self.SCHE.List([
            self.SCHE.Linear(lr1, lr1, right=300),
            self.SCHE.Linear(lr2, lr2, left=300, right=self.epoch),
        ])

        if self.dataset == 'cifar10':
            self.warm_up = 10 * 2
        elif self.dataset == 'cifar100':
            self.warm_up = 30 * 2

        self.rampup_sche = self.SCHE.Linear(start=0, end=1, left=self.warm_up, right=self.warm_up + 16)


class DivideMixTrainer(datasets.SyntheticNoisyMixin,
                       callbacks.callbacks.TrainCallback,
                       callbacks.BaseCBMixin,
                       models.BaseModelMixin,
                       acc.ClassifyAccMixin,
                       losses.CELoss, losses.MixMatchLoss, losses.MSELoss, losses.MinENTLoss,
                       Trainer):
    def datasets(self, params: DivideMixParams):
        from data.dataxy import datasets
        dataset_fn = datasets[params.dataset]

        test_x, test_y = dataset_fn(False)
        train_x, train_y = dataset_fn(True)

        mean, std = norm_val.get(params.dataset, [None, None])
        toTensor = ToNormTensor(mean, std)
        weak = Weak(mean, std)
        strong = Strong(mean, std)

        if params.noisy_type == 'asymmetric':
            from data.noisy import asymmetric_noisy
            noisy_y = asymmetric_noisy(train_y, params.noisy_ratio, n_classes=params.n_classes)

        elif params.noisy_type == 'symmetric':
            from data.noisy import symmetric_noisy
            noisy_y = symmetric_noisy(train_y, params.noisy_ratio, n_classes=params.n_classes)

        else:
            assert False, params.noisy_type
        self.train_set_pack = [train_x, np.array(train_y), noisy_y]

        self.logger.info('noisy acc = {}'.format((train_y == noisy_y).mean()))

        train_set = (
            DatasetBuilder(train_x, train_y)
                .add_labels(noisy_y, 'noisy_y')
                .toggle_id()
                .add_x(transform=weak)
                .add_x(transform=strong)
                .add_y()
                .add_y(source='noisy_y')
        )
        train_dataloader = train_set.DataLoader(batch_size=params.batch_size * 2,
                                                num_workers=params.num_workers,
                                                shuffle=True)
        from thexp import DataBundler

        self.eval_train_dataloader = (
            DataBundler()
                .add(
                DatasetBuilder(train_x, noisy_y)
                    .toggle_id()
                    .add_x(transform=toTensor)
                    .add_y()
                    .DataLoader(batch_size=params.batch_size,
                                num_workers=params.num_workers // 2,
                                shuffle=False)
            ).to(self.device)
        )

        test_dataloader = (
            DatasetBuilder(test_x, test_y)
                .add_x(transform=toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers // 2, shuffle=False)
        )

        self.regist_databundler(train=train_dataloader,
                                test=test_dataloader)
        self.to(self.device)

    def predict(self, xs) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(xs) + self.model2(xs)
            return outputs

    def models(self, params: DivideMixParams):
        from trainers.mixin.models import load_backbone
        self.model = load_backbone(params)
        from copy import deepcopy
        self.model2 = deepcopy(self.model)

        from thexp.contrib import ParamGrouper

        self.optim = params.optim.build(self.model.parameters())
        self.optim2 = params.optim.build(self.model2.parameters())
        self.to(self.device)

    def callbacks(self, params: DivideMixParams):
        super(DivideMixTrainer, self).callbacks(params)
        self.hook(self)

    def _regist_databundler(self, key, val):
        from torch.utils.data import DataLoader
        assert isinstance(val, (DataBundler, DataLoader))
        if isinstance(val, DataLoader):
            val = DataBundler().add(val)
        if key in self._databundler_dict:
            del self._databundler_dict[key]
        self._databundler_dict[key] = val

    def initial(self):
        super().initial()
        self.moving_loss_dic = torch.zeros(50000, device=self.device, dtype=torch.float)
        self.all_loss = [[], []]

    def eval_train(self, model, all_loss):
        from sklearn.mixture import GaussianMixture
        model.eval()
        losses = torch.zeros(50000, device=self.device)
        with torch.no_grad():
            for batch_idx, (ids, xs, ys) in enumerate(self.eval_train_dataloader):
                outputs = model(xs)
                loss = F.cross_entropy(outputs, ys, reduction='none')
                losses[ids] = loss
        losses = (losses - losses.min()) / (losses.max() - losses.min())
        all_loss.append(losses)

        if params.noisy_ratio >= 0.9:  # average loss over last 5 epochs to improve convergence stability
            history = torch.stack(all_loss)
            input_loss = history[-5:].mean(0)
            input_loss = input_loss.reshape(-1, 1)
        else:
            input_loss = losses.reshape(-1, 1)

        input_loss = input_loss.cpu().numpy()
        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]
        return prob, all_loss

    def on_train_epoch_begin(self, trainer: Trainer, func, params: DivideMixParams, *args, **kwargs):
        if params.eidx < params.warm_up:
            pass
        else:
            if params.eidx % 2 == 0:
                prob, self.all_loss[1] = self.eval_train(self.model2, self.all_loss[1])  # type: np.ndarray, list
                pred = (prob > params.p_threshold)
            else:
                prob, self.all_loss[0] = self.eval_train(self.model, self.all_loss[0])  # type: np.ndarray, list
                pred = (prob > params.p_threshold)

            pred_idx = pred.nonzero()[0]
            unpred_idx = (1 - pred).nonzero()[0]

            train_x, train_y, noisy_y = self.train_set_pack
            clean = (noisy_y == train_y)
            acc = (pred[clean]).mean()
            self.logger.info('Numer of labeled samples', pred.sum(), 'clean ratio = {}'.format(acc))

            mean, std = norm_val.get(params.dataset, [None, None])
            weak = Weak(mean, std)

            labeled_dataloader = (
                DatasetBuilder(train_x, train_y)
                    .add_labels(noisy_y, source_name='nys')
                    .add_labels(prob, source_name='nprob')
                    .add_x(transform=weak)
                    .add_x(transform=weak)
                    .add_y()
                    .add_y(source='nys')
                    .add_y(source='nprob')
                    .subset(pred_idx)
                    .DataLoader(params.batch_size, shuffle=True, drop_last=True, num_workers=params.num_workers)
            )

            unlabeled_dataloader = (
                DatasetBuilder(train_x, train_y)
                    .add_labels(noisy_y, source_name='nys')
                    .add_x(transform=weak)
                    .add_x(transform=weak)
                    .add_y()
                    .add_y(source='nys')
                    .subset(unpred_idx)
                    .DataLoader(params.batch_size, shuffle=True, drop_last=True, num_workers=params.num_workers)
            )
            bundler = DataBundler()
            bundler.add(labeled_dataloader).cycle(unlabeled_dataloader).zip_mode()
            self.logger.info('new training dataset', bundler)
            self.regist_databundler(train=bundler.to(self.device))

    def warmup_model(self, batch_data, model, optim, meter: Meter):
        (ids, xs, axs, ys, nys) = batch_data  # type:torch.Tensor
        optim.zero_grad()

        logits = model(xs)  # type:torch.Tensor
        meter.Lall = meter.Lall + self.loss_ce_(logits, nys, meter=meter, name='Lce')
        if params.noisy_type == 'asymmetric':  # penalize confident prediction for asymmetric noise
            meter.Lall = meter.Lall + self.loss_minent_(logits, meter=meter, name='Lpen')

        self.acc_precise_(logits.argmax(dim=-1), ys, meter=meter, name='acc')
        meter.Lall.backward()
        optim.step()

    def train_model(self, model, model2, optim, batch_data, params: DivideMixParams, meter: Meter):
        model.train()
        model2.eval()

        sup, unsup = batch_data
        (xs, xs2, ys, nys, prob) = sup
        (uxs, uxs2, uys, unys) = unsup

        n_targets = tricks.onehot(nys, params.n_classes)
        # nys = torch.zeros(params.batch_size, params.n_classes, device=self.device).scatter_(1, nys.view(-1, 1), 1)
        prob = prob.view(-1, 1).float()
        batch_size = xs.shape[0]
        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = model(uxs)
            outputs_u12 = model(uxs2)
            outputs_u21 = model2(uxs)
            outputs_u22 = model2(uxs2)

            pu = self.label_guesses_(outputs_u11, outputs_u12, outputs_u21, outputs_u22)
            targets_u = self.sharpen_(pu, params.T)  # temparature sharpening

            # label refinement of labeled samples
            outputs_x = model(xs)
            outputs_x2 = model(xs2)

            px = self.label_guesses_(outputs_x, outputs_x2)
            px = prob * n_targets + (1 - prob) * px
            targets_x = self.sharpen_(px, params.T)  # temparature sharpening

        # mixmatch
        l = np.random.beta(params.mix_beta, params.mix_beta)
        l = max(l, 1 - l)

        all_inputs = torch.cat([xs, xs2, uxs, uxs2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.shape[0], device=self.device)

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        logits = model(mixed_input)
        logits_x = logits[:batch_size * 2]
        logits_u = logits[batch_size * 2:]

        meter.Lall = meter.Lall + self.loss_ce_with_targets_(logits_x, mixed_target[:batch_size * 2],
                                                             meter=meter, name='Lx')
        meter.Lall = meter.Lall + self.loss_mse_(logits_u, mixed_target[batch_size * 2:],
                                                 w_mse=params.rampup_sche(params.eidx) * 25,
                                                 meter=meter, name='Lu')

        # regularization
        prior = torch.ones(params.n_classes, device=self.device) / params.n_classes

        pred_mean = torch.softmax(logits, dim=1).mean(0)
        meter.Lpen = torch.sum(prior * torch.log(prior / pred_mean))  # penalty

        meter.Lall = meter.Lall + meter.Lpen

        # compute gradient and do SGD step
        optim.zero_grad()
        meter.Lall.backward()
        optim.step()

    def train_batch(self, eidx, idx, global_step, batch_data, params: DivideMixParams, device: torch.device):
        meter = Meter()
        if eidx < params.warm_up:
            if eidx % 2 == 0:
                self.warmup_model(batch_data, self.model, self.optim, meter)
            else:
                self.warmup_model(batch_data, self.model2, self.optim2, meter)
        else:
            if eidx % 2 == 0:
                self.train_model(self.model, self.model2, self.optim, batch_data, params, meter)
            else:
                self.train_model(self.model2, self.model, self.optim2, batch_data, params, meter)
        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    params = DivideMixParams()
    params.num_workers = 4
    params.device = 'cuda:2'
    params.preresnet18()
    params.from_args()
    trainer = DivideMixTrainer(params)

    trainer.train()
