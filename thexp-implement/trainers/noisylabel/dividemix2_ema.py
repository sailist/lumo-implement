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
from data.transforms import BigStrong, BigWeak, BigToTensor
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
        self.epoch = 128
        self.batch_size = 48
        self.K = 1
        self.mix_beta = 0.5
        self.T = 0.5  # sharpening temperature
        self.burn_in_epoch = 0
        self.loss_p_percentile = 0.7
        self.optim = self.create_optim('SGD',
                                       lr=0.002,
                                       momentum=0.9,
                                       weight_decay=5e-4)
        self.lambda_u = 0  # weight for unsupervised loss
        self.noisy_ratio = 0.8
        self.ema = False
        self.p_threshold = 0.5  # clean probability threshold
        self.noisy_type = 'symmetric'
        self.widen_factor = 2  # 10 needs multi-gpu
        self.targets_ema = 0.3
        self.cut_size = 3360 * 2

    def initial(self):
        super(DivideMixParams, self).initial()
        self.lr_sche = self.SCHE.Log(start=self.optim.args.lr, end=0.00001,
                                     left=0, right=self.epoch)
        lr1 = self.optim.args.lr
        lr2 = self.optim.args.lr / 10
        self.lr_sche = self.SCHE.List([
            self.SCHE.Log(lr1, lr2, right=40),
            self.SCHE.Linear(lr2, lr2 / 10, left=40, right=self.epoch),
        ])

        # if self.dataset == 'cifar10':
        self.warm_up = 1
        self.rampup_sche = self.SCHE.Linear(start=0, end=1, left=self.warm_up, right=self.warm_up + 16)


class DivideMixTrainer(datasets.Clothing1mDatasetMixin,
                       callbacks.callbacks.TrainCallback,
                       callbacks.BaseCBMixin,
                       models.BaseModelMixin,
                       acc.ClassifyAccMixin,
                       tricks.Mixture,
                       losses.CELoss, losses.MixMatchLoss, losses.MSELoss, losses.MinENTLoss,
                       Trainer):
    def datasets(self, params: DivideMixParams):

        from data.dataxy_noisylabel import clothing1m_balance
        dataset_fn = clothing1m_balance

        test_x, test_y = dataset_fn(False)
        train_x, train_y = dataset_fn(True, params.cut_size)

        mean, std = norm_val.get('clothing1m', [None, None])
        toTensor = BigToTensor(mean, std)
        weak = BigWeak(mean, std)
        # strong = BigStrong(mean, std)

        self.train_set_pack = [train_x, np.array(train_y)]

        train_set = (
            DatasetBuilder(train_x, train_y)
                .toggle_id()
                .add_x(transform=weak)
                # .add_x(transform=strong)
                .add_y()
        )
        train_dataloader = train_set.DataLoader(batch_size=params.batch_size * 2,
                                                num_workers=params.num_workers,
                                                shuffle=True)
        from thexp import DataBundler
        self.train_set = train_set
        self.train_size = len(train_set)

        self.eval_train_dataloader = (
            DataBundler()
                .add(
                DatasetBuilder(train_x, train_y)
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
            outputs = self.model1(xs) + self.model2(xs)
            return outputs

    def models(self, params: DivideMixParams):
        from trainers.mixin.models import load_backbone
        self.model1 = load_backbone(params)
        self.model2 = load_backbone(params)
        self.optim1 = params.optim.build(self.model1.parameters())
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
        self.false_pred_mem1 = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.target_mem1 = torch.zeros(self.train_size, device=self.device, dtype=torch.float) / params.n_classes
        self.plabel_mem1 = torch.ones(self.train_size, params.n_classes, device=self.device,
                                      dtype=torch.float) / params.n_classes
        self.noisy_cls_mem1 = torch.zeros(self.train_size, dtype=torch.float, device=self.device)

        self.false_pred_mem2 = torch.zeros(self.train_size, params.epoch, dtype=torch.float, device=self.device)
        self.target_mem2 = torch.zeros(self.train_size, device=self.device, dtype=torch.float) / params.n_classes
        self.plabel_mem2 = torch.ones(self.train_size, params.n_classes, device=self.device,
                                      dtype=torch.float) / params.n_classes
        self.noisy_cls_mem2 = torch.zeros(self.train_size, dtype=torch.float, device=self.device)

    def eval_train(self, model,
                   target_mem, plabel_mem, false_pred_mem, noisy_cls_mem):
        eidx = params.eidx
        meter = AvgMeter()

        local_noisy_cls = torch.ones(self.train_size, dtype=torch.float, device=self.device)
        model.eval()

        with torch.no_grad():
            for batch_idx, (ids, xs, nys) in enumerate(self.eval_train_dataloader):
                targets = torch.softmax(model(xs), dim=-1)
                label_pred = targets.gather(1, nys.unsqueeze(dim=1)).squeeze()

                weight = label_pred - target_mem[ids]
                weight = weight + label_pred * 0.5 / params.n_classes - 0.25 / params.n_classes
                weight_mask = weight < 0

                fweight = torch.ones(nys.shape[0], dtype=torch.float, device=self.device)
                # if eidx >= params.burnin:
                fweight[weight_mask] -= params.gmm_w_sche(eidx)
                local_noisy_cls[ids] = fweight

                if eidx > 1:
                    targets = plabel_mem[ids] * params.targets_ema + targets * (1 - params.targets_ema)
                plabel_mem[ids] = targets

                false_pred = targets.gather(1, nys.unsqueeze(dim=1)).squeeze()  # [nys != nys]
                false_pred_mem[ids, eidx - 1] = false_pred
                self.logger.inline(batch_idx)
                self.acc_precise_(targets.argmax(dim=-1), nys, meter=meter)

            f_mean = false_pred_mem[:, :params.eidx].mean(
                dim=1).cpu().numpy()
            f_cur = false_pred_mem[:, params.eidx - 1].cpu().numpy()
            feature = self.create_feature(f_mean, f_cur)
            noisy_cls = self.bmm_predict(feature, mean=False, offset=0)
            noisy_cls_mem = torch.tensor(noisy_cls, device=self.device) * 0.3 + noisy_cls_mem * 0.7

            if params.eidx <= 2:
                noisy_cls = torch.tensor(noisy_cls, device=self.device)
                noisy_cls_mem = torch.tensor(noisy_cls, device=self.device)
            else:
                noisy_cls = noisy_cls_mem.clone()

            meter.mm = noisy_cls.mean()
            self.logger.info(meter)
            self.logger.info('mm raw ratio', noisy_cls.mean(), 'mm ratio', meter.mm)
            filter_mem = local_noisy_cls - noisy_cls
            self.logger.info('max', filter_mem.max(), 'min', filter_mem.min())

            self.logger.info('cls_max', noisy_cls.max(), 'cls_min', noisy_cls.min())
            return torch.relu(filter_mem).cpu().numpy()

    def on_train_epoch_begin(self, trainer: Trainer, func, params: DivideMixParams, *args, **kwargs):
        if params.eidx <= params.warm_up:
            pass
        else:
            self.logger.info('create semi dataset')
            if params.eidx % 2 == 1:
                prob = self.eval_train(self.model1,
                                       target_mem=self.target_mem1,
                                       plabel_mem=self.plabel_mem1,
                                       false_pred_mem=self.false_pred_mem1,
                                       noisy_cls_mem=self.noisy_cls_mem1)  # type: np.ndarray, list
            else:
                prob = self.eval_train(self.model2,
                                       target_mem=self.target_mem2,
                                       plabel_mem=self.plabel_mem2,
                                       false_pred_mem=self.false_pred_mem2,
                                       noisy_cls_mem=self.noisy_cls_mem2)  # type: np.ndarray, list
            pred = (prob > params.p_threshold)

            pred_idx = pred.nonzero()[0]
            unpred_idx = (1 - pred).nonzero()[0]

            train_x, train_y = self.train_set_pack

            mean, std = norm_val.get(params.dataset, [None, None])
            weak = BigWeak(mean, std)

            self.labeled_dataloader = (
                DatasetBuilder(train_x, train_y)
                    .add_labels(prob, source_name='nprob')
                    .add_x(transform=weak)
                    .add_x(transform=weak)
                    .add_y()
                    .add_y(source='nprob')
                    .subset(pred_idx)
                    .DataLoader(params.batch_size, shuffle=True, drop_last=True, num_workers=params.num_workers)
            )

            self.unlabeled_dataloader = (
                DatasetBuilder(train_x, train_y)
                    .add_x(transform=weak)
                    .add_x(transform=weak)
                    .add_y()
                    .subset(unpred_idx)
                    .DataLoader(params.batch_size, shuffle=True, drop_last=True, num_workers=params.num_workers)
            )
            self.unlabeled_dataloader_iter = None
            bundler = DataBundler()
            bundler.add(self.labeled_dataloader)  # .cycle(self.unlabeled_dataloader).zip_mode()
            self.logger.info('new training dataset', bundler, len(self.unlabeled_dataloader))
            self.regist_databundler(train=bundler.to(self.device))

    def warmup_model(self, batch_data, model, optim, meter: Meter):
        (ids, xs, nys) = batch_data  # type:torch.Tensor
        optim.zero_grad()

        logits = model(xs)  # type:torch.Tensor
        meter.Lall = meter.Lall + self.loss_ce_(logits, nys, meter=meter, name='Lce')

        self.acc_precise_(logits.argmax(dim=-1), nys, meter=meter, name='acc')
        meter.Lall.backward()
        optim.step()

    def train_model(self, model1, model2, optim, batch_data, params: DivideMixParams, meter: Meter):
        model1.train()
        model2.eval()

        # sup, unsup = batch_data
        sup = batch_data
        (xs, xs2, nys, prob) = sup

        try:
            if self.unlabeled_dataloader_iter is None:
                self.unlabeled_dataloader_iter = iter(self.unlabeled_dataloader)
            unsup = next(self.unlabeled_dataloader_iter)
        except:
            self.unlabeled_dataloader_iter = iter(self.unlabeled_dataloader)
            unsup = next(self.unlabeled_dataloader_iter)

        (uxs, uxs2, unys) = unsup
        (uxs, uxs2, unys) = (uxs.to(self.device), uxs2.to(self.device), unys.to(self.device))

        n_targets = tricks.onehot(nys, params.n_classes)
        # nys = torch.zeros(params.batch_size, params.n_classes, device=self.device).scatter_(1, nys.view(-1, 1), 1)
        prob = prob.view(-1, 1).float()
        batch_size = xs.shape[0]
        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = model1(uxs)
            outputs_u12 = model1(uxs2)
            outputs_u21 = model2(uxs)
            outputs_u22 = model2(uxs2)

            pu = self.label_guesses_(outputs_u11, outputs_u12, outputs_u21, outputs_u22)
            targets_u = self.sharpen_(pu, params.T)  # temparature sharpening

            # label refinement of labeled samples
            outputs_x = model1(xs)
            outputs_x2 = model1(xs2)

            px = self.label_guesses_(outputs_x, outputs_x2)
            px = prob * n_targets + (1 - prob) * px
            targets_x = self.sharpen_(px, params.T)  # temparature sharpening

            self.acc_precise_(outputs_x.argmax(dim=-1), nys, meter=meter)
            self.acc_precise_(outputs_u11.argmax(dim=-1), unys, meter=meter, name='uacc')

            # mixmatch
        l = np.random.beta(params.mix_beta, params.mix_beta)
        l = max(l, 1 - l)

        all_inputs = torch.cat([xs, xs2, uxs, uxs2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.shape[0])

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a[:batch_size * 2] + (1 - l) * input_b[:batch_size * 2]
        mixed_target = l * target_a[:batch_size * 2] + (1 - l) * target_b[:batch_size * 2]

        logits = model1(mixed_input)
        # logits_x = model(mixed_input[:batch_size * 2])
        # logits_u = model(mixed_input[batch_size * 2:])
        logits_x = logits

        meter.Lall = meter.Lall + self.loss_ce_with_targets_(logits_x, mixed_target[:batch_size * 2],
                                                             meter=meter, name='Lx')

        # regularization
        prior = torch.ones(params.n_classes, device=self.device) / params.n_classes
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        meter.Lpen = torch.sum(prior * torch.log(prior / pred_mean))  # penalty

        meter.Lall = meter.Lall + meter.Lpen

        # compute gradient and do SGD step
        optim.zero_grad()
        meter.Lall.backward()
        optim.step()

        return meter

    def train_batch(self, eidx, idx, global_step, batch_data, params: DivideMixParams, device: torch.device):
        meter = Meter()
        if eidx <= params.warm_up:
            if eidx % 2 == 1:
                self.warmup_model(batch_data, self.model1, self.optim1, meter)
            else:
                self.warmup_model(batch_data, self.model2, self.optim2, meter)
        else:
            if eidx % 2 == 1:
                self.train_model(self.model1, self.model2, self.optim1, batch_data, params, meter)
            else:
                self.train_model(self.model2, self.model1, self.optim2, batch_data, params, meter)
        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    params = DivideMixParams()
    params.eval_test_per_epoch = (0, 1)
    params.dataset = 'clothing1m'
    params.pretrain = True
    params.num_workers = 4
    params.device = 'cuda:0'
    params.resnet50()
    params.from_args()
    trainer = DivideMixTrainer(params)
    #
    trainer.train()

    # params.initial()
    # params.lr_sche.plot()
