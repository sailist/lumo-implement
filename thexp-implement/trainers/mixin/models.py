from torch import nn

import torch

from thexp import Trainer
from thexp.contrib import EMA, ParamGrouper

import arch
from .. import *


class ModelMixin(Trainer):

    def models(self, params: GlobalParams):
        raise NotImplementedError()

    def predict(self, xs):
        raise NotImplementedError()


def load_backbone(params: GlobalParams):
    if params.architecture == 'WRN':
        params.with_fc = params.default(True)
        from arch.wideresnet import WideResNet
        model = WideResNet(depth=params.depth,
                           widen_factor=params.widen_factor,
                           with_fc=params.with_fc,
                           dropout_rate=params.drop_rate,
                           num_classes=params.n_classes)
    elif params.architecture == 'Resnet':
        from arch import resnet
        model_name = 'resnet{}'.format(params.depth)
        if params.depth != 50:
            assert model_name in resnet.__dict__
            model = resnet.__dict__[model_name](num_classes=params.n_classes)
        else:
            model = resnet._resnet50(pretrained=params.pretrain)
            model.fc = nn.Linear(2048, params.n_classes)

    elif params.architecture == 'PreResnet':
        from arch import pre_resnet
        model_name = 'ResNet{}'.format(params.depth)
        assert model_name in pre_resnet.__dict__
        model = pre_resnet.__dict__[model_name](num_classes=params.n_classes,
                                                with_fc=params.with_fc)
    elif params.architecture == 'Lenet':
        from arch import lenet
        model = lenet.LeNet(params.n_classes, with_fc=params.with_fc)
    elif params.architecture == 'WN':
        from arch.widenet import WideResNet
        model = WideResNet(depth=params.depth,
                           widen_factor=params.widen_factor,
                           with_fc=params.with_fc,
                           dropout_rate=params.drop_rate,
                           num_classes=params.n_classes)
    elif params.architecture == "inception":
        from arch.InceptionResNetV2 import InceptionResNetV2
        model = InceptionResNetV2(num_classes=params.n_classes)
    else:
        assert False

    return model


class BaseModelMixin(ModelMixin):
    """base end-to-end model"""

    def predict(self, xs) -> torch.Tensor:
        with torch.no_grad():
            if self.params.ema:
                model = self.ema_model
            else:
                model = self.model
            with torch.no_grad():
                if not self.params.with_fc:
                    return model.fc(model(xs))
                return model(xs)

    def models(self, params: GlobalParams):
        model = load_backbone(params)

        if params.distributed:
            from torch.nn.modules import SyncBatchNorm
            model = SyncBatchNorm.convert_sync_batchnorm(model.cuda())
            self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[params.local_rank])
        else:
            self.model = model

        if params.ema:
            self.ema_model = EMA(self.model)

        grouper = ParamGrouper(self.model)
        noptim = params.optim.args.copy()
        noptim['weight_decay'] = 0
        param_groups = [
            grouper.create_param_group(grouper.kernel_params(with_norm=False), **params.optim.args),
            grouper.create_param_group(grouper.bias_params(with_norm=False), **noptim),
            grouper.create_param_group(grouper.norm_params(), **noptim),
        ]

        self.optim = params.optim.build(param_groups)
        if not params.distributed:
            self.to(self.device)


class SimCLRMixin(ModelMixin):
    """base end-to-end model"""

    def predict(self, xs) -> torch.Tensor:
        with torch.no_grad():
            if self.params.ema:
                model = self.ema_model
            else:
                model = self.model
            with torch.no_grad():
                return model.fc(model(xs))

    def models(self, params: SimCLRParams):
        params.with_fc = False
        backbone = load_backbone(params)

        from arch.ssmodel import ContrastiveModel
        model = ContrastiveModel(backbone, params.mid_dim,
                                 feature_dim=params.feature_dim,
                                 n_classes=params.n_classes)

        if params.distributed:
            from torch.nn.modules import SyncBatchNorm
            model = SyncBatchNorm.convert_sync_batchnorm(model.cuda())
            self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[params.local_rank])
        else:
            self.model = model

        if params.ema:
            self.ema_model = EMA(self.model)

        grouper = ParamGrouper(self.model)
        noptim = params.optim.args.copy()
        noptim['weight_decay'] = 0
        param_groups = [
            grouper.create_param_group(grouper.kernel_params(with_norm=False), **params.optim.args),
            grouper.create_param_group(grouper.bias_params(with_norm=False), **noptim),
            grouper.create_param_group(grouper.norm_params(), **noptim),
        ]

        self.optim = params.optim.build(param_groups)
        if not params.distributed:
            self.to(self.device)
