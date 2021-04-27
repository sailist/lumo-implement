"""
code changed from https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py (larger)

Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 23.25M
ResNet32  |    32  | 35.6M
ResNet44  |    44  | ----
ResNet56  |    56  | ----
ResNet110 |   110  |  ----
ResNet1202|  1202  | ----
which this implementation indeed has.

Resnet32 in mentormix do not use bottomneck, and the filter number will be [64, 128, 256, 512], instead of [16, 64, 128, 256],
so the number of parameters will be larger.


Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
"""
from torchvision.models.resnet import resnet50 as _resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    def __init__(self, in_features, out_features, stride=1, option='A', activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_features)
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.bn3 = nn.BatchNorm2d(out_features)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.activate_before_residual = activate_before_residual
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(stride, stride),
                nn.ZeroPad2d([0, 0, 0, 0, (out_features - in_features) // 2, (out_features - in_features) // 2])
            )

    def forward(self, x):
        if self.activate_before_residual:
            out = self.relu(x)
            out = self.bn1(out)
            x = out
        else:
            out = self.bn1(x)
            out = self.relu(out)
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, with_fc=True):
        super(ResNet, self).__init__()
        # self.in_planes = 16
        # filters = [16, 64, 128, 256]
        filters = [64, 128, 256, 512]
        self.conv1 = nn.Conv2d(3, filters[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, filters[0], filters[1], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, filters[1], filters[2], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, filters[2], filters[3], num_blocks[2], stride=2)
        self.fc = nn.Linear(512, num_classes)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.apply(_weights_init)
        self.with_fc = with_fc

    def _make_layer(self, block, in_features, out_features, num_blocks, stride):
        # strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        layers.append(block(in_features, out_features, stride))
        for _ in range(num_blocks):
            layers.append(block(out_features, out_features, 1))
            # self.in_planes = in_features * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        if self.with_fc:
            out = self.fc(out)
        return out


def resnet20(num_classes=10, with_fc=True):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, with_fc=with_fc)


def resnet32(num_classes=10, with_fc=True):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes, with_fc=with_fc)


def resnet44(num_classes=10, with_fc=True):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes, with_fc=with_fc)


# def resnet50(num_classes=101, with_fc=True):
#     model = _resnet50(with_fc = with_fc)
#     model.fc = nn.Linear(model.fc.weight.shape[1], num_classes)
#     return model


def resnet56(num_classes=10, with_fc=True):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes, with_fc=with_fc)


def resnet110(num_classes=10, with_fc=True):
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes, with_fc=with_fc)


def resnet1202(num_classes=10, with_fc=True):
    return ResNet(BasicBlock, [200, 200, 200], num_classes=num_classes, with_fc=with_fc)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


if __name__ == "__main__":
    x = torch.rand(16, 3, 32, 32)
    out = resnet32(10)(x)
    # print(out.shape)
    print(sum([i.numel() for i in resnet20(10).parameters()]) * 1e-6)
