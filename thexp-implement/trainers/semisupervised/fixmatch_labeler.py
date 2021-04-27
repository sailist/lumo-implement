"""
reimplement of 《FixMatch: Simplifying Semi-Supervised Learning with Consistency and Conﬁdence》
    https://arxiv.org/abs/2001.07685
"""
if __name__ == '__main__':
    import os
    import sys

    chdir = os.path.dirname(os.path.abspath(__file__))
    chdir = os.path.dirname(chdir)
    chdir = os.path.dirname(chdir)
    sys.path.append(chdir)

import torch

from thexp import Trainer, Meter
from trainers import SemiSupervisedParams

from trainers.mixin import *


class FixMatchParams(SemiSupervisedParams):

    def __init__(self):
        super().__init__()
        self.epoch = 16912  # 62 iter / epoch,
        self.batch_size = 64
        self.optim = self.create_optim('SGD',
                                       lr=0.03,
                                       weight_decay=0.0005,
                                       momentum=0.9,
                                       nesterov=True)
        self.pred_thresh = 0.95
        self.lambda_u = 1
        self.uratio = 7

    def initial(self):
        super().initial()
        if self.dataset == 'cifar100':
            self.optim.args.weight_decay = 0.001


class Labeler32Mixin(Trainer):
    """
    all 32*32 dataset, including cifar10, cifar100, svhn

    use base train data shape: ids, xs, aug_xs, ys

    test data shape: xs, ys
    """

    def datasets(self, params: FixMatchParams):
        from thexp import DatasetBuilder
        from torchvision.transforms import ToTensor
        dataset_fn = datasets.datasets[params.dataset]

        train_x, train_y = dataset_fn(True)

        toTensor = ToTensor()

        train_dataloader = (
            DatasetBuilder(train_x, train_y)
                .toggle_id()
                .add_x(transform=toTensor)
                .add_y()
                .DataLoader(batch_size=params.batch_size,
                            num_workers=params.num_workers,
                            shuffle=False)
        )
        self.regist_databundler(test=train_dataloader)
        self.to(self.device)


class FixMatchTrainer(callbacks.BaseCBMixin,
                      callbacks.callbacks.TrainCallback,
                      Labeler32Mixin,
                      models.BaseModelMixin,
                      acc.ClassifyAccMixin,
                      losses.CELoss, losses.FixMatchLoss,
                      Trainer):

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)

    def test_eval_logic(self, dataloader, param: FixMatchParams):
        from thexp.calculate import accuracy as acc

        param.topk = param.default([1, 5])

        with torch.no_grad():
            noisy_mem = torch.zeros(50000, device=self.device, dtype=torch.long)
            count_dict = Meter()
            for batch_data in dataloader:
                ids, xs, labels = batch_data
                preds = self.predict(xs)
                noisy_ys = preds.argmax(dim=1)
                noisy_mem[ids] = noisy_ys
                total, topk_res = acc.classify(preds, labels, topk=param.topk)
                count_dict["total"] += total
                for i, topi_res in zip(param.topk, topk_res):
                    count_dict["top{}".format(i)] += topi_res

        import numpy as np
        noisy_mem = noisy_mem.detach().cpu().numpy()
        np.save('noisy_{}.npy'.format(count_dict['top1']), noisy_mem)
        self.logger.info()
        return count_dict


if __name__ == '__main__':
    params = FixMatchParams()
    params.n_percls = 25
    params.device = 'cuda:2'
    params.from_args()
    trainer = FixMatchTrainer(params)
    model_names = [
        r'/home/yanghaozhe/jupyter_data/thexp-implement2/.thexp/experiments/thexp-implement2.19/fixmatch.semisupervised/0008.bf7da99b/modules/model.0000021.pth',
        r'/home/yanghaozhe/jupyter_data/thexp-implement2/.thexp/experiments/thexp-implement2.19/fixmatch.semisupervised/0008.bf7da99b/modules/model.0000029.pth',
        r'/home/yanghaozhe/jupyter_data/thexp-implement2/.thexp/experiments/thexp-implement2.19/fixmatch.semisupervised/0008.bf7da99b/modules/model.0000037.pth',
        r'/home/yanghaozhe/jupyter_data/thexp-implement2/.thexp/experiments/thexp-implement2.19/fixmatch.semisupervised/0008.bf7da99b/modules/model.0000045.pth',
        r'/home/yanghaozhe/jupyter_data/thexp-implement2/.thexp/experiments/thexp-implement2.19/fixmatch.semisupervised/0008.bf7da99b/modules/model.0000055.pth',
        r'/home/yanghaozhe/jupyter_data/thexp-implement2/.thexp/experiments/thexp-implement2.19/fixmatch.semisupervised/0008.bf7da99b/modules/model.0000073.pth',
        r'/home/yanghaozhe/jupyter_data/thexp-implement2/.thexp/experiments/thexp-implement2.19/fixmatch.semisupervised/0008.bf7da99b/modules/model.0000107.pth',
        r'/home/yanghaozhe/jupyter_data/thexp-implement2/.thexp/experiments/thexp-implement2.19/fixmatch.semisupervised/0008.bf7da99b/modules/model.0000107.pth',
        r'/home/yanghaozhe/jupyter_data/thexp-implement2/.thexp/experiments/thexp-implement2.19/fixmatch.semisupervised/0008.bf7da99b/modules/model.0000202.pth', ]

    model_names = [
        r'/home/yanghaozhe/jupyter_data/thexp-implement2/.thexp/experiments/thexp-implement2.19/fixmatch.semisupervised/0008.bf7da99b/modules/model.0000750.pth']

    for model_fn in reversed(model_names):
        trainer.load_model(model_fn)
        trainer.test()
        # trainer.train()
