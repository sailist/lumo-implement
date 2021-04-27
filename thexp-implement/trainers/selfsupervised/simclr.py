"""
cifar10
0009.5a1be50c, wideresnet282
/home/share/yanghaozhe/experiments/thexp-implement2.19/simclr.selfsupervised/0009.5a1be50c/modules/model.0000000.pth
0010.de6554b7，wideresnet282,batchsize=512

cifar10
0013.b80fb070, preresnet18, batchsize=256
/home/share/yanghaozhe/experiments/thexp-implement2.19/simclr.selfsupervised/0013.b80fb070/modules/model.0000000.pth

cifar100
0012.5bd4dad9, preresnet18, batchsize=256
/home/share/yanghaozhe/experiments/thexp-implement2.19/simclr.selfsupervised/0012.5bd4dad9/modules/model.0000000.pth


"""

if __name__ == '__main__':
    import os
    import sys

    chdir = os.path.dirname(os.path.abspath(__file__))
    chdir = os.path.dirname(chdir)
    chdir = os.path.dirname(chdir)
    sys.path.append(chdir)

import torch
from thextra import memory_bank
from thexp import Trainer, Meter, Params, AvgMeter
from trainers import SimCLRParams, GlobalParams
from trainers.mixin import *


class SimclrTrainer(callbacks.BaseCBMixin,
                    datasets.Base32Mixin,
                    models.SimCLRMixin,
                    losses.CELoss, losses.SimCLRLoss,
                    callbacks.callbacks.TrainCallback,
                    Trainer):

    def test_eval_logic(self, dataloader, param: Params):
        with torch.no_grad():
            meter = AvgMeter()

            for batch in dataloader:
                xs, ys = batch

                output = self.predict(xs)
                output = self.memory_bank_base.weighted_knn(output)

                acc1 = 100 * torch.mean(torch.eq(output, ys).float())
                meter.acc = acc1.item() / xs.size(0)

            return meter

    def initial(self):
        super().initial()
        torch.cuda.set_device(self.device)
        self.memory_bank_base = memory_bank.MemoryBank(self.train_size,
                                                       params.feature_dim,
                                                       params.n_classes, params.temperature, self.device).cuda()

        self.memory_bank_val = memory_bank.MemoryBank(self.train_size,
                                                      params.feature_dim,
                                                      params.n_classes, params.temperature, self.device).cuda()

    def callbacks(self, params: GlobalParams):
        super().callbacks(params)
        self.hook(self)

    def on_train_epoch_end(self, trainer: Trainer, func, params: Params, meter: Meter, *args, **kwargs):
        super().on_train_epoch_end(trainer, func, params, meter, *args, **kwargs)
        self.fill_memory_bank(self.eval_dataloader, self.memory_bank_base)

        # topk = 20
        # self.logger.info('Mine the nearest neighbors (Top-%d)' % (topk))
        # indices, acc = self.memory_bank_base.mine_nearest_neighbors(topk)
        # self.logger.info('Accuracy of top-%d nearest neighbors on train set is %.2f' % (topk, 100 * acc))
        # # np.save(p['topk_neighbors_train_path'], indices)
        #
        # # Mine the topk nearest neighbors at the very end (Val)
        # # These will be used for validation.
        # self.fill_memory_bank(self.eval_dataloader, self.memory_bank_val)
        # topk = 5
        # self.logger.info('Mine the nearest neighbors (Top-%d)' % (topk))
        # indices, acc = self.memory_bank_val.mine_nearest_neighbors(topk)
        # self.logger.info('Accuracy of top-%d nearest neighbors on val set is %.2f' % (topk, 100 * acc))
        # np.save(p['topk_neighbors_val_path'], indices)

    def fill_memory_bank(self, loader, memory_bank):
        with torch.no_grad():
            self.change_mode(False)
            memory_bank.reset()

            for i, batch in enumerate(loader):
                images, targets = batch
                output = self.to_logits(images)
                memory_bank.update(output, targets)
                if i % 100 == 0:
                    print('Fill Memory Bank [%d/%d]' % (i, len(loader)))

            self.change_mode(True)

    def train_batch(self, eidx, idx, global_step, batch_data, params: SimCLRParams, device: torch.device):
        super().train_batch(eidx, idx, global_step, batch_data, params, device)
        meter = Meter()
        _, xs, axs, _ = batch_data  # type:torch.Tensor

        b, c, h, w = xs.size()
        input_ = torch.cat([xs.unsqueeze(1), axs.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w)

        output = self.to_logits(input_).view(b, 2, -1)
        meter.Lall = meter.Lall + self.loss_sim_(output, params.temperature,
                                                 device=device, meter=meter)

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


if __name__ == '__main__':
    params = SimCLRParams()
    params.preresnet18()
    params.mid_dim = 512
    params.device = 'cuda:2'
    params.from_args()

    trainer = SimclrTrainer(params)
    trainer.train()
    trainer.saver.save_model(0, trainer.model.backbone.state_dict())
