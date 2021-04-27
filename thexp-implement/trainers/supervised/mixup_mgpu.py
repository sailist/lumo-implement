"""
"""
if __name__ == '__main__':
    import os
    import sys

    chdir = os.path.dirname(os.path.abspath(__file__))
    chdir = os.path.dirname(chdir)
    chdir = os.path.dirname(chdir)
    sys.path.append(chdir)

import torch
from thexp import Trainer, Meter, Params
from trainers import SupervisedParams
from trainers.mixin import *
import torch.distributed as dist


class SupervisedTrainer(callbacks.BaseCBMixin,
                        datasets.Base32Mixin,
                        models.BaseModelMixin,
                        acc.ClassifyAccMixin,
                        losses.CELoss, losses.MixMatchLoss,
                        callbacks.callbacks.TrainCallback,
                        Trainer):

    def train_epoch(self, eidx: int, params: SupervisedParams):
        from thexp import AvgMeter
        avg = AvgMeter()
        self.change_mode(True)
        print('epoch', params.local_rank, self._databundler_dict)
        print('epoch', params.local_rank, self.train_dataloader)
        for idx, batch_data in enumerate(self.train_dataloader):
            meter = self.train_batch(eidx, idx, self.params.global_step, batch_data, params, self.device)
            avg.update(meter)
            # del meter

            params.global_step += 1
            params.idx = idx
            if self.train_epoch_toggle:
                self.train_epoch_toggle = False
                break

        self.change_mode(False)
        return avg

    def train_batch(self, eidx, idx, global_step, batch_data, params: SupervisedParams, device: torch.device):
        super().train_batch(eidx, idx, global_step, batch_data, params, device)
        meter = Meter()
        ids, xs, axs, ys = batch_data  # type:torch.Tensor

        targets = tricks.onehot(ys, params.n_classes)
        mixed_xs, mixed_targets = self.mixup_(xs, targets)

        mixed_logits = self.to_logits(mixed_xs)
        meter.Lall = meter.Lall + self.loss_ce_with_targets_(mixed_logits, mixed_targets,
                                                             meter=meter)

        with torch.no_grad():
            self.acc_precise_(self.to_logits(xs).argmax(dim=1), ys,
                              meter=meter, name='acc')

        self.optim.zero_grad()
        meter.Lall.backward()
        self.optim.step()

        return meter

    def to_logits(self, xs) -> torch.Tensor:
        return self.model(xs)


# def main_worker(gpu, ngpus_per_node, args):
#     trainer = SupervisedTrainer(params)
#     trainer.train()

def run(rank, params):
    params.from_args()
    params.local_rank = rank
    print(rank)
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:12345',
                            rank=rank,
                            world_size=params.world_size)
    torch.cuda.set_device(params.local_rank)
    params.device = 'cuda:{}'.format(params.local_rank)
    trainer = SupervisedTrainer(params)
    trainer.train()


def op(trainer):
    trainer.train()


if __name__ == '__main__':
    params = SupervisedParams()
    params.wideresnet282()
    params.batch_size = 4
    params.device = 'cuda:1'
    params.distributed = True
    params.ema = False
    import os
    import torch.multiprocessing as mp

    params.init_method = 'tcp://localhost:12345'

    # mp.spawn(run, args=(params,), nprocs=ngpus_per_node)
    # run(0, params)

    from thexp.frame.trainer import DistributedTrainer

    dist_trainer = DistributedTrainer(SupervisedTrainer, params, op)
    # trainer = SupervisedTrainer(params)
    # trainer.train()

    dist_trainer.run()
