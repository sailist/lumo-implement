import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim

from thexp import Experiment


def example(rank, world_size, exp, func):
    print(rank)
    func()
    # create default process group
    device = rank
    dist.init_process_group("nccl", init_method='tcp://localhost:12345', rank=device, world_size=world_size)
    # create local model
    model = nn.Linear(10, 10).to(device)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[device])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    print(exp.test_dir)
    # forward pass
    for i in range(20):
        outputs = ddp_model(torch.randn(20, 10).to(device))
        labels = torch.randn(20, 10).to(device)
        # backward pass
        loss = loss_fn(outputs, labels)
        loss.backward()
        # update parameters
        optimizer.step()
        print(rank, device, world_size, i, loss, list(ddp_model.parameters())[0][0])


def func():
    print('123')


def main():
    world_size = 2
    exp = Experiment('mp_test')
    mp.spawn(example,
             args=(world_size, exp, func),
             nprocs=world_size, daemon=True)


if __name__ == "__main__":
    main()
