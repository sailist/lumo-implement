import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from thexp import Params,Trainer,callbacks


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
params = Params()

('model architecture: ' +
 ' | '.join(model_names) +
 ' (default: resnet18)')
params.arch = 'resnet18'
params.workers = 4  # number of data loading workers (default: 4)
params.epochs = 90  # number of total epochs to run
params.start_epoch = 0  # manual epoch number (useful on restarts)

'mini-batch size (default: 256), this is the total '
'batch size of all GPUs on the current node when '
'using Data Parallel or Distributed Data Parallel'
params.batch_size = 12
params.learning_rate = 0.1
params.lr = params.learning_rate  # initial learning rate
params.momentum = 0.9  # momentum
params.weight_decay = 1e-4  # weight decay (default: 1e-4)
params.print_freq = 10  # print frequency (default: 10)
params.resume = ''  # path to latest checkpoint (default: none)
params.evaluate = False  # evaluate model on validation set
params.pretrained = False  # use pre-trained model
params.world_size = -1  # number of nodes for distributed training
params.rank = -1  # node rank for distributed training
# parser.add_argument('--dist_url', default='tcp://localhost:23456', type=str,
#                     help='url used to set up distributed training')
params.dist_url = 'env://'  # url used to set up distributed training
params.dist_backend = 'nccl'  # distributed backend
params.seed = None  # seed for initializing training.
params.gpu = None  # GPU id to use.

'Use multi-processing distributed training to launch '
'N processes per node, which has N GPUs. This is the '
'fastest way to use PyTorch for either single node or '
'multi node data parallel training'
params.multiprocessing_distributed = True
params = params.from_args()
best_acc1 = 0
print(params)

def main():
    if params.seed is not None:
        random.seed(params.seed)
        torch.manual_seed(params.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if params.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if params.dist_url == "env://" and params.world_size == -1:
        # params.world_size = int(os.environ["WORLD_SIZE"])
        params.world_size = 1

    params.distributed = params.world_size > 1 or params.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if params.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        params.world_size = ngpus_per_node * params.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, params))
    else:
        # Simply call main_worker function
        main_worker(params.gpu, ngpus_per_node, params)


class MyTrainer(Trainer):

    def callbacks(self, params: Params):
        super().callbacks(params)
        callbacks.LoggerCallback().hook(self)

    def models(self, params: Params):
        super().models(params)
        if params.pretrained:
            print("=> using pre-trained model '{}'".format(params.arch))
            model = models.__dict__[params.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(params.arch))
            model = models.__dict__[params.arch]()
        if params.gpu is not None:
            torch.cuda.set_device(params.gpu)
            model.cuda(params.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            params.batch_size = int(params.batch_size / params.ngpus_per_node)
            params.workers = int((params.workers + params.ngpus_per_node - 1) / params.ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[params.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)


def main_worker(gpu, ngpus_per_node, params):
    global best_acc1
    params.gpu = gpu

    if params.gpu is not None:
        print("Use GPU: {} for training".format(params.gpu))

    if params.distributed:
        if params.dist_url == "env://" and params.rank == -1:
            params.rank = int(os.environ["RANK"])
        if params.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            params.rank = params.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=params.dist_backend, init_method=params.dist_url,
                                world_size=params.world_size, rank=params.rank)
    # create model
    if params.pretrained:
        print("=> using pre-trained model '{}'".format(params.arch))
        model = models.__dict__[params.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(params.arch))
        model = models.__dict__[params.arch]()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif params.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if params.gpu is not None:
            torch.cuda.set_device(params.gpu)
            model.cuda(params.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            params.batch_size = int(params.batch_size / ngpus_per_node)
            params.workers = int((params.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[params.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif params.gpu is not None:
        torch.cuda.set_device(params.gpu)
        model = model.cuda(params.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if params.arch.startswith('alexnet') or params.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(params.gpu)

    optimizer = torch.optim.SGD(model.parameters(), params.lr,
                                momentum=params.momentum,
                                weight_decay=params.weight_decay)

    # optionally resume from a checkpoint
    if params.resume:
        if os.path.isfile(params.resume):
            print("=> loading checkpoint '{}'".format(params.resume))
            if params.gpu is None:
                checkpoint = torch.load(params.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(params.gpu)
                checkpoint = torch.load(params.resume, map_location=loc)
            params.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if params.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(params.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(params.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(params.resume))

    cudnn.benchmark = True

    # Data loading code
    train_dataset = datasets.FakeData(transform=transforms.ToTensor())

    if params.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params.batch_size, shuffle=(train_sampler is None),
        num_workers=params.workers, pin_memory=True, sampler=train_sampler)

    for epoch in range(params.start_epoch, params.epochs):
        if params.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, params)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, params)
        print(epoch)
        # evaluate on validation set

        # remember best acc@1 and save checkpoint


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

    from torchvision.datasets import ImageFolder
