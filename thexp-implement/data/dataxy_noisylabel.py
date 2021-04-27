from torchvision.datasets import ImageFolder
import numpy as np
import os
from thexp.decorators import regist_func
from thexp.base_classes import llist
from thexp import globs
from typing import Dict, Callable, Tuple

root = globs['datasets']

datasets = {
    # 'cifar10': cifar10,
}  # type:Dict[str,Callable[[bool],Tuple[llist,llist]]]


@regist_func(datasets)
def clothing1m(train=True):
    if train:
        dataset = ImageFolder(root=os.path.join(root, 'clothing1m', 'noisy_train'))
    else:
        dataset = ImageFolder(root=os.path.join(root, 'clothing1m', 'clean_test'))
    xs, ys = list(zip(*dataset.imgs))
    return llist(xs), np.array(ys)


@regist_func(datasets)
def clothing1m_balance(train=True, per_cls=15000):
    if train:
        dataset = ImageFolder(root=os.path.join(root, 'clothing1m', 'noisy_train'))
    else:
        dataset = ImageFolder(root=os.path.join(root, 'clothing1m', 'clean_test'))
    xs, ys = list(zip(*dataset.imgs))
    xs, ys = llist(xs), np.array(ys)
    n_classes = len(set(ys))
    if train:
        mask = np.zeros(ys.shape, dtype=np.bool)
        for i in range(n_classes):
            _argids = np.where(ys == i)[0]
            selected = _argids[np.random.permutation(len(_argids))[:per_cls]]
            mask[selected] = True

        xs = xs[mask]
        ys = ys[mask]

    return xs, ys


@regist_func(datasets)
def clothing1m_clean_train():
    dataset = ImageFolder(root=os.path.join(root, 'clothing1m', 'clean_train'))
    xs, ys = list(zip(*dataset.imgs))
    return llist(xs), np.array(ys)


def clothing1m_clean_dividimix_train():
    from thexp import RndManager
    rnd = RndManager()
    rnd.mark("assign")
    xs, ys = clothing1m_balance(True, 6720)
    mix_fn = os.path.join(os.path.dirname(__file__), '19.1.npy')
    clean_res = np.load(mix_fn) > 0.5
    return xs[clean_res], ys[clean_res]


def clothing1m_clean_ema_train():
    from thexp import RndManager
    import torch
    rnd = RndManager()
    rnd.mark("assign")
    xs, ys = clothing1m_balance(True, 10080)
    mix_fn = os.path.join(os.path.dirname(__file__), 'filter_mem_20.pth')
    clean_res = (torch.load(mix_fn, map_location='cpu') > 0.5).detach().numpy()
    return xs[clean_res], ys[clean_res]


def webvision(train=True,per_cls=-1):
    pass