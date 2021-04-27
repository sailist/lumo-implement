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
def webvision_subcls(train=True, cls=50, per_cls=-1):
    dataset_root = os.path.join(root, 'webvision')
    xs = []
    ys = []
    if train:
        with open(os.path.join(dataset_root, 'info', 'train_filelist_google.txt'), 'r', encoding='utf-8') as r:
            for line in r:
                x, y = line.split(' ')
                y = int(y)
                if y == cls:
                    break
                x = os.path.join(os.path.abspath(dataset_root), x)
                xs.append(x)
                ys.append(y)
    else:
        with open(os.path.join(dataset_root, 'info', 'val_filelist.txt'), 'r', encoding='utf-8') as r:
            for line in r:
                x, y = line.split(' ')
                y = int(y)
                if y == cls:
                    break
                x = os.path.join(os.path.abspath(dataset_root), 'val_images_256', x)
                xs.append(x)
                ys.append(y)

    return llist(xs), np.array(ys)
