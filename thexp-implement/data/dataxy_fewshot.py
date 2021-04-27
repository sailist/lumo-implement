from torchvision.datasets.omniglot import Omniglot
from thexp.decorators import regist_func
from thexp.base_classes import llist
from thexp import globs
from typing import Dict, Callable, Tuple

root = globs['datasets']

datasets = {
    # 'cifar10': cifar10,
}  # type:Dict[str,Callable[[bool],Tuple[llist,llist]]]


@regist_func(datasets)
def omniglot(train=True):
    dataset = Omniglot(root=root, background=train)
    image_name, character_class = list(zip(*dataset._flat_character_images))
    return image_name, character_class
