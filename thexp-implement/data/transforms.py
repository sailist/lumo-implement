from torchvision.transforms import ToTensor
from PIL import Image
from torchvision import transforms
from thexp.contrib.data.augments.image import RandAugmentMC
from .autoaugment import ImageNetPolicy


def read(x):
    return Image.open(x).convert('RGB')


class BigWeak(object):

    def __init__(self, mean=None, std=None) -> None:
        super().__init__()
        lis = [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            # transforms.RandomResizedCrop(224, scale=(0.3, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        if mean is not None and std is not None:
            lis.append(transforms.Normalize(mean=mean, std=std))
        self.transform = transforms.Compose(lis)

    def __call__(self, x):
        x = read(x)
        return self.transform(x)


class BigStrong(object):

    def __init__(self, mean=None, std=None) -> None:
        super().__init__()
        lis = [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            # RandAugmentMC(n=2, m=10),
            ImageNetPolicy(),
            transforms.ToTensor(),
        ]
        if mean is not None and std is not None:
            lis.append(transforms.Normalize(mean=mean, std=std))
        self.transform = transforms.Compose(lis)

    def __call__(self, x):
        x = read(x)
        return self.transform(x)


class BigWeak2(object):

    def __init__(self, mean=None, std=None) -> None:
        super().__init__()
        lis = [
            transforms.Resize(320),
            transforms.RandomResizedCrop(299),
            # transforms.RandomResizedCrop(224, scale=(0.3, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        if mean is not None and std is not None:
            lis.append(transforms.Normalize(mean=mean, std=std))
        self.transform = transforms.Compose(lis)

    def __call__(self, x):
        x = read(x)
        return self.transform(x)


class BigStrong2(object):

    def __init__(self, mean=None, std=None) -> None:
        super().__init__()
        lis = [
            transforms.Resize(320),
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            # RandAugmentMC(n=2, m=10),
            ImageNetPolicy(),
            transforms.ToTensor(),
        ]
        if mean is not None and std is not None:
            lis.append(transforms.Normalize(mean=mean, std=std))
        self.transform = transforms.Compose(lis)

    def __call__(self, x):
        x = read(x)
        return self.transform(x)


class BigToTensor2(object):

    def __init__(self, mean=None, std=None) -> None:
        super().__init__()
        lis = [
            transforms.Resize(320),
            transforms.RandomResizedCrop(299),
            transforms.ToTensor(),
        ]
        if mean is not None and std is not None:
            lis.append(transforms.Normalize(mean=mean, std=std))
        self.transform = transforms.Compose(lis)

    def __call__(self, x):
        x = read(x)
        return self.transform(x)


class BigToTensor(object):

    def __init__(self, mean=None, std=None) -> None:
        super().__init__()
        lis = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
        if mean is not None and std is not None:
            lis.append(transforms.Normalize(mean=mean, std=std))
        self.transform = transforms.Compose(lis)

    def __call__(self, x):
        x = read(x)
        return self.transform(x)


class Weak(object):
    def __init__(self, mean=None, std=None):
        lis = [
            transforms.RandomCrop(size=32,
                                  padding=4,
                                  padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        if mean is not None and std is not None:
            lis.append(transforms.Normalize(mean=mean, std=std))
        self.weak = transforms.Compose(lis)

    def __call__(self, x):
        return self.weak(x)


class Strong():
    def __init__(self, mean=None, std=None) -> None:
        lis = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=4,
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
        ]
        if mean is not None and std is not None:
            lis.append(transforms.Normalize(mean=mean, std=std))
        self.weak = transforms.Compose(lis)
        self.strong = transforms.Compose(lis)

    def __call__(self, x):
        return self.strong(x)


class ToNormTensor():
    def __init__(self, mean=None, std=None):
        if mean is not None and std is not None:
            self.norm = transforms.Normalize(mean=mean, std=std)
        else:
            self.norm = None
        self.totensor = transforms.ToTensor()

    def __call__(self, x):
        val = self.totensor(x)
        if self.norm is not None:
            return self.norm(val)
        else:
            return val
