from torch import nn
from torch.nn import modules


class MultiHead(modules.Module):
    def __init__(self, feature_dim, n_classes, head=2):
        super().__init__()
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.head = head

        for i in range(head):
            self.__setattr__("fc{}".format(i), modules.Linear(feature_dim, n_classes))

    def forward(self, xs):
        return [self.__getattr__('fc{}'.format(i))(xs) for i in range(self.head)]

