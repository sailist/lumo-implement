from thexp import Params, globs
import os
from thexp.globals import _GITKEY


# os.environ['TMPDIR'] = globs['TMPDIR']


class GlobalParams(Params):

    def __init__(self):
        super().__init__()
        self.epoch = 400

        self.optim = self.create_optim('SGD',
                                       lr=0.05,
                                       momentum=0.9,
                                       weight_decay=5e-4,
                                       nesterov=True)

        self.eval_test_per_epoch = (5, 10)

        self.dataset = self.choice('dataset',
                                   'cifar10', 'cifar100', 'mnist', 'fashionmnist', 'svhn',
                                   'clothing1m', 'clothing1m_balance',
                                   'webvision',
                                   )
        self.dataset_args = ((), {})
        self.n_classes = 10
        self.topk = (1, 2, 3, 4)

        self.batch_size = 64
        self.num_workers = 4

        self.ema = True
        self.ema_alpha = 0.999

        self.val_size = 5000

        self.architecture = self.choice('architecture', 'WRN', 'Resnet', 'Lenet', 'PreResnet', 'WN',
                                        'inception')
        self.depth = 28  # for both wideresnet and resnet
        self.widen_factor = 2  # for wideresnet
        self.drop_rate = 0
        self.pretrain = False
        self.with_fc = True
        self.tmp_dir = '/home/share/tmp'

        self.distributed = False
        self.local_rank = -1

    def wideresnet282(self):
        """model for semisupervised"""
        self.architecture = 'WRN'
        self.depth = 28
        self.widen_factor = 2

    def wideresnet28_10(self):
        self.architecture = 'WRN'
        self.depth = 28
        self.widen_factor = 10

    def widenet282(self):
        self.architecture = 'WN'
        self.depth = 28
        self.widen_factor = 2

    def preresnet18(self):
        self.architecture = 'PreResnet'
        self.depth = 18

    def resnet32(self):
        """model for noisy label"""
        self.architecture = 'Resnet'
        self.depth = 32

    def inception50(self):
        self.architecture = 'inception'
        self.n_classes = 50

    def resnet50(self):
        """model for noisy label"""
        self.architecture = 'Resnet'
        self.pretrain = True
        self.depth = 50

    def initial(self):
        if self.dataset in {'cifar100'}:
            self.n_classes = 100
        elif 'clothing1m' in self.dataset:
            self.n_classes = 14

        if self.ENV.IS_PYCHARM_DEBUG:
            self.num_workers = 0

        self.lr_sche = self.SCHE.Cos(start=self.optim.args.lr,
                                     end=0.00001,
                                     right=self.epoch)


class SupervisedParams(GlobalParams):
    def __init__(self):
        super().__init__()
        self.epoch = 600
        self.batch_size = 128
        self.optim = self.create_optim('SGD',
                                       lr=0.06,
                                       momentum=0.9,
                                       weight_decay=5e-4,
                                       nesterov=True)

    def iter_baseline(self):
        self.architecture = 'WRN'
        for p in self.grid_search('dataset', ['cifar10', 'cifar100']):
            p.depth = 28
            for pp in p.grid_search('widen_factor', [2, 10]):
                yield pp
        self.architecture = 'Resnet'
        for p in self.grid_search('dataset', ['cifar10', 'cifar100']):
            for pp in p.grid_search('depth', [20, 32, 44, 50, 56]):
                yield pp


class SimCLRParams(GlobalParams):
    def __init__(self):
        super().__init__()
        self.epoch = 500
        self.temperature = 0.07
        self.batch_size = 256
        self.with_fc = False
        self.feature_dim = 128
        self.mid_dim = 128

        self.optim = self.create_optim('SGD',
                                       weight_decay=0.0001,
                                       momentum=0.9,
                                       lr=0.04)

    def initial(self):
        super().initial()
        if self.dataset == 'cifar100':
            self.preresnet18()
            self.mid_dim = 512


class SupConParams(GlobalParams):
    def __init__(self):
        super().__init__()
        self.epoch = 500
        self.temperature = 0.07
        self.batch_size = 256
        self.with_fc = False
        self.feature_dim = 128
        self.mid_dim = 128

        self.optim = self.create_optim('SGD',
                                       weight_decay=0.0001,
                                       momentum=0.9,
                                       lr=0.04)

    def initial(self):
        super().initial()
        if self.dataset == 'cifar100':
            self.preresnet18()
            self.mid_dim = 512


class SemiSupervisedParams(GlobalParams):
    def __init__(self):
        super().__init__()
        self.epoch = 2048
        self.batch_size = 64
        self.optim = self.create_optim('SGD',
                                       lr=0.03,
                                       momentum=0.9,
                                       weight_decay=5e-4,
                                       nesterov=True)

        self.uratio = 7
        self.pred_thresh = 0.95
        self.n_percls = 400

    def iter_baseline(self):
        self.architecture = 'WRN'
        self.depth = 28
        self.widen_factor = 2
        for p in self.grid_search('dataset', ['cifar10', 'cifar100', 'svhn']):
            yield p


class NoisyParams(GlobalParams):
    def __init__(self):
        super().__init__()
        self.epoch = 600
        self.batch_size = 128

        self.noisy_ratio = 0.8
        self.noisy_type = self.choice('noisy_type', 'symmetric', 'asymmetric')

        self.optim = self.create_optim('SGD',
                                       lr=0.3,
                                       momentum=0.9,
                                       weight_decay=5e-4,
                                       nesterov=True)

        self.l2_decay = 1e-4
        self.kl_factor = 20
        self.ce_factor = 5
        self.n_classes = 10
        self.sharpen_T = 0.5
        self.mix_beta = 0.75

        self.ss_pretrain = False
        # self.ss_pretrain_fn = '/home/share/yanghaozhe/experiments/thexp-implement2.19/simclr.selfsupervised/0009.5a1be50c/modules/model.0000000.pth'
        self.ss_pretrain_fn = '/home/share/yanghaozhe/experiments/thexp-implement2.19/simclr.selfsupervised/0010.de6554b7/modules/model.0000000.pth'

        # cifar100, preresnet18
        # self.ss_pretrain_fn = '/home/share/yanghaozhe/experiments/thexp-implement2.19/simclr.selfsupervised/0012.5bd4dad9/modules/model.0000000.pth'

        self.large_model = False
        self.wrn2810 = False

        self.order_sampler = False

    def initial(self):
        super().initial()
        self.init_eps_val = (1 / self.batch_size)
        self.plabel_sche = self.SCHE.Cos(0.1, 1, right=self.epoch // 2)
        self.gmm_w_sche = self.SCHE.Log(0.5, 1, right=self.epoch // 2)
        self.contrast_sche = self.SCHE.Cos(0.0001, end=1, left=0, right=self.epoch // 2)
        self.lr_sche = self.SCHE.Cos(start=self.optim.args.lr,
                                     end=0.0001,
                                     right=400)

        if self.dataset == 'cifar100' or self.large_model:
            self.preresnet18()
        if self.wrn2810:
            self.wideresnet28_10()


class MnistImblanceParams(GlobalParams):

    def __init__(self):
        super().__init__()

        self.optim = self.create_optim('SGD',
                                       lr=1e-3,
                                       momentum=0.9,
                                       weight_decay=5e-4,
                                       nesterov=True)
        self.epoch = 25  # is enough for mnist
        self.batch_size = 100
        self.train_classes = [9, 4]
        self.train_proportion = 0.995
        self.val_size = 1000
        self.ema = False


class FewshotParams(GlobalParams):
    def __init__(self):
        super().__init__()
        self.dataset = self.choice('dataset', 'omniglot', 'miniimagenet')
        self.n_way = 5
        self.k_shot = 15
        self.k_query = 5

        self.task_num = 32  # meta batch size, namely task num
        self.meta_lr = 0.4
        self.update_step = 5  # task-level inner update steps
