from thexp import Trainer

import numpy as np
from trainers import *
from thexp.contrib.data import splits
from thexp import DatasetBuilder, DataBundler

from data.constant import norm_val
from data.transforms import ToTensor
from data.dataxy import datasets
from data.transforms import Weak, Strong, ToNormTensor


class Base32Mixin(Trainer):
    """
    all 32*32 dataset, including cifar10, cifar100, svhn

    use base train data shape: ids, xs, aug_xs, ys

    test data shape: xs, ys
    """

    def datasets(self, params: GlobalParams):
        dataset_fn = datasets[params.dataset]

        test_x, testy = dataset_fn(False)
        train_x, train_y = dataset_fn(True)

        train_idx, val_idx = splits.train_val_split(train_y, val_size=params.val_size)

        mean, std = norm_val.get(params.dataset, [None, None])
        toTensor = ToNormTensor(mean, std)
        weak = Weak(mean, std)
        strong = Strong(mean, std)

        test_dataloader = (
            DatasetBuilder(test_x, testy)
                .add_x(transform=toTensor)
                .add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers)
        )

        train_dataloader = (
            DatasetBuilder(train_x, train_y)
                .toggle_id()
                .add_x(transform=weak)
                .add_x(transform=strong)
                .add_y()
                .subset(train_idx)
        )

        if params.distributed:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(train_dataloader)
        else:
            sampler = None
        self.train_size = len(train_dataloader)
        train_dataloader = train_dataloader.DataLoader(batch_size=params.batch_size,
                                                       num_workers=params.num_workers,
                                                       sampler=sampler,
                                                       shuffle=not params.distributed)

        val_datalaoder = (
            DatasetBuilder(train_x, train_y)
                .add_x(transform=toTensor).add_y()
                .subset(val_idx)
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers)
        )

        self.regist_databundler(train=train_dataloader,
                                eval=val_datalaoder,
                                test=test_dataloader)
        print('dataloader in rank {}'.format(self.params.local_rank))
        print(self.params.local_rank, self.train_dataloader)
        print(self.params.local_rank, self._databundler_dict)
        print(self.params.local_rank, train_dataloader)
        self.to(self.device)


class SupConsMixin(Trainer):
    """
    all 32*32 dataset, including cifar10, cifar100, svhn

    use base train data shape: ids, xs, aug_xs, ys

    test data shape: xs, ys
    """

    def datasets(self, params: GlobalParams):
        from torchvision import transforms
        dataset_fn = datasets[params.dataset]

        test_x, testy = dataset_fn(False)
        train_x, train_y = dataset_fn(True)

        train_idx, val_idx = splits.train_val_split(train_y, val_size=params.val_size)

        mean, std = norm_val.get(params.dataset, [None, None])
        toTensor = ToNormTensor(mean, std)
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        test_dataloader = (
            DatasetBuilder(test_x, testy)
                .add_x(transform=toTensor)
                .add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers)
        )

        train_dataloader = (
            DatasetBuilder(train_x, train_y)
                .toggle_id()
                .add_x(transform=train_transform)
                .add_x(transform=train_transform)
                .add_y()
                .subset(train_idx)
        )

        if params.distributed:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(train_dataloader)
        else:
            sampler = None
        self.train_size = len(train_dataloader)
        train_dataloader = train_dataloader.DataLoader(batch_size=params.batch_size,
                                                       num_workers=params.num_workers,
                                                       sampler=sampler,
                                                       shuffle=not params.distributed)

        val_datalaoder = (
            DatasetBuilder(train_x, train_y)
                .add_x(transform=toTensor).add_y()
                .subset(val_idx)
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers)
        )

        self.regist_databundler(train=train_dataloader,
                                eval=val_datalaoder,
                                test=test_dataloader)
        print('dataloader in rank {}'.format(self.params.local_rank))
        print(self.params.local_rank, self.train_dataloader)
        print(self.params.local_rank, self._databundler_dict)
        print(self.params.local_rank, train_dataloader)
        self.to(self.device)


class SyntheticNoisyMixin(Trainer):
    """
    supervised image dataset with synthetic noisy

    train: (ids, xs, axs, ys, nys)
    val: xs, ys
    test: xs, ys

    Note:
        the first 'ys' right label and is used to calculate accuracy, and will not be used to train model.
    """

    def datasets(self, params: GlobalParams):
        self.rnd.mark('noisy')

        params.noisy_type = params.default('symmetric', True)
        params.noisy_ratio = params.default(0.2, True)

        mean, std = norm_val.get(params.dataset, [None, None])
        toTensor = ToNormTensor(mean, std)
        weak = Weak(mean, std)
        strong = Strong(mean, std)

        dataset_fn = datasets[params.dataset]
        test_x, test_y = dataset_fn(False)
        train_x, train_y = dataset_fn(True)
        if params.val_size > 0:
            train_ids, val_ids = splits.train_val_split(train_y, val_size=5000)
            train_x, val_x = train_x[train_ids], train_x[val_ids]
            train_y, val_y = train_y[train_ids], train_y[val_ids]
            val_dataloader = (
                DatasetBuilder(val_x, val_y)
                    .add_x(transform=toTensor).add_y()
                    .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
            )
        else:
            val_dataloader = None

        if params.noisy_type == 'asymmetric':
            from data.noisy import asymmetric_noisy
            noisy_y = asymmetric_noisy(train_y, params.noisy_ratio, n_classes=params.n_classes)

        elif params.noisy_type == 'symmetric':
            from data.noisy import symmetric_noisy
            noisy_y = symmetric_noisy(train_y, params.noisy_ratio, n_classes=params.n_classes)

        else:
            assert False

        self.train_set = [train_x, train_y, noisy_y]

        self.logger.info('noisy acc = {}'.format((train_y == noisy_y).mean()))

        self.train_set = (
            DatasetBuilder(train_x, train_y)
                .add_labels(noisy_y, 'noisy_y')
                .toggle_id()
                .add_x(transform=weak)
                .add_x(transform=strong)
                .add_y()
                .add_y(source='noisy_y')
        )
        if params.distributed:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(self.train_set)
        else:
            sampler = None
        self.train_size = len(self.train_set)
        train_dataloader = self.train_set.DataLoader(batch_size=params.batch_size,
                                                     num_workers=params.num_workers,
                                                     drop_last=True,
                                                     sampler=sampler,
                                                     shuffle=True)

        test_dataloader = (
            DatasetBuilder(test_x, test_y)
                .add_x(transform=toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
        )

        self.regist_databundler(train=train_dataloader,
                                eval=val_dataloader,
                                test=test_dataloader)
        self.to(self.device)


class Clothing1mDatasetMixin(Trainer):
    """
    supervised image dataset with synthetic noisy

    train: (ids, xs, axs, ys, nys)
    val: xs, ys
    test: xs, ys

    Note:
        the first 'ys' right label and is used to calculate accuracy, and will not be used to train model.
    """

    def datasets(self, params: GlobalParams):
        self.rnd.mark('assign')
        from data.dataxy_noisylabel import clothing1m_balance, clothing1m
        from data.transforms import BigStrong, BigWeak, BigToTensor
        params.noisy_type = params.default('symmetric', True)
        params.noisy_ratio = params.default(0.2, True)
        params.cut_size = params.default(3360, True)

        mean, std = norm_val.get('clothing1m', [None, None])
        toTensor = BigToTensor(mean, std)
        weak = BigWeak(mean, std)
        strong = BigStrong(mean, std)

        dataset_fn = clothing1m_balance
        test_x, test_y = dataset_fn(False, per_cls=3360)
        train_x, noisy_y = dataset_fn(True, per_cls=params.cut_size)
        val_dataloader = None
        self.logger.info(train_x[:2])

        sub_ids = np.random.permutation(len(train_x))

        train_set = (
            DatasetBuilder(train_x, noisy_y)
                .toggle_id()
                .add_x(transform=weak)
                .add_x(transform=strong)
                .add_y()
                .subset(sub_ids)
        )
        self.train_set = train_set

        self.train_size = len(train_set)
        train_dataloader = train_set.DataLoader(batch_size=params.batch_size,
                                                num_workers=params.num_workers,
                                                drop_last=True,
                                                shuffle=False)

        test_dataloader = (
            DatasetBuilder(test_x, test_y)
                .add_x(transform=toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=False)
        )

        self.regist_databundler(train=train_dataloader,
                                eval=val_dataloader,
                                test=test_dataloader)
        self.to(self.device)


class WebVisionDatasetMixin(Trainer):
    """
    supervised image dataset with synthetic noisy

    train: (ids, xs, axs, ys, nys)
    val: xs, ys
    test: xs, ys

    Note:
        the first 'ys' right label and is used to calculate accuracy, and will not be used to train model.
    """

    def datasets(self, params: GlobalParams):
        self.rnd.mark('assign')
        from data.dataxy_webvision import webvision_subcls
        from data.transforms import BigStrong, BigWeak, BigToTensor, BigStrong2, BigWeak2, BigToTensor2
        params.noisy_type = params.default('symmetric', True)
        params.noisy_ratio = params.default(0.2, True)
        params.cut_size = params.default(3360, True)

        mean, std = norm_val.get('webvision', [None, None])
        toTensor = BigToTensor2(mean, std)
        weak = BigWeak2(mean, std)
        strong = BigStrong2(mean, std)

        dataset_fn = webvision_subcls
        test_x, test_y = dataset_fn(False)
        train_x, noisy_y = dataset_fn(True)
        val_dataloader = None
        self.logger.info(train_x[:2])

        sub_ids = np.random.permutation(len(train_x))

        train_set = (
            DatasetBuilder(train_x, noisy_y)
                .toggle_id()
                .add_x(transform=weak)
                .add_x(transform=strong)
                .add_y()
                .subset(sub_ids)
        )
        self.train_set = train_set

        self.train_size = len(train_set)
        train_dataloader = train_set.DataLoader(batch_size=params.batch_size,
                                                num_workers=params.num_workers,
                                                drop_last=True,
                                                shuffle=False)

        test_dataloader = (
            DatasetBuilder(test_x, test_y)
                .add_x(transform=toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=False)
        )

        self.regist_databundler(train=train_dataloader,
                                eval=val_dataloader,
                                test=test_dataloader)
        self.to(self.device)


class CleanClothing1mDatasetMixin(Trainer):
    """
    supervised image dataset with synthetic noisy

    train: (ids, xs, axs, ys, nys)
    val: xs, ys
    test: xs, ys

    Note:
        the first 'ys' right label and is used to calculate accuracy, and will not be used to train model.
    """

    def datasets(self, params: GlobalParams):
        self.rnd.mark('assign')
        from data.dataxy_noisylabel import (
            clothing1m_clean_train,
            clothing1m_balance,
            clothing1m_clean_dividimix_train,
            clothing1m_clean_ema_train)
        from data.transforms import BigStrong, BigWeak, BigToTensor
        params.noisy_type = params.default('symmetric', True)
        params.noisy_ratio = params.default(0.2, True)
        params.cut_size = params.default(3360, True)

        mean, std = norm_val.get('clothing1m', [None, None])
        toTensor = BigToTensor(mean, std)
        weak = BigWeak(mean, std)
        strong = BigStrong(mean, std)

        test_x, test_y = clothing1m_balance(False, per_cls=3360)
        # train_x, noisy_y = clothing1m_clean_train()
        train_x, noisy_y = clothing1m_clean_ema_train()
        val_dataloader = None
        self.logger.info(train_x[:2])

        sub_ids = np.random.permutation(len(train_x))

        train_set = (
            DatasetBuilder(train_x, noisy_y)
                .toggle_id()
                .add_x(transform=weak)
                .add_x(transform=strong)
                .add_y()
                .subset(sub_ids)
        )
        self.train_set = train_set

        self.train_size = len(train_set)
        train_dataloader = train_set.DataLoader(batch_size=params.batch_size,
                                                num_workers=params.num_workers,
                                                drop_last=True,
                                                shuffle=False)

        test_dataloader = (
            DatasetBuilder(test_x, test_y)
                .add_x(transform=toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=False)
        )

        self.regist_databundler(train=train_dataloader,
                                eval=val_dataloader,
                                test=test_dataloader)
        self.to(self.device)


class OrderSyntheticNoisyMixin(Trainer):
    """
    supervised image dataset with synthetic noisy

    train: (ids, xs, axs, ys, nys)
    val: xs, ys
    test: xs, ys

    Note:
        the first 'ys' right label and is used to calculate accuracy, and will not be used to train model.
    """

    def datasets(self, params: NoisyParams):
        self.rnd.mark('fix_noisy')
        params.noisy_type = params.default('symmetric', True)
        params.noisy_ratio = params.default(0.2, True)

        dataset_fn = datasets[params.dataset]
        test_x, test_y = dataset_fn(False)
        train_x, train_y = dataset_fn(True)

        train_ids, val_ids = splits.train_val_split(train_y, val_size=5000)

        train_x, val_x = train_x[train_ids], train_x[val_ids]
        train_y, val_y = train_y[train_ids], train_y[val_ids]

        mean, std = norm_val.get(params.dataset, [None, None])
        toTensor = ToNormTensor(mean, std)
        weak = Weak(mean, std)
        strong = Strong(mean, std)

        if params.noisy_type == 'asymmetric':
            from data.noisy import asymmetric_noisy
            noisy_y = asymmetric_noisy(train_y, params.noisy_ratio, n_classes=params.n_classes)

        elif params.noisy_type == 'symmetric':
            from data.noisy import symmetric_noisy
            noisy_y = symmetric_noisy(train_y, params.noisy_ratio, n_classes=params.n_classes)

        else:
            assert False
        clean_mask = train_y == noisy_y
        self.logger.info('noisy acc = {}'.format((train_y == noisy_y).mean()))
        self.rnd.shuffle()

        train_set = (
            DatasetBuilder(train_x, train_y)
                .add_labels(noisy_y, 'noisy_y')
                .toggle_id()
                .add_x(transform=weak)
                .add_x(transform=strong)
                .add_y()
                .add_y(source='noisy_y')
        )
        from thextra.noisy_sampler import NoisySampler
        sampler = None
        if params.order_sampler:
            sampler = NoisySampler(train_set, clean_mask)

        self.train_size = len(train_set)
        train_dataloader = train_set.DataLoader(batch_size=params.batch_size,
                                                num_workers=params.num_workers,
                                                drop_last=True,
                                                sampler=sampler,
                                                shuffle=True)

        val_dataloader = (
            DatasetBuilder(val_x, val_y)
                .add_x(transform=toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
        )

        test_dataloader = (
            DatasetBuilder(test_x, test_y)
                .add_x(transform=toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
        )

        self.regist_databundler(train=train_dataloader,
                                eval=val_dataloader,
                                test=test_dataloader)
        self.to(self.device)


class IEGSyntheticNoisyMixin(Trainer):
    """
    supervised image dataset with synthetic noisy

    train: (ids, xs, axs*K, ys,), (vxs, vys)
    val: no val dataset
    test: xs, ys

    Note:
        the first 'ys' right label and is used to calculate accuracy, and will not be used to train model.
    """

    def datasets(self, params: NoisyParams):
        params.noisy_type = params.default('symmetric', True)
        params.noisy_ratio = params.default(0.2, True)

        import numpy as np
        dataset_fn = datasets[params.dataset]
        test_x, test_y = dataset_fn(False)
        train_x, train_y = dataset_fn(True)

        # train_ids, query_ids = splits.train_val_split(train_y, val_size=params.val_size)

        query_ids, train_ids, eval_ids = splits.semi_split(train_y, params.query_size // params.n_classes,
                                                           val_size=params.val_size,
                                                           repeat_sup=False)

        # train_ids = train_ids[:3000]
        self.train_size = len(train_ids)
        train_x, query_x, eval_x = train_x[train_ids], train_x[query_ids], train_x[eval_ids]
        train_y, query_y, eval_y = train_y[train_ids], train_y[query_ids], train_y[eval_ids]

        mean, std = norm_val.get(params.dataset, [None, None])
        toTensor = ToNormTensor(mean, std)
        weak = Weak(mean, std)
        strong = Strong(mean, std)

        if params.noisy_type == 'asymmetric':
            from data.noisy import asymmetric_noisy
            noisy_y = asymmetric_noisy(train_y, params.noisy_ratio, n_classes=params.n_classes)

        elif params.noisy_type == 'symmetric':
            from data.noisy import symmetric_noisy
            noisy_y = symmetric_noisy(train_y, params.noisy_ratio, n_classes=params.n_classes)

        else:
            assert False

        self.logger.info('noisy acc = {}'.format((train_y == noisy_y).mean()))
        self.rnd.shuffle()

        self.logger.info(len(train_y), len(train_x), len(noisy_y))
        train_set = (
            DatasetBuilder(train_x, train_y)
                .add_labels(noisy_y, 'noisy_y')
                .toggle_id()
                .add_x(transform=strong)
        )

        params.K = params.default(0, True)
        for _ in range(params.K):
            train_set.add_x(transform=weak)

        train_set = (
            train_set
                .add_y()
                .add_y(source='noisy_y')
        )

        if params.distributed:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(train_set, num_replicas=4)
            self.sampler_a = sampler
        else:
            sampler = None
        train_set = train_set.DataLoader(batch_size=params.batch_size,
                                         num_workers=params.num_workers,
                                         sampler=sampler,
                                         shuffle=not params.distributed)

        query_set = (
            DatasetBuilder(query_x, query_y)
                .add_x(transform=strong)
                .add_y()
        )
        if params.distributed:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(train_set, num_replicas=4)
            self.sampler_b = sampler
        else:
            sampler = None
        query_set = query_set.DataLoader(batch_size=params.batch_size,
                                         num_workers=params.num_workers,
                                         sampler=sampler,
                                         shuffle=not params.distributed)

        val_dataloader = (
            DatasetBuilder(eval_x, eval_y)
                .add_x(transform=toTensor)
                .add_y()
                .DataLoader(batch_size=params.batch_size,
                            shuffle=False,  # do not shuffle # no shuffle for probe, so a batch is class balanced.(?)
                            num_workers=params.num_workers)
        )

        train_dataloader = DataBundler().add(train_set).cycle(query_set).zip_mode()

        test_dataloader = (
            DatasetBuilder(test_x, test_y)
                .add_x(transform=toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
        )

        self.regist_databundler(train=train_dataloader,
                                eval=val_dataloader,
                                test=test_dataloader)
        self.to(self.device)


class FixMatchDatasetMixin(Trainer):
    """
    semi-supervised image dataset for fixmatch


    sup, unsup = batch_data
    xs, ys = sup
    ids, un_xs, un_axs, un_ys = unsup

    """

    def datasets(self, params: SemiSupervisedParams):
        dataset_fn = datasets[params.dataset]

        test_x, test_y = dataset_fn(False)
        train_x, train_y = dataset_fn(True)

        indexs, un_indexs, val_indexs = splits.semi_split(train_y, n_percls=params.n_percls, val_size=params.val_size,
                                                          repeat_sup=False)
        self.logger.info('sup/unsup/val : {}'.format((len(indexs), len(un_indexs), len(val_indexs))))
        mean, std = norm_val.get(params.dataset, [None, None])
        toTensor = ToNormTensor(mean, std)
        weak = Weak(mean, std)
        strong = Strong(mean, std)

        sup_set = (
            DatasetBuilder(train_x, train_y)
                .add_x(transform=weak)
                .add_y()
                .subset(indexs)
        )
        if len(sup_set) < params.batch_size:
            sup_set.virtual_sample(params.batch_size)

        unsup_set = (
            DatasetBuilder(train_x, train_y)
                .toggle_id()
                .add_x(transform=weak)
                .add_x(transform=strong)
                .add_y()
                .subset(un_indexs)
        )
        self.cl_set = unsup_set

        sup_dataloader = sup_set.DataLoader(batch_size=params.batch_size, num_workers=params.num_workers,
                                            shuffle=True)
        self.sup_dataloader = sup_dataloader

        unsup_dataloader = unsup_set.DataLoader(batch_size=params.batch_size * params.uratio,
                                                num_workers=1,
                                                shuffle=True)

        self.unsup_dataloader = DataBundler().add(unsup_dataloader).to(self.device)

        val_dataloader = (
            DatasetBuilder(train_x[val_indexs], train_y[val_indexs])
                .add_x(transform=toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
        )

        test_dataloader = (
            DatasetBuilder(test_x, test_y)
                .add_x(transform=toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
        )

        self.regist_databundler(train=DataBundler().cycle(sup_dataloader).add(unsup_dataloader).zip_mode(),
                                eval=val_dataloader,
                                test=test_dataloader)
        self.to(self.device)


class FixMatchMNISTDatasetMixin(Trainer):
    """
    semi-supervised image dataset for fixmatch


    sup, unsup = batch_data
    xs, ys = sup
    ids, un_xs, un_axs, un_ys = unsup

    """

    def datasets(self, params: SemiSupervisedParams):
        from data.dataxy import mnist, fashionmnist

        test_x, test_y = mnist(False)
        train_x, train_y = mnist(True)

        indexs, un_indexs, val_indexs = splits.semi_split(train_y, n_percls=params.n_percls, val_size=params.val_size,
                                                          repeat_sup=False)
        self.logger.info('sup/unsup/val : {}'.format((len(indexs), len(un_indexs), len(val_indexs))))
        mean, std = norm_val.get('mnist', [None, None])
        toTensor = ToNormTensor(mean, std)

        sup_set = (
            DatasetBuilder(train_x, train_y)
                .add_x(transform=toTensor)
                .add_y()
                .subset(indexs)
        )
        if len(sup_set) < params.batch_size:
            sup_set.virtual_sample(params.batch_size)

        unsup_set = (
            DatasetBuilder(train_x, train_y)
                .toggle_id()
                .add_x(transform=toTensor)
                .add_x(transform=toTensor)
                .add_y()
                .subset(un_indexs)
        )
        self.cl_set = unsup_set

        sup_dataloader = sup_set.DataLoader(batch_size=params.batch_size, num_workers=params.num_workers,
                                            shuffle=True)
        self.sup_dataloader = sup_dataloader

        unsup_dataloader = unsup_set.DataLoader(batch_size=params.batch_size * params.uratio,
                                                num_workers=params.num_workers,
                                                shuffle=True)

        self.unsup_dataloader = DataBundler().add(unsup_dataloader).to(self.device)

        val_dataloader = (
            DatasetBuilder(train_x[val_indexs], train_y[val_indexs])
                .add_x(transform=toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
        )

        test_dataloader = (
            DatasetBuilder(test_x, test_y)
                .add_x(transform=toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
        )

        self.regist_databundler(train=DataBundler().cycle(sup_dataloader).add(unsup_dataloader).zip_mode(),
                                eval=val_dataloader,
                                test=test_dataloader)
        self.to(self.device)


class MixMatchDatasetMixin(Trainer):
    """semi-supervised image dataset for fixmatch"""

    def datasets(self, params: SemiSupervisedParams):
        dataset_fn = datasets[params.dataset]

        test_x, test_y = dataset_fn(False)
        train_x, train_y = dataset_fn(True)

        indexs, un_indexs, val_indexs = splits.semi_split(train_y, n_percls=params.n_percls, val_size=5000,
                                                          repeat_sup=False)

        mean, std = norm_val.get(params.dataset, [None, None])
        toTensor = ToNormTensor(mean, std)
        weak = Weak(mean, std)

        sup_set = (
            DatasetBuilder(train_x, train_y)
                .add_x(transform=weak)
                .add_y()
                .subset(indexs)
        )

        params.K = params.default(2, True)
        unsup_set = DatasetBuilder(train_x, train_y)
        for _ in range(params.K):
            unsup_set.add_x(transform=weak)
        unsup_set = unsup_set.add_y().subset(un_indexs)

        sup_dataloader = sup_set.DataLoader(batch_size=params.batch_size,
                                            num_workers=params.num_workers,
                                            shuffle=True)

        unsup_dataloader = unsup_set.DataLoader(batch_size=params.batch_size,
                                                num_workers=params.num_workers,
                                                shuffle=True)

        val_dataloader = (
            DatasetBuilder(train_x[val_indexs], train_y[val_indexs])
                .add_x(transform=toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
        )

        test_dataloader = (
            DatasetBuilder(test_x, test_y)
                .add_x(transform=toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
        )

        self.regist_databundler(train=DataBundler().cycle(sup_dataloader).add(unsup_dataloader).zip_mode(),
                                eval=val_dataloader,
                                test=test_dataloader)
        self.to(self.device)


class MnistImblanceDataset(Trainer):
    def datasets(self, params: MnistImblanceParams):
        super().datasets(params)
        from data.dataxy import mnist

        test_x, test_y = mnist(False)
        train_x, train_y = mnist(True)

        train_y = np.array(train_y, dtype=np.float32)
        test_y = np.array(test_y, dtype=np.float32)

        # search and mask sample with class [4, 9]
        train_mask_lis = [np.where(train_y == i)[0] for i in params.train_classes]
        test_mask_lis = [np.where(test_y == i)[0] for i in params.train_classes]

        for new_cls, i in enumerate(params.train_classes):
            train_y[train_mask_lis[new_cls]] = new_cls
            test_y[test_mask_lis[new_cls]] = new_cls

        train_mask = np.concatenate(train_mask_lis)
        test_mask = np.concatenate(test_mask_lis)

        test_x, test_y = test_x[test_mask], test_y[test_mask]
        train_x, train_y = train_x[train_mask], train_y[train_mask]

        # split train/val dataset
        train_ids, val_ids = splits.train_val_split(train_y, val_size=params.val_size)

        train_x, val_x = train_x[train_ids], train_x[val_ids]
        train_y, val_y = train_y[train_ids], train_y[val_ids]

        # reduce size of second class
        train_mask_lis = [np.where(train_y == i)[0] for i in range(len(params.train_classes))]
        sec_cls_size = int((1 - params.train_proportion) * len(train_mask_lis[0]))
        train_mask_lis[1] = train_mask_lis[1][:sec_cls_size]
        train_mask = np.concatenate(train_mask_lis)
        train_x, train_y = train_x[train_mask], train_y[train_mask]

        toTensor = ToNormTensor((0.1307,), (0.3081,))

        train_dataloader = (
            DatasetBuilder(train_x, train_y)
                .add_x(toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
        )

        val_dataloader = (
            DatasetBuilder(val_x, val_y)
                .add_x(transform=toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
        )

        test_dataloader = (
            DatasetBuilder(test_x, test_y)
                .add_x(transform=toTensor).add_y()
                .DataLoader(batch_size=params.batch_size, num_workers=params.num_workers, shuffle=True)
        )

        self.regist_databundler(
            train=DataBundler().add(train_dataloader).cycle(val_dataloader).zip_mode(),
            eval=val_dataloader,
            test=test_dataloader)
        self.to(self.device)
