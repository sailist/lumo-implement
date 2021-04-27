import numpy as np


def asymmetric_noisy(train_y, noisy_ratio, n_classes=None) -> np.ndarray:
    if n_classes is None:
        n_classes = len(set(train_y))
    noisy_map = {9: 1, 2: 0, 4: 7, 3: 5, 5: 3}

    noisy_ids = np.random.permutation(len(train_y))[:int(noisy_ratio * len(train_y))]
    noisy_ids = set(noisy_ids)
    noisy_y = np.array(train_y)

    cls_lis = []

    for i in range(n_classes):
        cls_lis.append(list(set(np.where(noisy_y == i)[0]) & noisy_ids))

    for i, idq in enumerate(cls_lis):
        if i in noisy_map:
            noisy_y[idq] = noisy_map[i]

    return noisy_y


def symmetric_noisy(train_y: np.ndarray, noisy_ratio: float, n_classes: int = None, force=False) -> np.ndarray:
    """

    :param train_y: raw clean labels
    :param noisy_ratio:
    :param n_classes:
    :param force:
    :return:
    """
    if n_classes is None:
        n_classes = len(set(train_y))

    noisy_ids = np.random.permutation(len(train_y))[:int(noisy_ratio * len(train_y))]
    noisy_y = np.array(train_y)

    _noisys = np.random.randint(0, n_classes, noisy_ids.shape[0])

    if force:
        _mask = np.where(_noisys == noisy_y[noisy_ids])
        while _mask[0].shape[0] != 0:
            _noisys[_mask] = np.random.randint(0, 10, _mask[0].shape[0])
            _mask = np.where(_noisys == noisy_y[noisy_ids])

    noisy_y[noisy_ids] = _noisys

    return noisy_y
