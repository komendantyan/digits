from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

import sklearn.datasets
import sklearn.preprocessing
import sklearn.cross_validation

import tensorflow as tf

__all__ = []


def register(func):
    __all__.append(func.func_name)
    return func


@register
def imgshow(img, target=''):
    side_size = int(img.size ** 0.5)
    assert side_size ** 2 == img.size
    square_img = img.reshape((side_size, side_size))

    plt.imshow(square_img, cmap='gray_r', interpolation='none')

    ticks_args = (
        np.arange(side_size + 1) - 0.5,
        [j if j % 7 == 0 else '' for j in np.arange(29)]
    )
    plt.xticks(*ticks_args)
    plt.yticks(*ticks_args)

    plt.grid()

    label = '' if target == '' else str(int(target))
    plt.text(side_size * (25/28.0), side_size * (3/28.0), label, fontsize=35)


@register
def show_samples(X, y=None, n=9, random_state=None):
    ncolumns = int(np.ceil(n ** 0.5))
    nrows = int(np.ceil(n / ncolumns))

    if random_state is not None:
        np.random.seed(random_state)
    perm = np.random.choice(np.arange(X.shape[0]), n, replace=False)

    plt.figure(figsize=(16, int(np.ceil(16 / ncolumns * nrows)) ))

    for j, index in enumerate(perm, 1):
        plt.subplot(nrows, ncolumns, j)
        if y is None:
            label = ''
        elif len(y.shape) == 1:
            label = y[index]
        elif len(y.shape) == 2:
            label = np.argmax(y[index])
        else:
            raise ValueError
        imgshow(X[index], label)


@register
def load_datasets():
    Bunch = sklearn.datasets.base.Bunch
    mnist = Bunch()

    for dataset_name in ['original', 'dirty_light', 'dirty_hard']:
        arr = (np.load('datasets/mnist_%s.npy' % dataset_name)
               .astype(np.float32) / 255)
        mnist[dataset_name] = Bunch(
            train=arr[:60000],
            test=arr[60000:]
        )

    arr = np.load('datasets/mnist_labels.npy')
    mnist['labels'] = Bunch(
        train=arr[:60000],
        test=arr[60000:]
    )

    return mnist


_one_hot_encoder = sklearn.preprocessing.OneHotEncoder(10, sparse=False)


@register
def one_hot(y):
    return _one_hot_encoder.fit_transform(y.reshape(-1, 1))


@register
def get_batch_generator(approx_size, total_size, count=None):
    def counter():
        if count is None:
            j = 0
            while True:
                yield j
                j += 1
        else:
            for j in xrange(count):
                yield j

    iterator = iter([])

    for j in counter():
        try:
            _, indices = next(iterator)
        except StopIteration:
            iterator = iter(sklearn.cross_validation.KFold(
                total_size, total_size // approx_size, shuffle=True))
            _, indices = next(iterator)

        yield indices


@register
def plot_loss_map(y_true, y):
    if len(y.shape) == 1:
        pass
    elif len(y.shape) == 2:
        label = np.argmax(y, 1)
    else:
        raise ValueError

    map_ = np.zeros((10, 10), np.int32)
    for pair in zip(y_real, y_get):
        map_[pair] += 1

    for j in xrange(10):
        map_[j, j] = 0

    plt.pcolormesh(map_)
    plt.colorbar()
    plt.xlabel('real, expected')
    plt.ylabel('predicted, got')
    plt.title('Loss map')
    plt.xticks(np.arange(10)+0.5, np.arange(10))
    plt.yticks(np.arange(10)+0.5, np.arange(10))
