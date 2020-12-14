import numpy as np
from numpy.lib.stride_tricks import as_strided


def seq2chunk(data: np.ndarray, win_size: int, overlap_size: int = 0,
              overhang_option: bool = False):
    """
    """
    assert data.ndim == 1 or data.ndim == 2, "data should be 1-D or 2-D."
    assert win_size > overlap_size, \
        "'win_size' sholud be greater than overlap_size"
    assert isinstance(overhang_option, bool), \
        "'overhang_option' should be boolean type."

    if data.ndim == 1:
        data = data.reshape((-1, 1))

    # get the number of overlapping windows that fit into the data
    n_win = (data.shape[0]-win_size)//(win_size-overlap_size) + 1
    overhang = data.shape[0] - (n_win*win_size - (n_win-1)*overlap_size)

    # if there's overhang, need an extra window and a zero pad on the data
    if overhang != 0:
        n_win += 1
        newdata = np.zeros((
            n_win*win_size - (n_win-1)*overlap_size,
            data.shape[1]))
        newdata[:data.shape[0]] = data
        data = newdata
    else:
        overhang_option = True

    sz = data.dtype.itemsize
    ret = as_strided(x=data,
                     shape=(n_win, win_size*data.shape[1]),
                     strides=((win_size-overlap_size)*data.shape[1]*sz, sz))
    if overhang_option:
        return ret.reshape((n_win, -1, data.shape[1]))
    else:
        return ret.reshape((n_win, -1, data.shape[1]))[:-1]
