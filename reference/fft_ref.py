import numpy as np


def fft_ref(x):
    return np.fft.fft(np.asarray(x, dtype=np.complex128), axis=-1)


def ifft_ref(x):
    return np.fft.ifft(np.asarray(x, dtype=np.complex128), axis=-1)


def rel_l2(y, y_ref):
    return np.linalg.norm(y - y_ref, axis=-1) / np.linalg.norm(y_ref, axis=-1)


def max_abs(y, y_ref):
    return np.abs(y - y_ref).max(axis=-1)
