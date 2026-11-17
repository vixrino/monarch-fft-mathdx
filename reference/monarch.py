import numpy as np

# Four-step / Monarch FFT, N = n1 * n2, forward sign -1, no normalization.
#
# Index maps (derivation in docs/derivation.md):
#   input   n = n2*i1 + i2,  i1 < n1, i2 < n2   ->  a[i1, i2]  (row-major reshape to (n1, n2))
#   output  k = j1 + n1*j2,  j1 < n1, j2 < n2   <-  d[j1, j2]  (column-major read, i.e. d.T.reshape(N))
#
#   b = F_n1 @ a            n2 DFTs of size n1, one per column of a       b[j1, i2]
#   c = b * tw              tw[j1, i2] = w_N^(j1*i2)
#   d = c @ F_n2            n1 DFTs of size n2, one per row of c          d[j1, j2]
#   X = d.T.reshape(N)
#
# All functions act on the last axis; leading axes are a batch.


def dft_matrix(n):
    # reduce j*k mod n before exp so the angle stays in [0, 2pi)
    jk = np.outer(np.arange(n), np.arange(n)) % n
    return np.exp(-2j * np.pi * jk / n)


def twiddles(n1, n2):
    return np.exp(-2j * np.pi * np.outer(np.arange(n1), np.arange(n2)) / (n1 * n2))


def monarch_fft(x, n1, n2):
    x = np.asarray(x, dtype=np.complex128)
    a = x.reshape(*x.shape[:-1], n1, n2)
    b = dft_matrix(n1) @ a
    c = b * twiddles(n1, n2)
    # F_n2 is symmetric, so right-multiplying applies a DFT to each row
    d = c @ dft_matrix(n2)
    return np.swapaxes(d, -1, -2).reshape(x.shape)


def monarch_ifft(x, n1, n2):
    return np.conj(monarch_fft(np.conj(x), n1, n2)) / (n1 * n2)


def to_half(z):
    # fp16-representable values stored as complex64: products of two such values are exact in fp32
    out = np.empty(z.shape, dtype=np.complex64)
    out.real = z.real.astype(np.float16)
    out.imag = z.imag.astype(np.float16)
    return out


def monarch_fft_fp16_emu(x, n1, n2):
    # CPU emulation of fp16 GEMM operands with fp32 accumulation; see docs/derivation.md
    x = np.asarray(x)
    a = to_half(x.reshape(*x.shape[:-1], n1, n2))
    b = to_half(dft_matrix(n1)) @ a
    c = to_half(b * twiddles(n1, n2).astype(np.complex64))
    d = c @ to_half(dft_matrix(n2))
    return np.swapaxes(d, -1, -2).reshape(x.shape)
