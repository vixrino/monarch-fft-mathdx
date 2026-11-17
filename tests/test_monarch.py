import numpy as np
import pytest

from reference.fft_ref import fft_ref, ifft_ref, max_abs, rel_l2
from reference.monarch import monarch_fft, monarch_fft_fp16_emu, monarch_ifft

# every power-of-two N in [16, 16384] and every N = n1*n2 with n1, n2 >= 4
CASES = [(1 << m, 1 << a, 1 << (m - a)) for m in range(4, 15) for a in range(2, m - 1)]
IDS = [f"N{n}-{n1}x{n2}" for n, n1, n2 in CASES]
SEEDS = 5


def rand_c(seed, shape):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, shape) + 1j * rng.uniform(-1, 1, shape)


@pytest.mark.parametrize("n,n1,n2", CASES, ids=IDS)
def test_forward(n, n1, n2):
    x = rand_c(n + n1, (SEEDS, n))
    assert rel_l2(monarch_fft(x, n1, n2), fft_ref(x)).max() < 1e-13


@pytest.mark.parametrize("n,n1,n2", CASES, ids=IDS)
def test_inverse(n, n1, n2):
    x = rand_c(n + n1, (SEEDS, n))
    assert rel_l2(monarch_ifft(x, n1, n2), ifft_ref(x)).max() < 1e-13
    assert rel_l2(monarch_ifft(monarch_fft(x, n1, n2), n1, n2), x).max() < 1e-13


def test_unit_impulse():
    # x = e_m gives X[k] = w_N^(m*k): checks the index maps and sign independently of random data
    n, n1, n2, m = 512, 8, 64, 37
    x = np.zeros(n, dtype=np.complex128)
    x[m] = 1
    k = np.arange(n)
    assert np.allclose(monarch_fft(x, n1, n2), np.exp(-2j * np.pi * m * k / n), atol=1e-13)


def test_fp16_emulation_report():
    print("\nCPU EMULATION (NumPy), NOT A GPU RESULT: fp16-rounded operands, fp32 accumulation.")
    print(f"median over {SEEDS} inputs ~ U(-1,1) complex, vs FP64 np.fft")
    print(f"{'N':>6} {'n1':>5} {'n2':>5} {'rel_l2':>10} {'max_abs':>10}")
    for n, n1, n2 in CASES:
        x = rand_c(n + n1, (SEEDS, n))
        y, y_ref = monarch_fft_fp16_emu(x, n1, n2), fft_ref(x)
        e, m = np.median(rel_l2(y, y_ref)), np.median(max_abs(y, y_ref))
        print(f"{n:>6} {n1:>5} {n2:>5} {e:>10.3e} {m:>10.3e}")
        # loose sanity bound, a few hundred fp16 ulps; the table is the output, not this bound
        assert e < 1e-2
