#!/usr/bin/env python3
# Error floor of fp16 storage alone: an exact (fp64) FFT whose output, and optionally input, is rounded to fp16.
# Same metric as the sweeps (rel L2 over 64 FFTs, inputs U(-1, 1) complex); NumPy's generator, not the CUDA hash.
#   python3 scripts/storage_floor.py
import numpy as np


def r16(z):
    return z.real.astype(np.float16).astype(float) + 1j * z.imag.astype(np.float16).astype(float)


def main():
    rng = np.random.default_rng(0)
    print("N      output rounded   input and output rounded")
    for n in (256, 512, 1024, 2048, 4096, 8192, 16384):
        x = rng.uniform(-1, 1, (64, n)) + 1j * rng.uniform(-1, 1, (64, n))
        y = np.fft.fft(x, axis=1)
        out = np.linalg.norm(r16(y) - y) / np.linalg.norm(y)
        both = np.linalg.norm(r16(np.fft.fft(r16(x), axis=1)) - y) / np.linalg.norm(y)
        print(f"{n:<6d} {out:.3e}        {both:.3e}")


if __name__ == "__main__":
    main()
