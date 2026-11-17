# Four-step / Monarch FFT: derivation

Notation matches `reference/monarch.py`. N = n1·n2, w_M = exp(-2πi/M).

## Convention

Forward DFT, no normalization:

    X[k] = Σ_{n=0}^{N-1} x[n] · w_N^(n·k)

Inverse, normalized once by 1/N:

    x[n] = (1/N) Σ_k X[k] · w_N^(-n·k)

This is `np.fft.fft` / `np.fft.ifft` with the default `norm="backward"`.

## Index maps

Split the input and output indices with *different* strides:

    n = n2·i1 + i2      i1 ∈ [0, n1),  i2 ∈ [0, n2)
    k = j1 + n1·j2      j1 ∈ [0, n1),  j2 ∈ [0, n2)

Both maps are bijections onto [0, N). Expanding the exponent:

    n·k = (n2·i1 + i2)(j1 + n1·j2)
        = n2·i1·j1  +  N·i1·j2  +  i2·j1  +  n1·i2·j2

and using w_N^n2 = w_n1, w_N^N = 1, w_N^n1 = w_n2:

    w_N^(n·k) = w_n1^(i1·j1) · w_N^(j1·i2) · w_n2^(i2·j2)

So

    X[j1 + n1·j2] = Σ_{i2} w_n2^(i2·j2) · [ w_N^(j1·i2) · Σ_{i1} w_n1^(i1·j1) · x[n2·i1 + i2] ]
                    \_______ step 3 ______/ \_ step 2 _/   \___________ step 1 ___________/

The term N·i1·j2 vanishing is the whole point, and it is why the two index
maps use different strides (n2 on i1 for the input, n1 on j2 for the output):
those strides are exactly what turn two of the four terms into w_n1 and w_n2
powers and the third into a multiple of N.

## As matrix products

Let `a` be the n1×n2 matrix

    a[i1, i2] = x[n2·i1 + i2]                  a = x.reshape(n1, n2)   (row-major)

Then, with F_m[j, i] = w_m^(j·i) the dense m×m DFT matrix (symmetric):

| step | operation                         | shape                      | meaning                               |
|------|-----------------------------------|----------------------------|---------------------------------------|
| 1    | b = F_n1 · a                      | (n1×n1)(n1×n2) → n1×n2     | n2 DFTs of size n1, one per column    |
| 2    | c = b ∘ T,  T[j1,i2] = w_N^(j1·i2) | n1×n2, elementwise          | twiddles                              |
| 3    | d = c · F_n2                      | (n1×n2)(n2×n2) → n1×n2     | n1 DFTs of size n2, one per row        |
| 4    | X[j1 + n1·j2] = d[j1, j2]         | X = dᵀ.reshape(N)          | transpose, then row-major flatten     |

Step 3 uses `c · F_n2` rather than `(F_n2 · cᵀ)ᵀ`; the two are equal because
F_n2 = F_n2ᵀ. Step 4 is a column-major read of `d`; writing it as an explicit
transpose keeps the reshape convention row-major everywhere.

For a batch, every step acts on the last two axes; the code uses `@` and
`np.swapaxes(d, -1, -2)` for that.

## Kronecker / Monarch form

With vec(·) stacking columns and vec(A·X·B) = (Bᵀ ⊗ A)·vec(X):

- vec(a) = P·x, where P is the stride permutation (P x)[i1 + n1·i2] = x[n2·i1 + i2]
  (the transpose of a row-major n1×n2 array),
- step 1: vec(F_n1·a) = (I_n2 ⊗ F_n1)·vec(a),
- step 2: vec(b ∘ T) = Ω·vec(b), Ω = diag(vec(T)),
- step 3: vec(c·F_n2) = (F_n2ᵀ ⊗ I_n1)·vec(c) = (F_n2 ⊗ I_n1)·vec(c),
- step 4: X = vec(d), since vec(d)[j1 + n1·j2] = d[j1, j2].

Hence

    F_N = (F_n2 ⊗ I_n1) · Ω · (I_n2 ⊗ F_n1) · P

(I_n2 ⊗ F_n1) is block-diagonal with n2 copies of F_n1. (F_n2 ⊗ I_n1) is
block-diagonal up to a perfect-shuffle permutation Q: (F_n2 ⊗ I_n1) =
Q·(I_n1 ⊗ F_n2)·Qᵀ. So F_N is a product of two block-diagonal matrices, each
wrapped in permutations, with the diagonal Ω folded into one factor. That is
the Monarch structure of Dao et al. (2022), used for FFTs on tensor cores in
FlashFFTConv (Fu et al., 2024). On a GPU each block-diagonal factor is a
batched GEMM against a dense F_m, which is what makes tensor cores usable.

## Cost

A radix FFT does O(N log₂ N) complex flops. The two dense stages do
n1·(n1·n2) + n2·(n1·n2) = N·(n1 + n2) complex multiply-adds, plus N twiddle
multiplies. For n1 = n2 = √N this is 2·N·√N, i.e. (2√N)/log₂N times more
arithmetic than a radix FFT: 8/3 ≈ 2.7× at N = 64, 128/12 ≈ 10.7× at N = 4096.
A complex GEMM on real tensor cores also costs 4 real GEMMs (3 with Gauss's
trick). The Monarch path only wins if the tensor-core throughput advantage
over the FFT's FMA-pipe arithmetic, plus the GEMM's more regular data
movement, outweighs that factor. This is the crossover the project measures;
the ratio above is arithmetic, not a measurement.

## Inverse

`monarch_ifft(x) = conj(monarch_fft(conj(x))) / N`. This follows from
conj(F_N) = F_N⁻¹·N. The 1/N is applied once at the end, never split per
stage, so the forward and inverse share every matrix and twiddle.

## Twiddle and DFT-matrix precision

- F_m entries use (j·i mod m) before the exp, so the angle is always in
  [0, 2π). Without the reduction, j·i reaches (m−1)², and in fp32 the argument
  itself loses bits before the sin/cos are evaluated.
- Twiddle exponents j1·i2 are < N already, so no reduction is needed.
- Every entry is computed directly in fp64 and rounded once to the target
  precision. No recurrences (w^(k+1) = w^k·w), which accumulate error along k.

## fp16 emulation (CPU only)

`monarch_fft_fp16_emu` models what an fp16 tensor-core GEMM with fp32
accumulation does, on the CPU, in NumPy. Rounding points:

1. x → fp16 (input as stored for GEMM 1),
2. F_n1, F_n2 → fp16 (GEMM operands),
3. GEMM 1 product and sum in fp32 (`complex64` matmul; a product of two fp16
   values is exact in fp32, so only the accumulation rounds),
4. twiddle multiply in fp32 with fp32 twiddles,
5. c → fp16 (operand of GEMM 2),
6. GEMM 2 in fp32, output kept in fp32.

This is not bit-exact with the GPU: summation order differs, complex
products go through `cgemm` rather than 4 real GEMMs, and the tensor-core
accumulator's internal rounding is not modelled. It is a check on the
algorithm's error budget, not a GPU result. Measured numbers on T4 replace
it in a later phase.

## Three easy mistakes and what catches them

For each, I injected one concrete instance of the bug into
`reference/monarch.py`, ran pytest and restored the file. What was injected
and how many tests failed is in docs/decisions.md.

1. **Index maps / final transpose.** Reading the output row-major
   (`d.reshape(N)` instead of `dᵀ.reshape(N)`), or using stride n1 on the
   input. The result is a permutation of X, so norms look fine.
   Caught by `test_forward` on every factorization and by
   `test_unit_impulse`, which checks X[k] = w_N^(m·k) entry by entry.

2. **Twiddle layout or root.** Using w_n2 instead of w_N fails everywhere.
   The nasty one is a layout bug: a twiddle table written row-major as
   n1×n2 but read with stride n1. For n1 = n2 the table is symmetric and the
   bug is invisible. Caught only because the tests cover non-square
   factorizations (n1 ≠ n2); square-only tests pass with this bug.

3. **Inverse normalization and sign.** Scaling by 1/n1 or 1/n2 instead of
   1/N, or forgetting one of the two conjugations. Caught by `test_inverse`,
   which compares against `np.fft.ifft` and also checks the round trip
   ifft(fft(x)) = x.
