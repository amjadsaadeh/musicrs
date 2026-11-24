# Why musicrs is fast, and where it is only on par

musicrs and the [vectorized numpy baseline](../bench/bench_numpy.py) run the same algorithm,
single-threaded, on the same recordings. musicrs is clearly faster for narrowband DOA,
faster for IFB-MUSIC over many bins, and about level with numpy for IFB-MUSIC on a single
bin. This page lists the techniques behind that, how much each saves, and why the single-bin
case is still close.

Whole-call times are from the committed benchmark (`bench/results/`, measured on
[`6abc8bd`](https://github.com/amjadsaadeh/musicrs/commit/6abc8bd980f1e0b2cfc3cab00e38822d1d29a597),
see the README). The per-stage times come from a separate timing harness on the same
machine (Intel N150, 1 s recording, 4 channels, 16 kHz). The machine was busier during those
runs, so treat them as proportions, accurate to about ±30 %, not as exact costs.

## Where the time goes

### Narrowband DOA (1 s recording)

| Stage | numpy | musicrs |
|---|---:|---:|
| Input check, scaling to a power of two, splitting into channels | — | ~16 µs + ~130 µs |
| Analytic signal: complex FFT + inverse FFT per channel (numpy) / real FFT per channel (musicrs) | ~2.5 ms | ~0.32 ms |
| Covariance: 16 000-sample matrix product (numpy) / sum over 8 001 bins (musicrs) | ~0.36 ms | ~0.10 ms |
| Eigendecomposition, 4×4 | ~34 µs | ~3 µs |
| Spectrum, 720 angles | ~160 µs | ~110 µs |
| **Whole call (benchmark mean)** | **3.38 ms** | **0.97 ms** |

### IFB-MUSIC, one bin (1 s recording, `nfft` 160)

| Stage | numpy | musicrs |
|---|---:|---:|
| Input check, scaling, splitting into channels | — | ~16 µs + ~130 µs |
| STFT: 4 × 100 windows of 160 samples | ~250 µs | ~200–240 µs |
| Power of every bin (reference for the silence check) | — | ~26 µs |
| Cross-spectral matrix of the bin | ~10 µs | ~1 µs |
| Eigendecomposition + spectrum + log | ~216 µs | ~115 µs |
| **Whole call (benchmark mean)** | **563 µs** | **485 µs** |

## The techniques

### 1. Covariance from the real FFT, without an analytic signal (narrowband)

pyMUSIC and the numpy baseline build each channel's analytic signal (FFT, zero the
negative frequencies, inverse FFT) and then multiply a 4 × N complex matrix by its
conjugate transpose. `estimate_doa_from_real` skips the analytic signal entirely. By
Parseval's theorem,

    Σₜ zᵢ(t) zⱼ(t)* = (1/N) Σₖ |Hₖ|² Xᵢ(k) Xⱼ(k)*

where X is the channel's DFT and H the one-sided window (|H|² = 4 for positive
frequencies, 1 at DC and Nyquist, 0 for negative frequencies). Only bins 0..=N/2 of a
**real** FFT are needed (`analytic_covariance` in `src/music.rs`). The inverse FFT and the
N-sample complex matrix are gone, the forward FFT costs half as much, and the covariance
sums over N/2 bins instead of N complex samples. **About 2.9 ms of numpy work becomes about
0.4 ms**, and this accounts for almost all of the 3.5× narrowband lead.

### 2. Real-to-complex FFTs

The signals are real, so a complex FFT computes every value twice (the negative half of the
spectrum mirrors the positive half). musicrs uses `realfft` for the narrowband covariance and
for the STFT (`stft` in `src/stft.rs`). The STFT's four channels went from about 310 µs
with a complex FFT and a new buffer per window to about 200 µs.

### 3. Cached FFT plans

Planning a 16 000-point real FFT (twiddle factors, choosing an algorithm) takes about 215 µs,
two thirds of the cost of the four transforms themselves. `real_fft_plan` keeps the last four
plan lengths per thread, so repeated calls on recordings of the same length skip planning.

### 4. No allocation in hot loops

- `spectrum` writes each steering vector into one reused buffer (`steering_vector_into`)
  instead of allocating 720 vectors.
- The STFT reuses one input, output and scratch buffer for all windows.
- `cross_spectral_matrix` sums each entry over the windows directly. The old version
  built an nalgebra vector, its adjoint and a 4×4 product for every window: about **8 µs →
  1.3 µs per bin**, almost all of it allocation.

### 5. Reductions written so the compiler can vectorize them

The input check computes the largest magnitude and rejects NaN and ±inf. As a single running
`f64::max` fold, every step waits for the previous one, and the compiler may not reorder
floating-point operations to split the chain, so the loop runs at about 2.4 ns per sample.
`check_real_samples` keeps **8 independent lanes** instead, combining them at the end, and
detects non-finite values with `x * 0.0` (0 for finite x, NaN otherwise) summed per lane. The
pass over 64 000 samples went from **~157 µs to ~16 µs**.

### 6. Each pass over the data done once

The largest magnitude used to be computed twice, once to check the input and once more
(`data.amax()`) to scale it. The maximum is now reused, and scaling and splitting into
channels happen in one pass (`scaled_rows`). The scale is a power of two, so it is exact
and changes no result; no later rescaling is needed.

### 7. A Hermitian eigensolver on a tiny matrix

`faer`'s self-adjoint solver handles a 4×4 matrix in about 3 µs. `np.linalg.eig` needs about
34 µs, mostly call overhead plus a general (non-Hermitian) solver. This matters once per
bin. It also gives an orthonormal noise basis; see the golden-file notes in `CLAUDE.md`.

### 8. Shared with numpy: the cheaper spectrum formula

Both compute the denominator as ‖Eₙᴴ a‖² (one inner product per noise eigenvector) rather
than aᴴ Eₙ Eₙᴴ a, and both accumulate the IFB geometric mean as a sum of logarithms. Neither
of these accounts for any difference between the two.

## Why numpy is level for one bin

numpy is slow when Python does the looping and fast when a few large array operations do the
work. Single-bin IFB-MUSIC is the second kind:

- **numpy's heavy lifting is compiled code too.** `data.reshape(4, 100, 160)` is a free
  view, and `np.fft.rfft(..., axis=2)` transforms all 400 windows in one call into
  pocketfft (C). The spectrum is one vectorized `exp` over 2 880 elements and one matrix
  product. With one bin there are only about a dozen numpy calls, so Python overhead is
  small.
- **musicrs does work that numpy skips**: it validates the input (finite values), copies and
  scales it into contiguous channels (~130 µs, the largest remaining item, where numpy
  reshapes for free), and computes the power of every bin, because the silence check
  compares the requested bin with the strongest bin of the whole spectrum.
- **The spectrum's cost is mostly `sin`/`cos`.** `Complex::from_polar` for 720 angles ×
  4 microphones is about 80 µs of the ~110 µs, one scalar call at a time.

Narrowband wins because musicrs does far less arithmetic (technique 1). Many-bin IFB-MUSIC
wins because numpy pays Python overhead and a general eigensolver for every bin.
Single-bin IFB-MUSIC has neither advantage, so both end up close to the speed of the
underlying FFT and trigonometry.

## Caveat: the 80-bin benchmark does not compare equal work

The test recordings are exactly periodic: the 1400 Hz recording repeats every 80 samples, and
each channel holds only 80 distinct values. With `nfft` 160, only **20 of the 80 bins carry
energy; the other 60 are exactly zero**. musicrs skips bins whose power is negligible (their
noise subspace would be arbitrary; see `SILENCE_RATIO`). numpy and pyMUSIC process all 80. For a silent bin the cross-spectral matrix is zero, so the
eigensolver's noise subspace carries no information about the source, and their merged
spectra differ from musicrs's as well.

Measured back to back:

| | numpy | musicrs |
|---|---:|---:|
| bins 1..=80 | ~17.9 ms (80 processed) | ~3.3 ms (20 processed) |
| the 20 active bins only | ~4.7 ms | ~3.2 ms |

On the same 20 bins musicrs is about **1.5× faster** (about 130 µs vs 220 µs per processed
bin), not the 6.3× of the 80-bin row. Most of that 6.3× comes from doing a quarter of the
work. A fair comparison would use a recording with broadband noise, or have the numpy
baseline skip silent bins too.

## What is left

- **Read STFT windows straight from the column-major input.** Splitting into channels
  (`scaled_rows`, one `push` per sample) takes ~130 µs. Writing into zeroed rows with a
  strided iterator measured ~85 µs, and skipping the copy entirely for IFB would save all of
  it.
- **Cheaper steering vectors.** The phase is 2πf (pⱼ · d̂(θ)) / c; its sine and cosine could
  come from a per-antenna rotation recurrence over the angle grid, or from precomputed
  direction cosines, instead of 2 880 `sin`/`cos` calls per spectrum.
- **Only the requested bins.** A direct DFT of one bin over all windows takes ~55 µs, compared
  with ~200 µs for the full STFT. But the silence check needs the strongest bin of the whole
  spectrum, so this would change what counts as silent.
