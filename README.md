# musicrs

Rust implementation of MUltiple SIgnal Classification (MUSIC), described by R. O. Schmidt
([paper](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=1143830)), for
Direction-of-Arrival (DOA) estimation with microphone arrays.

It is a port of [pyMUSIC](https://github.com/amjadsaadeh/pyMUSIC), the implementation written
for empirical investigations of MUSIC in my master thesis. Pure Rust, no LAPACK.

## Features

- Narrowband MUSIC: covariance → noise subspace (Hermitian eigendecomposition via `faer`) →
  pseudo-spectrum `1 / ‖Eₙᴴ a(θ)‖²` → peak picking.
- Multi-source peak picking (circular local maxima, highest first). pyMUSIC's
  `getSourceAngles` clustered all estimates around the first peak; this is fixed here.
- IFB-MUSIC (wideband): per-bin cross-spectral matrices from an STFT, narrowband spectra
  combined by geometric mean.
- FFT-based analytic signal (Hilbert transform) and STFT helpers.

## Usage

```rust
use musicrs::MusicEstimator;
use nalgebra::DMatrix;

// Default: Kinect 4-mic linear array, 720 angle steps over [0, 2π), 16 kHz
let estimator = MusicEstimator::default();

// 4 × N real-valued microphone samples, e.g. read from a WAV file
let data: DMatrix<f64> = /* ... */;

// Narrowband DOA of one 1400 Hz source, in radians
let angles = estimator.estimate_doa_from_real(&data, 1, 1400.0)?;

// Wideband: IFB-MUSIC spectrum over chosen STFT bins (nfft = 160 → 100 Hz bins)
let (spectrum, angles) = estimator.ifb_music_spectrum(&data, 160, 1, Some(&[14]))?;
```

Bad input (wrong channel count, NaN, silence, too many sources, an unusable angle grid, …)
is returned as a `MusicError`.

For other arrays use `MusicEstimator::new(antenna_positions, angle_steps_num, search_interval,
sample_rate)` with `[x, y]` positions in metres. If you already have complex (analytic) data,
call `estimate_doa` directly.

### Things to keep in mind

- A linear array cannot distinguish θ from 360° − θ; restrict `search_interval` to [0, π]
  if you want a single answer.
- IFB-MUSIC only sees frequencies up to Nyquist (`sample_rate / 2`); a tone recorded above
  Nyquist is aliased and its DOA cannot be recovered from the STFT bins.
- Running IFB-MUSIC over *all* bins on a narrowband signal is dominated by noise-only bins;
  select the bins that carry signal energy.

## Performance vs pyMUSIC and numpy

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/benchmark-dark.svg">
  <img alt="Dot plot of mean time per call on a log scale, with p5 to p95 lines, for pyMUSIC, a vectorized numpy implementation of the same algorithm, and musicrs. pyMUSIC takes 146 to 164 ms per call, and 11.63 s for IFB-MUSIC over 80 bins. musicrs is 3.03 and 2.05 times faster than vectorized numpy for narrowband DOA (1.08 ms vs 3.28 ms, 2.66 ms vs 5.47 ms) and 3.94 times faster for IFB-MUSIC over 80 bins (4.75 ms vs 18.7 ms), but 2.12 and 2.33 times slower for single-bin IFB-MUSIC (1.17 ms vs 552 µs). It is 61 to 2,447 times faster than pyMUSIC." src="docs/benchmark.svg">
</picture>

The same calls on the same recordings (1400 Hz, 40°), from the real-valued samples to the
result, in three implementations:

- **pyMUSIC**, run unmodified;
- **numpy (vectorized)**, pyMUSIC's algorithm written the way numpy is meant to be used
  ([`bench/bench_numpy.py`](bench/bench_numpy.py)): the same analytic signal,
  `np.linalg.eig` and steering vectors, but in `float64`, with all 720 angles in one array
  operation, on one thread. Before timing, its spectra are checked against pyMUSIC's (relative
  difference below 1e-7, same peak);
- **musicrs**, single-threaded.

**Against pyMUSIC, musicrs is 61× to 2,447× faster, but that is not a Rust-vs-Python
result.** Almost all of pyMUSIC's time goes to a Python loop that builds the 720 steering
vectors one element at a time in software-emulated `float128`, with `np.matrix` products.
That loop costs ~145–165 ms per pseudo-spectrum whatever the recording length, and IFB-MUSIC
needs one spectrum per bin. Written idiomatically, the same algorithm in numpy takes
milliseconds.

**Against vectorized numpy, musicrs is 2.0× to 3.0× faster on narrowband DOA and 3.9× faster
on IFB-MUSIC over 80 bins, but 2.1× to 2.3× slower on single-bin IFB-MUSIC.**

- *Narrowband* never forms the analytic signals. Their covariance is computed directly from
  each channel's real FFT (Parseval's theorem), so there is no inverse FFT and no complex
  N-sample matrix, and the FFT plans are reused between calls. numpy does a complex FFT and an
  inverse FFT per channel and then a matrix product over all samples.
- *IFB-MUSIC over 80 bins* is dominated by the per-bin work of 80 eigendecompositions and 80
  spectra. numpy pays Python and call overhead for each bin.
- *Single-bin IFB-MUSIC* hasn't been optimised yet. musicrs computes the STFT and the power of
  every bin, then uses only one of them. It has not been profiled.

| Workload | Implementation | n | Mean ± stddev | Median | p5 – p95 | CV | musicrs speedup (mean) |
|---|---|---:|---:|---:|---:|---:|---:|
| Narrowband DOA · 1 s recording | pyMUSIC | 200 | 164 ms ± 12.1 ms | 160 ms | 153 ms – 191 ms | 7.4% | 152× |
|  | numpy (vectorized) | 1000 | 3.28 ms ± 370 µs | 3.2 ms | 2.88 ms – 4.06 ms | 11.3% | 3.03× |
|  | musicrs | 1000 | 1.08 ms ± 90 µs | 1.07 ms | 982 µs – 1.22 ms | 8.3% |  |
| Narrowband DOA · 2 s recording | pyMUSIC | 200 | 161 ms ± 8.79 ms | 158 ms | 154 ms – 175 ms | 5.5% | 61× |
|  | numpy (vectorized) | 1000 | 5.47 ms ± 639 µs | 5.33 ms | 4.88 ms – 6.45 ms | 11.7% | 2.05× |
|  | musicrs | 1000 | 2.66 ms ± 998 µs | 2.48 ms | 2.24 ms – 3.44 ms | 37.5% |  |
| IFB-MUSIC, 1 bin · 1 s recording | pyMUSIC | 200 | 146 ms ± 5.88 ms | 144 ms | 141 ms – 161 ms | 4.0% | 124× |
|  | numpy (vectorized) | 1000 | 552 µs ± 53 µs | 538 µs | 497 µs – 657 µs | 9.6% | 0.47× |
|  | musicrs | 1000 | 1.17 ms ± 111 µs | 1.16 ms | 1.05 ms – 1.32 ms | 9.5% |  |
| IFB-MUSIC, 1 bin · 2 s recording | pyMUSIC | 200 | 150 ms ± 5.8 ms | 148 ms | 145 ms – 163 ms | 3.9% | 70× |
|  | numpy (vectorized) | 1000 | 922 µs ± 166 µs | 869 µs | 792 µs – 1.23 ms | 18.0% | 0.43× |
|  | musicrs | 1000 | 2.15 ms ± 178 µs | 2.11 ms | 1.95 ms – 2.44 ms | 8.3% |  |
| IFB-MUSIC, 80 bins · 1 s recording | pyMUSIC | 30 | 11.63 s ± 216 ms | 11.55 s | 11.37 s – 11.92 s | 1.9% | 2,447× |
|  | numpy (vectorized) | 1000 | 18.7 ms ± 1.35 ms | 18.4 ms | 17.4 ms – 21.5 ms | 7.2% | 3.94× |
|  | musicrs | 1000 | 4.75 ms ± 1.29 ms | 4.45 ms | 4.15 ms – 5.64 ms | 27.2% |  |

*musicrs speedup*: the row's mean time over musicrs's; below 1× musicrs is slower. Every call
is timed separately (pyMUSIC: 200 calls per case, 30 for 80 bins; numpy and musicrs: 1000)
after 3 warm-up calls, the three implementations one after the other on an otherwise idle
machine. Machine: Intel N150 (3 cores), Linux, `performance` governor, rustc 1.88, Python
3.10 / numpy 1.23.5 (OpenBLAS, one thread); see [`bench/`](bench/README.md) for the harnesses
and how to reproduce, and [`bench/results/`](bench/results/) for the raw samples.

### musicrs over time

The same benchmark at three points in musicrs's history, each a separate run on the same
machine. Times are mean (median) per call. Separate runs differ by up to ~20 % (see
*Run to run* below), so only changes well beyond that mean anything.

| Workload | [`b2756f8`](https://github.com/amjadsaadeh/musicrs/commit/b2756f8278c7bf36e28386c87594b274ca8d2b51) | [`4b52605`](https://github.com/amjadsaadeh/musicrs/commit/4b52605f9151899167d8bc147e4d004e8556bfbf) | [`db81fd9`](https://github.com/amjadsaadeh/musicrs/commit/db81fd940f6d08090f0fb057e02a85d528b94247) | numpy (vectorized), latest run |
|---|---:|---:|---:|---:|
| Narrowband DOA · 1 s | 5.63 ms (5.41 ms) | 6.29 ms (6.04 ms) | **1.08 ms (1.07 ms)** | 3.28 ms |
| Narrowband DOA · 2 s | 11.7 ms (11.5 ms) | 13.4 ms (13.0 ms) | **2.66 ms (2.48 ms)** | 5.47 ms |
| IFB-MUSIC, 1 bin · 1 s | 757 µs (770 µs) | 1.12 ms (1.10 ms) | 1.17 ms (1.16 ms) | 552 µs |
| IFB-MUSIC, 1 bin · 2 s | 1.24 ms (1.22 ms) | 2.48 ms (2.50 ms) | 2.15 ms (2.11 ms) | 922 µs |
| IFB-MUSIC, 80 bins · 1 s | 14.7 ms (14.5 ms) | 5.43 ms (5.46 ms) | 4.75 ms (4.45 ms) | 18.7 ms |

- [`b2756f8`](https://github.com/amjadsaadeh/musicrs/commit/b2756f8278c7bf36e28386c87594b274ca8d2b51): the first benchmark. Its results were committed with that commit;
  which code they were measured on was not recorded.
- [`4b52605`](https://github.com/amjadsaadeh/musicrs/commit/4b52605f9151899167d8bc147e4d004e8556bfbf): `main` after #23–#29. Among other changes, IFB-MUSIC builds its
  cross-spectral matrices one bin at a time (#23) and the channels share one FFT planner (#26).
  Over 80 bins IFB-MUSIC became ~3× faster, while single-bin IFB-MUSIC became 1.5–2× slower.
  That slowdown has not been profiled.
- [`db81fd9`](https://github.com/amjadsaadeh/musicrs/commit/db81fd940f6d08090f0fb057e02a85d528b94247): narrowband DOA computes the covariance from real FFTs (#34), and
  `spectrum` no longer allocates per angle. Narrowband DOA is 5–6× faster than at
  [`4b52605`](https://github.com/amjadsaadeh/musicrs/commit/4b52605f9151899167d8bc147e4d004e8556bfbf).

The raw samples of each run are in the repository, e.g.
`git show d53ba39:bench/results/musicrs.csv` (run on [`4b52605`](https://github.com/amjadsaadeh/musicrs/commit/4b52605f9151899167d8bc147e4d004e8556bfbf)) and
`git show a8d2fe3:bench/results/musicrs.csv` (run on [`db81fd9`](https://github.com/amjadsaadeh/musicrs/commit/db81fd940f6d08090f0fb057e02a85d528b94247)); `bench/results/environment.txt`
in the same commit records the machine and the musicrs commit.

### Distribution of the timings

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/benchmark-distribution-dark.svg">
  <img alt="Histograms of time per call divided by the median, one per workload and implementation. All are right-skewed: a sharp peak near the fastest calls and a tail of slower calls. musicrs narrowband DOA on the 2 s recording and musicrs IFB-MUSIC over 80 bins each have a small group of calls at twice the median or more." src="docs/benchmark-distribution.svg">
</picture>

- **Not normal.** Shapiro-Wilk rejects normality for every case, on t and on log t (p < 0.001;
  p ≈ 0.03 for pyMUSIC over 80 bins, which has only 30 calls). Every distribution is
  right-skewed (skew 0.5 to 4.8, and 10 to 16 for the two musicrs cases below): a sharp edge
  near the fastest call and a tail of slower ones, the shape of interference that can only add
  time. A log-normal fits better than a normal in every case, but none is log-normal either.
- **Scheduling moves the mean by a few percent.** The mean is at most 7.3 % above the median,
  and the outliers (calls slower than median + 3 robust sigmas) raise the mean by at most
  6.0 % (*tail excess*). Both maxima come from musicrs on narrowband DOA (2 s recording) and
  IFB-MUSIC over 80 bins. Each has a small group of calls at twice the median or more (the
  last histogram bin), which no other case has. For these two, the median is the better single
  number. Mean and median give the same ratios to within 7 %. Because the distributions are
  skewed, ± stddev overstates the spread below the mean; the p5 – p95 column shows the real
  range.
- **Slow calls come in runs.** Lag-1 autocorrelation is 0.2 to 0.7, so neighbouring calls are
  alike. Isolated preemptions would hit calls independently; runs point to slower changes in
  machine state (CPU frequency, caches, the core the process runs on).
- **Run to run**, numpy's means moved by up to 9 % and pyMUSIC's by up to 3 % against a run
  the day before, and a shorter musicrs run an hour earlier gave means up to 9 % lower. Read the numbers
  as about ± 20 %. The ratios between musicrs and numpy (2× to 4×) are larger than that, and
  the ratios to pyMUSIC far larger.

<details>
<summary>Shape statistics per case</summary>

| Workload | Implementation | Shape | Skew | Excess kurtosis | Shapiro p (t / log t) | (Mean − median) / median | Outliers | Tail excess | Lag-1 autocorr. | Drift |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Narrowband DOA · 1 s recording | pyMUSIC | right-skewed, heavy tail, not normal (closer to log-normal) | 2.15 | 5.8 | 2.7e-16 / 5.7e-15 | +2.6% | 17.0% | +2.8% | 0.68 | -0.6% |
|  | numpy (vectorized) | right-skewed, heavy tail, not normal (closer to log-normal) | 2.25 | 8.5 | 7e-32 / 1.9e-26 | +2.5% | 6.7% | +2.3% | 0.66 | -13.5% |
|  | musicrs | right-skewed, heavy tail, not normal (closer to log-normal) | 3.85 | 38.6 | 7.2e-34 / 4.9e-26 | +1.4% | 1.4% | +0.5% | 0.51 | -2.8% |
| Narrowband DOA · 2 s recording | pyMUSIC | right-skewed, heavy tail, not normal (closer to log-normal) | 2.65 | 9.2 | 1.7e-17 / 3.1e-16 | +1.8% | 7.5% | +1.2% | 0.35 | -1.6% |
|  | numpy (vectorized) | right-skewed, heavy tail, not normal (closer to log-normal) | 4.31 | 37.4 | 4.1e-38 / 1.2e-30 | +2.7% | 4.8% | +1.8% | 0.73 | +5.6% |
|  | musicrs | right-skewed, heavy tail, not normal (closer to log-normal) | 15.97 | 352.9 | 2.2e-52 / 1.5e-41 | +7.3% | 7.5% | +6.0% | 0.24 | -5.3% |
| IFB-MUSIC, 1 bin · 1 s recording | pyMUSIC | right-skewed, not normal (closer to log-normal) | 1.61 | 1.8 | 8.8e-16 / 2.7e-15 | +1.6% | 13.5% | +1.4% | 0.38 | +0.7% |
|  | numpy (vectorized) | right-skewed, not normal (closer to log-normal) | 1.46 | 2.8 | 1.6e-26 / 1e-22 | +2.7% | 3.6% | +1.1% | 0.66 | +3.7% |
|  | musicrs | right-skewed, heavy tail, not normal (closer to log-normal) | 3.82 | 29.8 | 3.3e-36 / 1.1e-28 | +0.9% | 0.9% | +0.6% | 0.40 | +6.4% |
| IFB-MUSIC, 1 bin · 2 s recording | pyMUSIC | right-skewed, not normal (closer to log-normal) | 1.82 | 2.9 | 4.8e-16 / 1.7e-15 | +1.3% | 13.0% | +1.3% | 0.47 | +2.4% |
|  | numpy (vectorized) | right-skewed, heavy tail, not normal (closer to log-normal) | 4.78 | 51.2 | 7.2e-40 / 1.1e-31 | +6.1% | 10.8% | +4.9% | 0.69 | +1.8% |
|  | musicrs | right-skewed, heavy tail, not normal (closer to log-normal) | 2.63 | 14.9 | 7.6e-32 / 1.7e-26 | +1.8% | 2.5% | +0.8% | 0.38 | +1.1% |
| IFB-MUSIC, 80 bins · 1 s recording | pyMUSIC | right-skewed, not normal (closer to log-normal) | 0.53 | -0.7 | 0.026 / 0.029 | +0.7% | 0.0% | +0.0% | 0.26 | -0.6% |
|  | numpy (vectorized) | right-skewed, heavy tail, not normal (closer to log-normal) | 2.16 | 7.2 | 5e-32 / 8.4e-29 | +1.8% | 7.2% | +1.5% | 0.62 | +3.7% |
|  | musicrs | right-skewed, heavy tail, not normal (closer to log-normal) | 10.40 | 167.8 | 2.4e-50 / 6.9e-43 | +6.8% | 10.1% | +5.6% | 0.49 | -0.1% |

*Outliers*: calls slower than median + 3 · 1.4826 · MAD. *Tail excess*: mean of all calls
over the mean without the outliers. *Drift*: mean of the last 10 % of calls over the first 10 %.

</details>

## Differences from pyMUSIC

| | pyMUSIC | musicrs |
|---|---|---|
| Eigendecomposition | `np.linalg.eig` (general) | Hermitian solver (`faer`), orthonormal noise basis |
| Sign convention | conjugated analytic signal, `exp(−j2πfτ)` | non-conjugated analytic signal, `exp(+j2πfτ)` (equivalent) |
| Multi-source peaks | broken (see TODO in `getSourceAngles`) | circular local maxima |
| IFB default bins | all, including DC | all except DC |
| Precision | `float128` steering vectors | `f64` |

On nearly noiseless data the noise eigenvalues are degenerate (~1e-15) and `np.linalg.eig`
returns a non-orthogonal noise basis, so pyMUSIC's spectrum values depend on that arbitrary
basis (≈9 % on the test data); its peak locations are unaffected.

## Development

```sh
cargo test                                  # unit, doc and integration tests
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

Test data in `tests/data/single-source/` are 4-channel, 24-bit, 16 kHz WAV recordings named
`single-source_<freq>Hz_<doa>deg_<duration>s.wav`. `tests/golden_test.rs` checks parity with
spectra produced by the original pyMUSIC code; see `tests/golden/generate_golden.py` to
regenerate them.

`examples/bench_pymusic.rs` and `bench/` hold the runtime benchmark against pyMUSIC; see
[`bench/README.md`](bench/README.md).

## License

MIT
