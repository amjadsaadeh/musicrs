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
  <img alt="Dot plot of mean time per call on a log scale, with p5 to p95 lines, for pyMUSIC, a vectorized numpy implementation of the same algorithm, and musicrs. pyMUSIC takes 152 to 165 ms per call, and 11.75 s for IFB-MUSIC over 80 bins. musicrs is 3.50 and 2.55 times faster than vectorized numpy for narrowband DOA (966 µs vs 3.38 ms, 2.08 ms vs 5.29 ms) and 6.30 times faster for IFB-MUSIC over 80 bins (3.01 ms vs 19 ms). For single-bin IFB-MUSIC it is 1.16 times faster on the 1 s recording (485 µs vs 563 µs) and as fast on the 2 s recording (889 µs vs 891 µs). It is 79 to 3,904 times faster than pyMUSIC." src="docs/benchmark.svg">
</picture>

The same calls on the same recordings (1400 Hz, 40°), from the real-valued samples to the
result, in three implementations:

- **pyMUSIC**, run unmodified;
- **numpy (vectorized)**, pyMUSIC's algorithm written the way numpy is meant to be used
  ([`bench/bench_numpy.py`](bench/bench_numpy.py)): the same analytic signal,
  `np.linalg.eig` and steering vectors, but in `float64`, with all 720 angles in one array
  operation, on one thread. Before timing, its spectra are checked against pyMUSIC's (relative
  difference below 1e-7, same peak), and for narrowband DOA also the covariance, built with
  pySoundToolbox's analytic signal as pyMUSIC does;
- **musicrs**, single-threaded.

**Against pyMUSIC, musicrs is 79× to 3,904× faster, but that is not a Rust-vs-Python
result.** Almost all of pyMUSIC's time goes to a Python loop that builds the 720 steering
vectors one element at a time in software-emulated `float128`, with `np.matrix` products.
That loop costs ~150–165 ms per pseudo-spectrum whatever the recording length, and IFB-MUSIC
needs one spectrum per bin. Written idiomatically, the same algorithm in numpy takes
milliseconds.

**Against vectorized numpy, musicrs is 2.6× to 3.5× faster on narrowband DOA and 6.3× faster
on IFB-MUSIC over 80 bins, and level on single-bin IFB-MUSIC (1.16× on the 1 s recording,
1.00× on the 2 s one).**

- *Narrowband* never forms the analytic signals. Their covariance is computed directly from
  each channel's real FFT (Parseval's theorem), so there is no inverse FFT and no complex
  N-sample matrix, and the FFT plans are reused between calls. numpy does a complex FFT and an
  inverse FFT per channel and then a matrix product over all samples.
- *IFB-MUSIC over 80 bins* is dominated by the per-bin work of 80 eigendecompositions and 80
  spectra. numpy pays Python and call overhead for each bin.
- *Single-bin IFB-MUSIC* is mostly preprocessing, not MUSIC: checking and scaling the samples
  and the STFT of every bin. The silence check compares the requested bin with the strongest
  bin of the whole spectrum, so all bins are computed. With one vectorizable input pass and a
  real FFT (#35), musicrs does this about as fast as numpy's single batched `rfft`.

| Workload | Implementation | n | Mean ± stddev | Median | p5 – p95 | CV | musicrs speedup (mean) |
|---|---|---:|---:|---:|---:|---:|---:|
| Narrowband DOA · 1 s recording | pyMUSIC | 200 | 162 ms ± 7.49 ms | 161 ms | 153 ms – 175 ms | 4.6% | 168× |
|  | numpy (vectorized) | 1000 | 3.38 ms ± 338 µs | 3.32 ms | 2.94 ms – 4.07 ms | 10.0% | 3.50× |
|  | musicrs | 1000 | 966 µs ± 91 µs | 938 µs | 869 µs – 1.15 ms | 9.4% |  |
| Narrowband DOA · 2 s recording | pyMUSIC | 200 | 165 ms ± 7.96 ms | 163 ms | 155 ms – 178 ms | 4.8% | 79× |
|  | numpy (vectorized) | 1000 | 5.29 ms ± 529 µs | 5.18 ms | 4.57 ms – 6.26 ms | 10.0% | 2.55× |
|  | musicrs | 1000 | 2.08 ms ± 178 µs | 2.04 ms | 1.86 ms – 2.42 ms | 8.6% |  |
| IFB-MUSIC, 1 bin · 1 s recording | pyMUSIC | 200 | 153 ms ± 10.4 ms | 150 ms | 142 ms – 176 ms | 6.8% | 316× |
|  | numpy (vectorized) | 1000 | 563 µs ± 50 µs | 561 µs | 495 µs – 649 µs | 8.9% | 1.16× |
|  | musicrs | 1000 | 485 µs ± 34 µs | 475 µs | 455 µs – 554 µs | 7.0% |  |
| IFB-MUSIC, 1 bin · 2 s recording | pyMUSIC | 200 | 152 ms ± 10.8 ms | 147 ms | 145 ms – 170 ms | 7.1% | 171× |
|  | numpy (vectorized) | 1000 | 891 µs ± 113 µs | 859 µs | 767 µs – 1.15 ms | 12.7% | 1.00× |
|  | musicrs | 1000 | 889 µs ± 74 µs | 868 µs | 812 µs – 1.04 ms | 8.3% |  |
| IFB-MUSIC, 80 bins · 1 s recording | pyMUSIC | 30 | 11.75 s ± 281 ms | 11.70 s | 11.39 s – 12.19 s | 2.4% | 3,904× |
|  | numpy (vectorized) | 1000 | 19 ms ± 1.28 ms | 18.8 ms | 17.3 ms – 21.6 ms | 6.8% | 6.30× |
|  | musicrs | 1000 | 3.01 ms ± 186 µs | 2.96 ms | 2.81 ms – 3.41 ms | 6.2% |  |

*musicrs speedup*: the row's mean time over musicrs's; below 1× musicrs is slower. Every call
is timed separately (pyMUSIC: 200 calls per case, 30 for 80 bins; numpy and musicrs: 1000)
after 3 warm-up calls, the three implementations one after the other on an otherwise idle
machine. Machine: Intel N150 (3 cores), Linux, `performance` governor, rustc 1.100, Python
3.10 / numpy 1.23.5 (OpenBLAS, one thread); see [`bench/`](bench/README.md) for the harnesses
and how to reproduce, and [`bench/results/`](bench/results/) for the raw samples.

### musicrs over time

The same benchmark at four points in musicrs's history, each a separate run on the same
machine. Times are mean (median) per call. Separate runs differ by up to ~20 % (see
*Run to run* below), so only changes well beyond that mean anything.

| Workload | [`b2756f8`](https://github.com/amjadsaadeh/musicrs/commit/b2756f8278c7bf36e28386c87594b274ca8d2b51) | [`4b52605`](https://github.com/amjadsaadeh/musicrs/commit/4b52605f9151899167d8bc147e4d004e8556bfbf) | [`3e2b808`](https://github.com/amjadsaadeh/musicrs/commit/3e2b8081e300fc4e8b8a06d40613dc43ad410385) | [`6abc8bd`](https://github.com/amjadsaadeh/musicrs/commit/6abc8bd980f1e0b2cfc3cab00e38822d1d29a597) | numpy (vectorized), latest run |
|---|---:|---:|---:|---:|---:|
| Narrowband DOA · 1 s | 5.63 ms (5.41 ms) | 6.29 ms (6.04 ms) | **1.08 ms (1.07 ms)** | 966 µs (938 µs) | 3.38 ms |
| Narrowband DOA · 2 s | 11.7 ms (11.5 ms) | 13.4 ms (13.0 ms) | **2.66 ms (2.48 ms)** | 2.08 ms (2.04 ms) | 5.29 ms |
| IFB-MUSIC, 1 bin · 1 s | 757 µs (770 µs) | 1.12 ms (1.10 ms) | 1.17 ms (1.16 ms) | **485 µs (475 µs)** | 563 µs |
| IFB-MUSIC, 1 bin · 2 s | 1.24 ms (1.22 ms) | 2.48 ms (2.50 ms) | 2.15 ms (2.11 ms) | **889 µs (868 µs)** | 891 µs |
| IFB-MUSIC, 80 bins · 1 s | 14.7 ms (14.5 ms) | 5.43 ms (5.46 ms) | 4.75 ms (4.45 ms) | **3.01 ms (2.96 ms)** | 19 ms |

- [`b2756f8`](https://github.com/amjadsaadeh/musicrs/commit/b2756f8278c7bf36e28386c87594b274ca8d2b51): the first benchmark. Its results were committed with that commit;
  which code they were measured on was not recorded.
- [`4b52605`](https://github.com/amjadsaadeh/musicrs/commit/4b52605f9151899167d8bc147e4d004e8556bfbf): `main` after #23–#29. Among other changes, IFB-MUSIC builds its
  cross-spectral matrices one bin at a time (#23) and the channels share one FFT planner (#26).
  Over 80 bins IFB-MUSIC became ~3× faster, while single-bin IFB-MUSIC became 1.5–2× slower.
  Profiled later: most of a single-bin call was preprocessing, not MUSIC (see [`6abc8bd`](https://github.com/amjadsaadeh/musicrs/commit/6abc8bd980f1e0b2cfc3cab00e38822d1d29a597)).
- [`3e2b808`](https://github.com/amjadsaadeh/musicrs/commit/3e2b8081e300fc4e8b8a06d40613dc43ad410385): narrowband DOA computes the covariance from real FFTs (#34), and
  `spectrum` no longer allocates per angle. Narrowband DOA is 5–6× faster than at
  [`4b52605`](https://github.com/amjadsaadeh/musicrs/commit/4b52605f9151899167d8bc147e4d004e8556bfbf).
- [`6abc8bd`](https://github.com/amjadsaadeh/musicrs/commit/6abc8bd980f1e0b2cfc3cab00e38822d1d29a597): real input is checked and scaled in one vectorizable pass, the STFT uses a real
  FFT with reused buffers, and the per-bin cross-spectral matrices are built without
  allocating per window (#35). Single-bin IFB-MUSIC is 2.4× faster than at
  [`3e2b808`](https://github.com/amjadsaadeh/musicrs/commit/3e2b8081e300fc4e8b8a06d40613dc43ad410385), IFB-MUSIC over 80 bins 1.6× faster. Narrowband DOA,
  which shares the input check, is 11–22 % faster, within the run-to-run spread. This run
  was built with rustc 1.100 (the earlier ones with 1.88).

The raw samples of each run are in the repository, e.g.
`git show 14e54d6:bench/results/musicrs.csv` (run on [`4b52605`](https://github.com/amjadsaadeh/musicrs/commit/4b52605f9151899167d8bc147e4d004e8556bfbf)) and
`git show ed4faa7:bench/results/musicrs.csv` (run on [`3e2b808`](https://github.com/amjadsaadeh/musicrs/commit/3e2b8081e300fc4e8b8a06d40613dc43ad410385)); `bench/results/environment.txt`
in the same commit records the machine and the musicrs commit. The latest run (on [`6abc8bd`](https://github.com/amjadsaadeh/musicrs/commit/6abc8bd980f1e0b2cfc3cab00e38822d1d29a597)) is
the one in [`bench/results/`](bench/results/).

### Distribution of the timings

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/benchmark-distribution-dark.svg">
  <img alt="Histograms of time per call divided by the median, one per workload and implementation. All are right-skewed: a sharp peak near the fastest calls and a tail of slower calls. No call takes more than 1.71 times the median of its case." src="docs/benchmark-distribution.svg">
</picture>

- **Not normal.** Shapiro-Wilk rejects normality for every case, on t and on log t (p < 0.001;
  p ≈ 0.01–0.03 for pyMUSIC over 80 bins, which has only 30 calls). Every distribution is
  right-skewed (skew 0.6 to 3.1): a sharp edge
  near the fastest call and a tail of slower ones, the shape of interference that can only add
  time. A log-normal fits better than a normal in every case, but none is log-normal either.
- **Scheduling moves the mean by a few percent.** The mean is at most 3.8 % above the median,
  and the outliers (calls slower than median + 3 robust sigmas) raise the mean by at most
  2.8 % (*tail excess*). No call takes more than 1.71× the median of its case, so mean and
  median are both fair single numbers; they give the same ratios to within 3 %. Because the
  distributions are skewed, ± stddev overstates the spread below the mean; the p5 – p95
  column shows the real range.
- **Slow calls come in runs.** Lag-1 autocorrelation is 0.4 to 0.8, so neighbouring calls are
  alike. Isolated preemptions would hit calls independently; runs point to slower changes in
  machine state (CPU frequency, caches, the core the process runs on). pyMUSIC on single-bin
  IFB-MUSIC (2 s recording) also drifted: its last 10 % of calls were 12 % slower than its
  first.
- **Run to run**, numpy's means moved by up to 3.4 % and pyMUSIC's by up to 4.8 % against the
  run the day before; earlier comparisons gave up to 9 %. Read the numbers as about ± 20 %.
  The ratios between musicrs and numpy on narrowband DOA and on 80 bins (2.6× to 6.3×) are
  larger than that, and the ratios to pyMUSIC far larger. On single-bin IFB-MUSIC the two are
  level within that margin.

<details>
<summary>Shape statistics per case</summary>

| Workload | Implementation | Shape | Skew | Excess kurtosis | Shapiro p (t / log t) | (Mean − median) / median | Outliers | Tail excess | Lag-1 autocorr. | Drift |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Narrowband DOA · 1 s recording | pyMUSIC | right-skewed, heavy tail, not normal (closer to log-normal) | 1.94 | 7.5 | 4.3e-12 / 2.1e-10 | +1.0% | 5.0% | +0.7% | 0.54 | -2.1% |
|  | numpy (vectorized) | right-skewed, not normal (closer to log-normal) | 0.91 | 0.6 | 5.9e-19 / 3.1e-14 | +1.9% | 3.3% | +0.9% | 0.61 | +5.7% |
|  | musicrs | right-skewed, not normal (closer to log-normal) | 0.99 | 0.2 | 8.9e-26 / 6.6e-24 | +3.0% | 2.0% | +0.5% | 0.73 | -4.2% |
| Narrowband DOA · 2 s recording | pyMUSIC | right-skewed, heavy tail, not normal (closer to log-normal) | 1.94 | 6.4 | 5.4e-13 / 1.8e-11 | +1.0% | 8.0% | +1.1% | 0.40 | +0.3% |
|  | numpy (vectorized) | right-skewed, not normal (closer to log-normal) | 0.66 | -0.2 | 1.2e-16 / 3.2e-13 | +2.0% | 0.5% | +0.2% | 0.70 | +3.0% |
|  | musicrs | right-skewed, not normal (closer to log-normal) | 0.73 | -0.3 | 1.3e-20 / 1.5e-18 | +2.0% | 0.2% | +0.1% | 0.72 | +0.1% |
| IFB-MUSIC, 1 bin · 1 s recording | pyMUSIC | right-skewed, not normal (closer to log-normal) | 1.24 | 0.7 | 1.6e-12 / 1.5e-11 | +2.5% | 14.5% | +2.4% | 0.81 | +0.7% |
|  | numpy (vectorized) | right-skewed, not normal (closer to log-normal) | 0.56 | 0.1 | 2.8e-13 / 4.5e-10 | +0.3% | 0.2% | +0.1% | 0.66 | -3.3% |
|  | musicrs | right-skewed, heavy tail, not normal (closer to log-normal) | 2.02 | 9.2 | 1.8e-32 / 4.2e-30 | +2.3% | 5.6% | +1.1% | 0.60 | -5.1% |
| IFB-MUSIC, 1 bin · 2 s recording | pyMUSIC | right-skewed, heavy tail, not normal (closer to log-normal) | 3.05 | 11.2 | 1.9e-20 / 2.1e-19 | +2.9% | 19.5% | +2.8% | 0.74 | +12.0% |
|  | numpy (vectorized) | right-skewed, not normal (closer to log-normal) | 1.41 | 1.9 | 1.1e-27 / 2.7e-23 | +3.8% | 6.3% | +2.4% | 0.80 | +5.0% |
|  | musicrs | right-skewed, not normal (closer to log-normal) | 1.07 | 0.4 | 1.2e-26 / 6.2e-25 | +2.5% | 2.4% | +0.6% | 0.76 | -6.3% |
| IFB-MUSIC, 80 bins · 1 s recording | pyMUSIC | right-skewed, not normal (closer to log-normal) | 1.26 | 2.2 | 0.014 / 0.025 | +0.4% | 3.3% | +0.3% | 0.38 | +1.8% |
|  | numpy (vectorized) | right-skewed, not normal (closer to log-normal) | 1.05 | 1.4 | 5.9e-20 / 2e-16 | +1.1% | 3.1% | +0.6% | 0.62 | +2.2% |
|  | musicrs | right-skewed, not normal (closer to log-normal) | 1.20 | 1.0 | 2.1e-26 / 1.1e-24 | +1.6% | 3.7% | +0.7% | 0.62 | -2.5% |

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
