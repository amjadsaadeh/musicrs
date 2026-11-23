# Runtime benchmark: musicrs vs pyMUSIC and numpy

Times three implementations of the same algorithm on the same calls and recordings, keeps
every timed call, and summarises the distributions. The results and charts are in the
top-level README, section *Performance vs pyMUSIC and numpy*.

- **pyMUSIC**: the original [pyMUSIC](https://github.com/amjadsaadeh/pyMUSIC), unmodified
  (`bench_pymusic.py`).
- **numpy (vectorized)**: pyMUSIC's algorithm written idiomatically in numpy
  (`bench_numpy.py`). pyMUSIC spends almost all of its time in a Python loop that builds every
  steering vector element by element in `float128` with `np.matrix`, so on its own it
  measures how pyMUSIC is written, not what Python can do. This one keeps the same analytic
  signal, `np.linalg.eig`, eigenvector order, steering vectors and peak, but works in
  `float64` and builds the steering vectors of all angles in one array operation each call
  (nothing is cached between calls). It runs on one BLAS thread, as musicrs runs on one
  thread. Before timing each case, it checks its result against pyMUSIC's on the same
  recording (spectrum within a relative 1e-6, same peak up to the θ / 2π − θ ambiguity of the
  linear array) and stops if they differ.
- **musicrs** (`examples/bench_pymusic.rs`).

## Workloads

All use the 1400 Hz / 40° recordings in `tests/data/single-source/` (4 channels, 16 kHz)
and the default estimator (Kinect array, 720 angles over [0, 2π)).

| Workload | pyMUSIC | numpy (`bench_numpy.py`) | musicrs | Recordings |
|---|---|---|---|---|
| `narrowband` | `MUSICEst.getSourceAngles(data, 1, 1400)` | `source_angles(data, 1, 1400)` | `estimate_doa_from_real(&data, 1, 1400.0)` | 1 s, 2 s |
| `ifb_one_bin` | `IFBMUSICSpectrum(data, pos, 160, bins=[14])` | `ifb_music_spectrum(data, 160, [14])` | `ifb_music_spectrum(&data, 160, 1, Some(&[14]))` | 1 s, 2 s |
| `ifb_all_bins` | `IFBMUSICSpectrum(data, pos, 160, bins=range(1, 81))` | `ifb_music_spectrum(data, 160, range(1, 81))` | `ifb_music_spectrum(&data, 160, 1, Some(&(1..=80)))` | 1 s |

Each call includes everything from the real-valued samples to the result: analytic signal
or STFT, covariance, eigendecomposition and the spectrum. Reading the WAV file is not timed.
`ifb_all_bins` lists the 80 non-DC bins explicitly because pyMUSIC's `bins=None` would also
include DC and musicrs's would not. It runs on the 1 s recording only: one pyMUSIC call
takes about 11 s, and the cost is the 720-angle spectrum per bin, not the recording length.

All harnesses make 3 untimed warm-up calls, then time every call on its own
(`time.perf_counter_ns` / `std::time::Instant`) and write it to a CSV. pyMUSIC gets 200
samples per case (30 for `ifb_all_bins`), numpy and musicrs 1000. pyMUSIC runs unmodified with
`np.linalg.eig`; the only shim is the Python-2 index fix in pySoundToolbox, as in
`tests/golden/generate_golden.py`.

## Statistics

`analyze.py` reports mean ± stddev and the median for every case, and investigates the
shape of the distribution, because scheduling noise (preemption, interrupts, frequency
changes) only ever adds time and so appears as a right tail rather than as symmetric
spread:

- spread: MAD, p5/p95, min/max, coefficient of variation (stddev / mean);
- shape: skewness, excess kurtosis, Shapiro-Wilk tests on t and on log t, and whether a
  normal or a log-normal fit has the lower AIC;
- outliers: calls slower than median + 3 robust sigmas (1.4826 · MAD), their share, and the
  *tail excess*, how much they raise the mean over the mean of the other calls.

A small gap between mean and median and a small tail excess mean scheduling has little
influence on the result; the median is then a fair single number, and the mean ± stddev
an honest one.

## Reproduce

Run the three benchmarks one after the other on an otherwise idle machine, so they do not
compete for cores:

```sh
git clone https://github.com/amjadsaadeh/pyMUSIC /path/to/pyMUSIC
git clone https://github.com/Trion/pySoundToolbox /path/to/pySoundToolbox

cargo run --release --example bench_pymusic -- bench/results/musicrs.csv      # ~1 min
uv run --python 3.10 --with 'numpy==1.23.5' python -I bench/bench_pymusic.py \
    /path/to/pyMUSIC /path/to/pySoundToolbox bench/results/pymusic.csv        # ~10 min
uv run --python 3.10 --with 'numpy==1.23.5' python -I bench/bench_numpy.py \
    /path/to/pyMUSIC /path/to/pySoundToolbox bench/results/numpy.csv          # ~1 min
uv run --python 3.12 --with numpy --with scipy --with matplotlib \
    python -I bench/analyze.py
```

`BENCH_SCALE=0.1` scales every sample count down for a quick check (all harnesses).
`analyze.py` writes `bench/results/summary.{csv,md}` and the charts in `docs/`.
`bench/results/environment.txt` records the machine and the musicrs commit the committed
results come from; re-run all three after a change to musicrs, so the README never compares
one implementation's old numbers with another's new ones. After a re-run, add a column for
the new musicrs commit to *musicrs over time* in the README, linking the commit the results
were measured on (`environment.txt`).
