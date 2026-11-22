"""Time a vectorized numpy implementation of pyMUSIC's algorithm and write every sample to a CSV.

Usage (from the repo root; same environment as bench/bench_pymusic.py):

    uv run --python 3.10 --with 'numpy==1.23.5' python -I bench/bench_numpy.py \
        /path/to/pyMUSIC /path/to/pySoundToolbox bench/results/numpy.csv

pyMUSIC spends almost all of its time in a Python loop over the 720 angles that builds every
steering vector element by element in float128 with np.matrix. Timing only that measures how
pyMUSIC is written, not what Python with numpy can do. This script runs the same algorithm
written the way numpy is meant to be used, as the baseline musicrs should be measured against:

  * the same analytic signal (pySoundToolbox.genAnalyticSignal: FFT, one-sided spectrum, IFFT,
    conjugate), the same covariance and cross-spectral matrices, the same np.linalg.eig with
    eigenvectors sorted by |eigenvalue|, the same steering vectors exp(-j 2 pi f tau) and the
    same peak (the first maximum of the spectrum);
  * but in float64, with the steering vectors of all angles built in one array operation
    every call (nothing is cached between calls), all channels transformed at once, and
    1 / ||En^H a||^2 instead of a^H En En^H a (equal, as En En^H is Hermitian);
  * the IFB geometric mean is accumulated as a sum of logarithms, as in musicrs.

numpy is limited to one thread (musicrs is single-threaded). Before timing, every workload is
checked against pyMUSIC itself on the same recording: same peak angle, spectrum within a
relative 1e-6. The workloads, sample counts and CSV format mirror bench_pymusic.py and
examples/bench_pymusic.rs; see bench/README.md.
"""

import os

# Before numpy is imported: one BLAS/LAPACK thread, as musicrs
for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"

import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
import wave  # noqa: E402

import numpy as np  # noqa: E402

pymusic_dir, toolbox_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
SCALE = float(os.environ.get("BENCH_SCALE", "1"))

DATA_DIR = "tests/data/single-source"
NFFT = 160
FREQ = 1400
WARMUP = 3
SAMPLE_RATE = 16000
SPEED_OF_SOUND = 343.2  # m/s, as pyMUSIC
# pyMUSIC's default MUSICEst: Kinect array, 720 angles over [0, 2 pi)
MIC_POSITIONS = np.array([[0.113, -0.036, -0.076, -0.113], [0.0, 0.0, 0.0, 0.0]])
ANGLE_STEPS = 720


def angle_grid():
    return np.arange(0, 2 * np.pi, 2 * np.pi / ANGLE_STEPS)


def spectrum(noise_mat, frequency):
    """MUSIC pseudo-spectrum over the angle grid, as MUSICEst.getSpectrum (cov mode)."""
    angles = angle_grid()
    delays = MIC_POSITIONS.T @ np.vstack([np.cos(angles), np.sin(angles)]) / SPEED_OF_SOUND
    steering = np.exp(-2j * np.pi * frequency * delays)  # (mics, angles)
    proj = noise_mat.conj().T @ steering  # (noise dims, angles)
    return 1.0 / np.einsum("ij,ij->j", proj.conj(), proj).real, angles


def noise_mat(cov, n_sources):
    """As MUSICEst.getNoiseMat: np.linalg.eig, eigenvectors of the smallest |eigenvalues|."""
    values, vectors = np.linalg.eig(cov)
    return vectors[:, np.argsort(np.abs(values), kind="stable")[: cov.shape[0] - n_sources]]


def analytic_signal(data):
    """pySoundToolbox.genAnalyticSignal, for all channels (rows) at once."""
    n = data.shape[1]
    h = np.zeros(n)
    h[0] = 1
    h[n // 2] = 1
    h[1 : n // 2] = 2
    return np.fft.ifft(np.fft.fft(data, axis=1) * h, axis=1).conj()


def source_angles(data, n_sources, frequency):
    """As MUSICEst.getSourceAngles for one source: the angle of the first maximum."""
    a = analytic_signal(data)
    cov = a @ a.conj().T / data.shape[1]
    spec, angles = spectrum(noise_mat(cov, n_sources), frequency)
    return angles[np.argsort(-spec, kind="stable")[:n_sources]]


def ifb_music_spectrum(data, nfft, bins, n_sources=1):
    """As pyMUSIC.IFBMUSICSpectrum: non-overlapping rectangular STFT, geometric mean of bins."""
    n_windows = data.shape[1] // nfft
    frames = data[:, : n_windows * nfft].reshape(data.shape[0], n_windows, nfft)
    stft = np.fft.rfft(frames, axis=2)  # (mics, windows, bins)
    freqs = np.fft.rfftfreq(nfft, 1 / SAMPLE_RATE)
    log_sum = np.zeros(ANGLE_STEPS)
    for b in bins:
        x = stft[:, :, b]
        cross = x @ x.conj().T / n_windows
        # pyMUSIC's 'spec' mode: the cross-spectral matrix rotates the other way, so -f
        spec, angles = spectrum(noise_mat(cross, n_sources), -freqs[b])
        log_sum += np.log(spec)
    return np.exp(log_sum / len(bins)), angles


def read_wav_4ch(path):
    """24-bit PCM, 4 channels -> (4, n) float array normalised by 2**23 (as the Rust tests)."""
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 4 and w.getsampwidth() == 3
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.uint8)
    b = raw.reshape(-1, 3).astype(np.int32)
    v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    v = np.where(v & 0x800000, v - (1 << 24), v)
    return (v.reshape(-1, 4).T / 2.0**23).astype(np.float64)


def load_pymusic():
    """pyMUSIC with the pySoundToolbox shim of bench_pymusic.py, for the parity check only."""
    with open(os.path.join(toolbox_dir, "pySoundToolbox.py")) as f:
        src = f.read()
    start = src.index("def genAnalyticSignal")
    end = src.index("\nclass ", start)
    toolbox_mod = types.ModuleType("pySoundToolbox.pySoundToolbox")
    toolbox_mod.np = np
    exec(src[start:end].replace("n / 2", "n // 2"), toolbox_mod.__dict__)
    pkg = types.ModuleType("pySoundToolbox")
    pkg.pySoundToolbox = toolbox_mod
    sys.modules["pySoundToolbox"] = pkg
    sys.modules["pySoundToolbox.pySoundToolbox"] = toolbox_mod
    sys.path.insert(0, pymusic_dir)
    import pyMUSIC

    return pyMUSIC


def check_parity(pymusic, name, data):
    """Same result as pyMUSIC on this recording, or the timings would not be comparable."""
    est = pymusic.MUSICEst()
    if name == "narrowband":
        ref, got = est.getSourceAngles(data, 1, FREQ), source_angles(data, 1, FREQ)
        # The linear array cannot tell theta from 2 pi - theta: both peaks are equal up to
        # rounding, so either may come first
        mirror = np.mod(2 * np.pi - got, 2 * np.pi)
        assert np.allclose(ref, got) or np.allclose(ref, mirror), \
            f"{name}: pyMUSIC {np.degrees(ref)}, numpy {np.degrees(got)}"
        # The angle alone could agree by chance: compare the spectra too
        a = analytic_signal(data)
        nm = est.getNoiseMat(np.matrix(a @ a.conj().T / data.shape[1]), 1)
        ref_spec, _ = est.getSpectrum(nm, FREQ)
        got_spec, _ = spectrum(noise_mat(a @ a.conj().T / data.shape[1], 1), FREQ)
    else:
        bins = BINS[name]
        ref_spec, _ = pymusic.IFBMUSICSpectrum(data, est.antennaPositions, NFFT, bins=bins)
        got_spec, _ = ifb_music_spectrum(data, NFFT, bins)
    rel = float(np.max(np.abs(got_spec / ref_spec - 1)))
    # pyMUSIC's peak is also a peak here (ties between mirror angles may break either way)
    assert rel < 1e-6 and got_spec[np.argmax(ref_spec)] >= got_spec.max() * (1 - 1e-6), \
        f"{name}: rel. diff {rel}"
    return rel


BINS = {"ifb_one_bin": [FREQ * NFFT // SAMPLE_RATE], "ifb_all_bins": list(range(1, NFFT // 2 + 1))}

# Same workloads, recordings and sample counts as examples/bench_pymusic.rs
WORKLOADS = [
    ("narrowband", lambda d: source_angles(d, 1, FREQ), ("1.0", "2.0"), 1000),
    ("ifb_one_bin", lambda d: ifb_music_spectrum(d, NFFT, BINS["ifb_one_bin"]), ("1.0", "2.0"), 1000),
    ("ifb_all_bins", lambda d: ifb_music_spectrum(d, NFFT, BINS["ifb_all_bins"]), ("1.0",), 1000),
]


def main():
    pymusic = load_pymusic()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as out:
        out.write("implementation,workload,recording_s,iteration,seconds\n")
        for name, run, durations, samples in WORKLOADS:
            for duration in durations:
                data = read_wav_4ch(os.path.join(DATA_DIR, f"single-source_{FREQ}Hz_40deg_{duration}s.wav"))
                rel = check_parity(pymusic, name, data)
                n = max(5, round(samples * SCALE))
                for _ in range(WARMUP):
                    run(data)
                for i in range(n):
                    t0 = time.perf_counter_ns()
                    run(data)
                    dt = time.perf_counter_ns() - t0
                    out.write(f"numpy,{name},{duration},{i},{dt / 1e9!r}\n")
                out.flush()
                print(f"numpy {name} {duration}s: {n} samples (max rel. diff to pyMUSIC {rel:.1e})",
                      file=sys.stderr)


if __name__ == "__main__":
    main()
