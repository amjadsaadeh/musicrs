"""Generate golden MUSIC spectra with the original pyMUSIC implementation.

Usage (from the repo root):

    git clone https://github.com/amjadsaadeh/pyMUSIC /path/to/pyMUSIC
    git clone https://github.com/Trion/pySoundToolbox /path/to/pySoundToolbox
    uv run --python 3.10 --with 'numpy==1.23.5' --with 'scipy==1.10.1' \
        python -I tests/golden/generate_golden.py \
        /path/to/pyMUSIC /path/to/pySoundToolbox

pyMUSIC was written against an old numpy/Python 2. numpy 1.23 still has the
`np.float` / `np.complex` aliases and still converts the 1x1 matrix in
`getSpectrum` to a scalar, so pyMUSIC.py runs unchanged. One shim remains:
  * pySoundToolbox.genAnalyticSignal uses `n / 2` as an index (Python 2
    integer division); it is re-executed with `n // 2`.
The algorithm itself is not modified.

Each case is computed twice:
  * `*.eig.csv`  – pyMUSIC exactly as published (`np.linalg.eig`).
  * `*.eigh.csv` – the same code with `np.linalg.eig` swapped for the Hermitian solver
    `np.linalg.eigh`. On this nearly noiseless data the noise eigenvalues are ~1e-15,
    and `eig` returns non-orthogonal noise eigenvectors, so the as-published spectrum
    depends on an arbitrary basis. Only its peak location is meaningful; musicrs
    (which uses a Hermitian solver) is compared value-by-value against `eigh`.

Output: CSVs in tests/golden/, columns `angle_rad,spectrum`; and `*.hilbert_n<N>.scipy.csv`
(columns `re,im`), the analytic signal from `scipy.signal.hilbert`, which does not
depend on pyMUSIC.
"""

import os
import sys
import types
import wave

import numpy as np
import scipy.signal

pymusic_dir, toolbox_dir = sys.argv[1], sys.argv[2]

# Load genAnalyticSignal with the Python-2 index fix, exposed as pySoundToolbox.pySoundToolbox
with open(os.path.join(toolbox_dir, "pySoundToolbox.py")) as f:
    src = f.read()
start = src.index("def genAnalyticSignal")
end = src.index("\nclass ", start)
func_src = src[start:end].replace("n / 2", "n // 2")
toolbox_mod = types.ModuleType("pySoundToolbox.pySoundToolbox")
toolbox_mod.np = np
exec(func_src, toolbox_mod.__dict__)
pkg = types.ModuleType("pySoundToolbox")
pkg.pySoundToolbox = toolbox_mod
sys.modules["pySoundToolbox"] = pkg
sys.modules["pySoundToolbox.pySoundToolbox"] = toolbox_mod

sys.path.insert(0, pymusic_dir)
import pyMUSIC  # noqa: E402

DATA_DIR = "tests/data/single-source"
OUT_DIR = "tests/golden"
SAMPLE_RATE = 16000
NFFT = 160  # 100 Hz bins: every test tone falls exactly on a bin


def read_wav_4ch(path):
    """24-bit PCM, 4 channels -> (4, n) float array normalised by 2**23 (same as the Rust tests)."""
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 4 and w.getsampwidth() == 3
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.uint8)
    b = raw.reshape(-1, 3).astype(np.int32)
    v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    v = np.where(v & 0x800000, v - (1 << 24), v)
    return (v.reshape(-1, 4).T / 2.0**23).astype(np.float64)


def write_csv(name, spectrum, angles):
    with open(os.path.join(OUT_DIR, name), "w") as f:
        f.write("angle_rad,spectrum\n")
        for a, s in zip(angles, spectrum):
            f.write(f"{float(a)!r},{float(s)!r}\n")


CASES = [
    ("single-source_300Hz_80deg_1.0s.wav", 300),
    ("single-source_1400Hz_40deg_1.0s.wav", 1400),
    ("single-source_4700Hz_120deg_1.0s.wav", 4700),
    ("single-source_2500Hz_0deg_2.0s.wav", 2500),
    ("single-source_5800Hz_160deg_2.0s.wav", 5800),
    ("single-source_6900Hz_120deg_1.0s.wav", 6900),
    # All five angles at 3600 Hz and 6900 Hz, alternating 1 s / 2 s recordings
    ("single-source_3600Hz_0deg_1.0s.wav", 3600),
    ("single-source_3600Hz_40deg_2.0s.wav", 3600),
    ("single-source_3600Hz_80deg_1.0s.wav", 3600),
    ("single-source_3600Hz_120deg_2.0s.wav", 3600),
    ("single-source_3600Hz_160deg_1.0s.wav", 3600),
    ("single-source_6900Hz_0deg_2.0s.wav", 6900),
    ("single-source_6900Hz_40deg_1.0s.wav", 6900),
    ("single-source_6900Hz_80deg_2.0s.wav", 6900),
    ("single-source_6900Hz_160deg_1.0s.wav", 6900),
]

# Multi-bin IFB on the sum of several recordings at the same DOA (one bin per tone)
COMPOSITE_CASES = [
    ("composite_300+1400+2500Hz_40deg_2.0s", [300, 1400, 2500], 40, "2.0s"),
    ("composite_3600+4700+5800Hz_120deg_1.0s", [3600, 4700, 5800], 120, "1.0s"),
]

# Narrowband with a search interval other than the full circle:
# (recording, frequency, interval name, interval in rad). 360 steps keep the 0.5° grid.
INTERVAL_CASES = [
    ("single-source_1400Hz_40deg_1.0s.wav", 1400, "0-180deg", (0, np.pi)),
    ("single-source_4700Hz_120deg_1.0s.wav", 4700, "180-360deg", (np.pi, 2 * np.pi)),
]

# IFB at other STFT sizes: (recording, nfft, bins). nfft 320 has 50 Hz bins, so the tone
# is on a bin; at nfft 1024 (15.625 Hz bins) it falls between the two bins used.
NFFT_CASES = [
    ("single-source_1400Hz_40deg_1.0s.wav", 320, [28]),
    ("single-source_4700Hz_120deg_1.0s.wav", 320, [94]),
    ("single-source_1400Hz_40deg_1.0s.wav", 1024, [89, 90]),
    ("single-source_4700Hz_120deg_1.0s.wav", 1024, [300, 301]),
]

# Narrowband with two sources: the sum of two recordings at different angles and
# frequencies (same-frequency recordings would be coherent, giving a rank-1 covariance).
# The second tone is off the analysed frequency, but the covariance has rank 2, so its
# two-dimensional noise subspace is well defined. (parts, analysed frequency, duration)
TWO_SOURCE_CASES = [
    ([(1400, 40), (2500, 120)], 1400, "1.0s"),
]

# IFB with two sources at different angles and frequencies, one bin per tone (each bin
# holds one source, so one source per bin): (parts, bins, duration)
TWO_SOURCE_IFB_CASES = [
    ([(1400, 40), (2500, 120)], [14, 25], "1.0s"),
]

# Analytic signal vs scipy: (recording, channel, lengths): even and odd lengths.
HILBERT_CASES = [
    ("single-source_1400Hz_40deg_1.0s.wav", 0, [2048, 2047]),
]


def narrowband(est, data, freq, n_sources=1):
    analytic = np.matrix(np.vstack([pyMUSIC.genAnalyticSignal(row) for row in data]).astype(np.complex128))
    cov = (analytic * analytic.H) / data.shape[1]
    noise_mat = est.getNoiseMat(cov, n_sources)
    return est.getSpectrum(noise_mat, freq)


def run_interval(fname, freq, interval_name, interval, solver):
    data = read_wav_4ch(os.path.join(DATA_DIR, fname))
    est = pyMUSIC.MUSICEst(angleStepsNum=360, searchInterval=interval)
    spectrum, angles = narrowband(est, data, freq)
    write_csv(f"{fname[:-4]}.narrowband_{interval_name}.{solver}.csv", spectrum, angles[:360])
    print("wrote", fname, interval_name, solver)


def run_nfft(fname, nfft, bins, solver):
    data = read_wav_4ch(os.path.join(DATA_DIR, fname))
    est = pyMUSIC.MUSICEst()
    spectrum, angles = pyMUSIC.IFBMUSICSpectrum(data, est.antennaPositions, nfft, bins=bins)
    write_csv(f"{fname[:-4]}.ifb_nfft{nfft}_bins{'-'.join(map(str, bins))}.{solver}.csv", spectrum, angles)
    print("wrote", fname, nfft, bins, solver)


def sum_recordings(parts, duration):
    """Sum of the recordings `(frequency, doa)` of one duration, trimmed to the shortest."""
    datas = [read_wav_4ch(os.path.join(DATA_DIR, f"single-source_{f}Hz_{doa}deg_{duration}.wav")) for f, doa in parts]
    n = min(d.shape[1] for d in datas)
    return sum(d[:, :n] for d in datas)


def run_two_sources(parts, freq, duration, solver):
    data = sum_recordings(parts, duration)
    spectrum, angles = narrowband(pyMUSIC.MUSICEst(), data, freq, n_sources=2)
    name = "+".join(f"{f}Hz_{doa}deg" for f, doa in parts)
    write_csv(f"composite_{name}_{duration}.narrowband_{freq}Hz_2src.{solver}.csv", spectrum, angles)
    print("wrote", name, solver)


def run_two_source_ifb(parts, bins, duration, solver):
    data = sum_recordings(parts, duration)
    est = pyMUSIC.MUSICEst()
    spectrum, angles = pyMUSIC.IFBMUSICSpectrum(data, est.antennaPositions, NFFT, bins=bins)
    name = "+".join(f"{f}Hz_{doa}deg" for f, doa in parts)
    write_csv(f"composite_{name}_{duration}.ifb_nfft{NFFT}_bins{'-'.join(map(str, bins))}.{solver}.csv", spectrum, angles)
    print("wrote", name, "ifb", solver)


def run_hilbert(fname, channel, lengths):
    data = read_wav_4ch(os.path.join(DATA_DIR, fname))
    for n in lengths:
        z = scipy.signal.hilbert(data[channel, :n])
        with open(os.path.join(OUT_DIR, f"{fname[:-4]}.ch{channel}.hilbert_n{n}.scipy.csv"), "w") as f:
            f.write("re,im\n")
            for c in z:
                f.write(f"{float(c.real)!r},{float(c.imag)!r}\n")
        print("wrote hilbert", fname, n)


def run_case(fname, freq, solver):
    data = read_wav_4ch(os.path.join(DATA_DIR, fname))
    stem = fname[: -len(".wav")]
    est = pyMUSIC.MUSICEst()

    # Narrowband: same steps as MUSICEst.getSourceAngles, stopping before its (buggy) peak picking
    analytic = np.matrix(np.vstack([pyMUSIC.genAnalyticSignal(row) for row in data]).astype(np.complex128))
    cov = (analytic * analytic.H) / data.shape[1]
    noise_mat = est.getNoiseMat(cov, 1)
    spectrum, angles = est.getSpectrum(noise_mat, freq)
    write_csv(f"{stem}.narrowband.{solver}.csv", spectrum, angles)

    # IFB on the single bin holding the tone
    bin_idx = freq * NFFT // SAMPLE_RATE
    spectrum, angles = pyMUSIC.IFBMUSICSpectrum(data, est.antennaPositions, NFFT, bins=[bin_idx])
    write_csv(f"{stem}.ifb_nfft{NFFT}_bin{bin_idx}.{solver}.csv", spectrum, angles)
    print("wrote", stem, solver)


def run_composite(name, freqs, doa, duration, solver):
    parts = [read_wav_4ch(os.path.join(DATA_DIR, f"single-source_{f}Hz_{doa}deg_{duration}.wav")) for f in freqs]
    n = min(p.shape[1] for p in parts)
    data = sum(p[:, :n] for p in parts)
    bins = [f * NFFT // SAMPLE_RATE for f in freqs]
    est = pyMUSIC.MUSICEst()
    spectrum, angles = pyMUSIC.IFBMUSICSpectrum(data, est.antennaPositions, NFFT, bins=bins)
    write_csv(f"{name}.ifb_nfft{NFFT}_bins{'-'.join(map(str, bins))}.{solver}.csv", spectrum, angles)
    print("wrote", name, solver)


original_eig = np.linalg.eig
for solver, eig in [("eig", original_eig), ("eigh", np.linalg.eigh)]:
    np.linalg.eig = eig
    for fname, freq in CASES:
        run_case(fname, freq, solver)
    for name, freqs, doa, duration in COMPOSITE_CASES:
        run_composite(name, freqs, doa, duration, solver)
    for fname, freq, interval_name, interval in INTERVAL_CASES:
        run_interval(fname, freq, interval_name, interval, solver)
    for fname, nfft, bins in NFFT_CASES:
        run_nfft(fname, nfft, bins, solver)
    for parts, freq, duration in TWO_SOURCE_CASES:
        run_two_sources(parts, freq, duration, solver)
    for parts, bins, duration in TWO_SOURCE_IFB_CASES:
        run_two_source_ifb(parts, bins, duration, solver)
np.linalg.eig = original_eig

for fname, channel, lengths in HILBERT_CASES:
    run_hilbert(fname, channel, lengths)
