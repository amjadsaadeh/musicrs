"""Generate golden MUSIC spectra with the original pyMUSIC implementation.

Usage (from the repo root):

    git clone https://github.com/amjadsaadeh/pyMUSIC /path/to/pyMUSIC
    git clone https://github.com/Trion/pySoundToolbox /path/to/pySoundToolbox
    uv run --python 3.10 --with 'numpy==1.23.5' python -I tests/golden/generate_golden.py \
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

Output: CSVs in tests/golden/, columns `angle_rad,spectrum`.
"""

import os
import sys
import types
import wave

import numpy as np

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
]

# Multi-bin IFB on the sum of several recordings at the same DOA (one bin per tone)
COMPOSITE_CASES = [
    ("composite_300+1400+2500Hz_40deg_2.0s", [300, 1400, 2500], 40, "2.0s"),
]

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
np.linalg.eig = original_eig
