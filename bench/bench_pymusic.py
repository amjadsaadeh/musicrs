"""Time the original pyMUSIC on the benchmark workloads and write every sample to a CSV.

Usage (from the repo root; same environment as tests/golden/generate_golden.py):

    git clone https://github.com/amjadsaadeh/pyMUSIC /path/to/pyMUSIC
    git clone https://github.com/Trion/pySoundToolbox /path/to/pySoundToolbox
    uv run --python 3.10 --with 'numpy==1.23.5' python -I bench/bench_pymusic.py \
        /path/to/pyMUSIC /path/to/pySoundToolbox bench/results/pymusic.csv

pyMUSIC runs as published (np.linalg.eig). The only shim is the Python-2 index fix in
pySoundToolbox.genAnalyticSignal (`n / 2` -> `n // 2`), as in generate_golden.py.

The workloads, sample counts and CSV format mirror examples/bench_pymusic.rs; see
bench/README.md.
"""

import os
import sys
import time
import types
import wave

import numpy as np

pymusic_dir, toolbox_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
# Scales every sample count (e.g. 0.1 for a quick smoke run)
SCALE = float(os.environ.get("BENCH_SCALE", "1"))

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
import pyMUSIC  # noqa: E402

DATA_DIR = "tests/data/single-source"
NFFT = 160
FREQ = 1400
WARMUP = 3


def read_wav_4ch(path):
    """24-bit PCM, 4 channels -> (4, n) float array normalised by 2**23 (as the Rust tests)."""
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 4 and w.getsampwidth() == 3
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.uint8)
    b = raw.reshape(-1, 3).astype(np.int32)
    v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    v = np.where(v & 0x800000, v - (1 << 24), v)
    return (v.reshape(-1, 4).T / 2.0**23).astype(np.float64)


est = pyMUSIC.MUSICEst()
# Every non-DC bin, given explicitly: pyMUSIC's bins=None would also include DC, musicrs's not
ALL_BINS = list(range(1, NFFT // 2 + 1))

# (workload, function of the data, recordings, samples per recording); keep in sync with the
# Rust side. IFB over all bins takes ~11 s per call in pyMUSIC (the 720-angle spectrum per bin
# dominates, not the recording length), so it runs on the 1 s recording only.
WORKLOADS = [
    ("narrowband", lambda d: est.getSourceAngles(d, 1, FREQ), ("1.0", "2.0"), 200),
    ("ifb_one_bin", lambda d: pyMUSIC.IFBMUSICSpectrum(d, est.antennaPositions, NFFT, bins=[FREQ * NFFT // 16000]), ("1.0", "2.0"), 200),
    ("ifb_all_bins", lambda d: pyMUSIC.IFBMUSICSpectrum(d, est.antennaPositions, NFFT, bins=ALL_BINS), ("1.0",), 30),
]

os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
with open(out_path, "w") as out:
    out.write("implementation,workload,recording_s,iteration,seconds\n")
    for name, run, durations, samples in WORKLOADS:
        for duration in durations:
            data = read_wav_4ch(os.path.join(DATA_DIR, f"single-source_{FREQ}Hz_40deg_{duration}s.wav"))
            n = max(5, round(samples * SCALE))
            for _ in range(WARMUP):
                run(data)
            for i in range(n):
                t0 = time.perf_counter_ns()
                run(data)
                dt = time.perf_counter_ns() - t0
                out.write(f"pymusic,{name},{duration},{i},{dt / 1e9!r}\n")
            out.flush()
            print(f"pymusic {name} {duration}s: {n} samples", file=sys.stderr)
