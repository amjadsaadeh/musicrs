# CLAUDE.md

## Project Overview

**musicrs** is a Rust port of the MUSIC (MUltiple SIgnal Classification) algorithm for Direction-of-Arrival (DOA) estimation with microphone/antenna arrays. Based on R. O. Schmidt's 1986 algorithm and the Python pyMUSIC implementation.

## Commands

```bash
# Build
cargo build
cargo build --release

# Test
cargo test              # all tests except slow ones (#[ignore = "slow"])
cargo test -- --include-ignored   # full suite, including slow tests (CI runs this)
cargo test --lib        # unit tests only
cargo test --test '*'   # integration tests only
cargo test --test golden_test   # parity with the original pyMUSIC spectra
cargo clippy --all-targets -- -D warnings   # CI runs this and `cargo fmt --check`

# Documentation
cargo doc --open
```

## Architecture

```
src/
├── lib.rs      # Public API exports and crate-level docs
├── music.rs    # Core MUSIC algorithm (MusicEstimator, spectrum, DOA estimation)
└── stft.rs     # Signal processing (STFT, analytic signal via Hilbert transform)
tests/
├── common/mod.rs                # Shared WAV/filename/angle helpers
├── integration_test.rs          # Narrowband DOA on every WAV recording (±5°)
├── ifb_test.rs                  # IFB-MUSIC on the tone's bin (nfft 160), below Nyquist (±5°)
├── golden_test.rs               # Spectra vs. original pyMUSIC output in tests/golden/
├── broadband_test.rs            # IFB-MUSIC on composite multi-tone signals
├── noisy_data_test.rs           # Narrowband DOA with added white noise
├── golden/                      # pyMUSIC reference CSVs + generate_golden.py
└── data/single-source/*.wav     # 4-channel 24-bit 16kHz test recordings
```

**Key types:**
- `MusicEstimator` — main struct; configure antenna positions, angle search range/step, sample rate, and number of sources
- `MusicEstimator::estimate_doa_from_real()` — real-data entry point (analytic signal per channel, then `estimate_doa`), like pyMUSIC's `getSourceAngles`
- `MusicEstimator::ifb_music_spectrum()` — wideband variant; bin frequencies use the estimator's `sample_rate`, so tones above Nyquist cannot be recovered
- `analytic_signal()` — converts real signal to complex via Hilbert transform (FFT-based)
- `stft()` — Short-Time Fourier Transform with configurable window and overlap

**Core algorithm flow:**
1. Compute covariance matrix of complex multichannel input
2. Hermitian eigendecomposition (`faer`) → signal/noise subspace split
3. Compute MUSIC pseudo-spectrum using steering vectors
4. Peak detection → DOA estimates in radians

## Dependencies

| Crate | Purpose |
|-------|---------|
| `nalgebra` | Matrix types and linear algebra |
| `num-complex` | Complex number arithmetic |
| `rustfft` | FFT for STFT and Hilbert transform |
| `faer` | Numerically stable Hermitian eigendecomposition |
| `hound` (dev) | Reading WAV test fixtures |
| `approx` (dev) | Floating-point approximate equality in tests |

## Testing

Integration tests load WAV files from `tests/data/single-source/` and verify DOA estimates within **±5°** of the known ground truth. File naming convention:

```
single-source_<freq>Hz_<doa>deg_<duration>s.wav
```

Test data covers:
- Frequencies: 300 Hz – 9.1 kHz (in ~1.1 kHz steps)
- Angles: 0°, 40°, 80°, 120°, 160°
- Durations: 1 s and 2 s
- Array: Kinect 4-mic geometry

## Workflow

- Before implementing a non-trivial change, propose a plan and get it approved first. While planning, ask clarifying questions about anything ambiguous (scope, API shape, defaults, test data) rather than guessing.

- Make changes on a branch, never directly on `main`:
  - `feat/<feature-name>` for new features
  - `fix/<bugfix-name>` for bug fixes
- Structure work in reasonably sized commits: each commit is one logical change that builds and passes tests on its own.
- A project Stop hook (`.claude/settings.json` → `.claude/hooks/run-tests.sh`) runs `cargo test` when a turn ends with changed code under `src/`, `tests/` or `Cargo.*`, and sends failures back to Claude. Tests that take more than ~10 s are marked `#[ignore = "slow"]`, so the hook skips them; run `cargo test -- --include-ignored` for the full suite. Set `MUSICRS_SKIP_HOOK_TESTS=1` before starting Claude Code to turn the hook off.

## Notes

- Rust edition 2024, MSRV follows current stable toolchain
- No `rustfmt.toml` or `clippy.toml` — use `cargo fmt` and `cargo clippy` defaults
- Golden files: `*.eigh.csv` (pyMUSIC with `np.linalg.eigh`) must match value-by-value; `*.eig.csv` (pyMUSIC as published) only by peak location, because `np.linalg.eig` returns a non-orthogonal noise basis on this near-noiseless data. Regenerate with the command in `tests/golden/generate_golden.py`.
- Sign convention: musicrs uses a non-conjugated analytic signal with an `exp(+j2πfτ)` steering vector; pyMUSIC conjugates the analytic signal and uses `exp(−j2πfτ)`. Equivalent below Nyquist. The above-Nyquist sign flip and the real steering vector at Nyquist are musicrs additions tuned to the test data.
