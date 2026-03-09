# CLAUDE.md

## Project Overview

**musicrs** is a Rust port of the MUSIC (MUltiple SIgnal Classification) algorithm for Direction-of-Arrival (DOA) estimation with microphone/antenna arrays. Based on R. O. Schmidt's 1986 algorithm and the Python pyMUSIC implementation.

## Commands

```bash
# Build
cargo build
cargo build --release

# Test
cargo test              # all tests (unit + integration)
cargo test --lib        # unit tests only
cargo test --test '*'   # integration tests only

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
├── integration_test.rs          # End-to-end tests against WAV recordings
└── data/single-source/*.wav     # 4-channel 24-bit 16kHz test recordings
```

**Key types:**
- `MusicEstimator` — main struct; configure antenna positions, angle search range/step, sample rate, and number of sources
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

## Notes

- Rust edition 2024, MSRV follows current stable toolchain
- No `rustfmt.toml` or `clippy.toml` — use `cargo fmt` and `cargo clippy` defaults
- There is a typo in the existing readme filename (`REAME.md`); do not rename it without checking history
