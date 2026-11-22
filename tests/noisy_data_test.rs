// Integration tests for musicrs with added white noise.
//
// For each WAV file in `tests/data/single-source/` the real audio is
// corrupted with additive white Gaussian noise (AWGN) at three SNR levels
// (10 dB, 5 dB, 1 dB) and the DOA estimate is checked against the known
// ground-truth angle encoded in the filename.

mod common;

use musicrs::MusicEstimator;
use nalgebra::DMatrix;

// ─────────────────────────────────────────────────────────────────────────────
// Minimal seeded PRNG (LCG) + Box-Muller for Gaussian noise
// ─────────────────────────────────────────────────────────────────────────────

/// A simple linear-congruential PRNG used to produce reproducible noise
/// without requiring an external `rand` crate.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Returns a uniform sample in `(0, 1]`.
    fn next_f64(&mut self) -> f64 {
        // Knuth / Newlib multiplier and increment
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Use the upper 32 bits for better statistical quality
        let bits = (self.state >> 32) as u32;
        // Map to (0, 1] — avoid zero so ln() is always finite
        (bits as f64 + 1.0) / (u32::MAX as f64 + 2.0)
    }

    /// Box-Muller: returns a sample from N(0, 1).
    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Add additive white Gaussian noise (AWGN) to a multichannel signal.
///
/// The noise level is specified as a signal-to-noise ratio in decibels:
/// ```text
/// SNR_dB = 10 · log10(P_signal / P_noise)
/// ```
/// The signal power is computed per-channel and averaged across channels so
/// that the SNR is consistent regardless of the signal amplitude.
///
/// `seed` makes the noise reproducible across test runs.
fn add_white_noise(signal: &DMatrix<f64>, snr_db: f64, seed: u64) -> DMatrix<f64> {
    let n_channels = signal.nrows();
    let n_samples = signal.ncols();

    // Mean signal power across all channels and samples
    let signal_power: f64 =
        signal.iter().map(|x| x * x).sum::<f64>() / (n_channels * n_samples) as f64;

    // Desired noise standard deviation derived from the target SNR
    let noise_std = (signal_power / 10.0_f64.powf(snr_db / 10.0)).sqrt();

    let mut rng = Lcg::new(seed);
    let mut noisy = signal.clone();
    for sample in noisy.iter_mut() {
        *sample += noise_std * rng.next_gaussian();
    }
    noisy
}

/// Loads every WAV recording, corrupts it with AWGN at `snr_db`, runs narrowband MUSIC and
/// checks every estimate against `tolerance_deg`.
fn run_noisy_doa_test(snr_db: f64, tolerance_deg: f64) {
    let estimator = MusicEstimator::default();
    let mut log = common::ErrorLog::new(format!("narrowband WAV @ {snr_db} dB SNR"));

    for (file_idx, path) in common::wav_files().iter().enumerate() {
        let meta = common::parse_meta(path);
        let (_, real_data) = common::read_wav_4ch(path);
        // Seed from the SNR and the file's position in the (sorted) file list, so each
        // file gets different but reproducible noise.
        let seed = (snr_db as u64).wrapping_mul(1_000) ^ (file_idx as u64).wrapping_mul(7_919);
        let noisy = add_white_noise(&real_data, snr_db, seed);

        let estimate = estimator
            .estimate_doa_from_real(&noisy, 1, meta.frequency_hz)
            .unwrap();
        let truths =
            common::equivalent_angles(meta.doa_deg, meta.frequency_hz, estimator.sample_rate);
        let error = estimate.first().map_or(f64::INFINITY, |a| {
            common::angular_error(a.to_degrees(), &truths)
        });
        log.record(path.file_name().unwrap().to_string_lossy(), error);
    }

    log.check(tolerance_deg);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

/// DOA estimation under 10 dB SNR.
///
/// At 10 dB the noise power is one-tenth of the signal power — a comfortable
/// operating regime for MUSIC. Worst measured error is 3.0° (300 Hz, 0°); the
/// 10 dB and 5 dB tolerances leave headroom for a different noise draw.
#[test]
#[ignore = "slow"]
fn test_single_source_doa_noisy_10db() {
    run_noisy_doa_test(10.0, 5.0);
}

/// DOA estimation under 5 dB SNR.
///
/// At 5 dB signal and noise are of comparable magnitude. MUSIC degrades
/// gracefully. Worst measured error is 1.5°.
#[test]
#[ignore = "slow"]
fn test_single_source_doa_noisy_5db() {
    run_noisy_doa_test(5.0, 5.0);
}

/// DOA estimation under 1 dB SNR.
///
/// At 1 dB the noise is almost as strong as the signal. This tests the
/// lower bound of practical MUSIC performance with the available recording
/// length and array geometry. Worst measured error is 5.0° (300 Hz, 0°).
#[test]
#[ignore = "slow"]
fn test_single_source_doa_noisy_1db() {
    run_noisy_doa_test(1.0, 8.0);
}
