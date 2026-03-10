// Integration tests for musicrs with added white noise.
//
// For each WAV file in `tests/data/single-source/` the real audio is
// corrupted with additive white Gaussian noise (AWGN) at three SNR levels
// (10 dB, 5 dB, 1 dB) and the DOA estimate is checked against the known
// ground-truth angle encoded in the filename.

use std::fs;
use std::path::{Path, PathBuf};

use musicrs::{MusicEstimator, analytic_signal};
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

/// Read a 4-channel, 24-bit PCM WAV file with `hound` and return
/// `(sample_rate, data)` where `data` is a `(4 × n_samples)` real matrix
/// normalised to `[-1, 1]`.
fn read_wav_4ch(path: &PathBuf) -> (u32, DMatrix<f64>) {
    let mut reader = hound::WavReader::open(path).expect("Failed to open WAV file");
    let spec = reader.spec();
    assert_eq!(spec.channels, 4, "Expected 4-channel WAV");
    assert_eq!(spec.bits_per_sample, 24, "Expected 24-bit PCM");

    let raw_samples: Vec<i32> = reader
        .samples::<i32>()
        .map(|s| s.expect("Sample read error"))
        .collect();

    let n_channels = spec.channels as usize;
    let n_samples = raw_samples.len() / n_channels;
    let scale = 2.0_f64.powi(23);

    let mut data = DMatrix::<f64>::zeros(n_channels, n_samples);
    for (idx, &sample) in raw_samples.iter().enumerate() {
        let ch = idx % n_channels;
        let t = idx / n_channels;
        data[(ch, t)] = sample as f64 / scale;
    }

    (spec.sample_rate, data)
}

/// Parse metadata from a test-data filename.
/// Format: `single-source_<freq>Hz_<doa>deg_<duration>s.wav`
fn parse_test_file_meta_params(path: &Path) -> Option<Vec<String>> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|file_name| {
            file_name
                .strip_suffix(".wav")
                .unwrap()
                .split('_')
                .map(str::to_owned)
                .collect()
        })
}

/// Core helper shared by all SNR-level tests.
///
/// Loads every WAV from `tests/data/single-source/`, corrupts it with AWGN at
/// `snr_db`, runs the MUSIC estimator, and asserts that the error is within
/// `tolerance_deg` for each file.
fn run_noisy_doa_test(snr_db: f64, tolerance_deg: f64) {
    let estimator = MusicEstimator::default();

    let paths = fs::read_dir("./tests/data/single-source")
        .expect("Test data directory not found")
        .filter(|p| {
            let p = p.as_ref().unwrap().path();
            p.extension().and_then(|ext| ext.to_str()) == Some("wav")
        });

    let mut total = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for (file_idx, entry) in paths.enumerate() {
        let path = entry.unwrap().path();
        let parts = parse_test_file_meta_params(&path).unwrap();

        let freq_hz: f64 = parts[1].strip_suffix("Hz").unwrap().parse().unwrap();
        let true_doa_deg: f64 = parts[2].strip_suffix("deg").unwrap().parse().unwrap();

        // ── Read WAV, add noise, build complex signal matrix ──────────────────
        let (sample_rate, real_data) = read_wav_4ch(&path);
        // Use file index as part of the seed so each file gets different noise
        // while remaining reproducible across test runs.
        let seed = (snr_db as u64).wrapping_mul(1_000) ^ (file_idx as u64).wrapping_mul(7_919);
        let noisy_data = add_white_noise(&real_data, snr_db, seed);

        let n_mics = noisy_data.nrows();
        let n_samples = noisy_data.ncols();

        let mut complex_data = DMatrix::<num_complex::Complex<f64>>::zeros(n_mics, n_samples);
        for m in 0..n_mics {
            let row: Vec<f64> = noisy_data.row(m).iter().copied().collect();
            let analytic = analytic_signal(&row);
            for (t, c) in analytic.into_iter().enumerate() {
                complex_data[(m, t)] = c;
            }
        }

        // ── Estimate DOA ──────────────────────────────────────────────────────
        let estimated_angles = estimator.estimate_doa(&complex_data, 1, freq_hz);

        let estimated_deg = if estimated_angles.is_empty() {
            failures.push(format!(
                "{:?}: no peak found (expected {:.1}°, SNR {snr_db:.0} dB)",
                path.file_name().unwrap(),
                true_doa_deg
            ));
            total += 1;
            continue;
        } else {
            estimated_angles[0].to_degrees()
        };

        // ── Check accuracy ────────────────────────────────────────────────────
        let nyquist = sample_rate as f64 / 2.0;
        let is_nyquist = (freq_hz - nyquist).abs() < 0.5;
        let angular_dist = |a: f64, b: f64| -> f64 {
            let d = (a - b).abs();
            if d > 180.0 { 360.0 - d } else { d }
        };
        let equivalent_angles: &[f64] = if is_nyquist {
            &[
                true_doa_deg,
                360.0 - true_doa_deg,
                180.0 - true_doa_deg,
                180.0 + true_doa_deg,
            ]
        } else {
            &[true_doa_deg, 360.0 - true_doa_deg]
        };
        let diff_wrapped = equivalent_angles
            .iter()
            .map(|&t| angular_dist(estimated_deg, t))
            .fold(f64::INFINITY, f64::min);

        if diff_wrapped > tolerance_deg {
            failures.push(format!(
                "{} @ SNR {snr_db:.0} dB: expected {:.1}° (equivalents: {:?}), \
                 got {:.1}° (error {:.1}°)",
                path.file_name().unwrap().to_str().unwrap(),
                true_doa_deg,
                equivalent_angles,
                estimated_deg,
                diff_wrapped,
            ));
        }
        total += 1;
    }

    if !failures.is_empty() {
        panic!(
            "{}/{} files failed noisy DOA test (SNR {snr_db:.0} dB, tolerance ±{tolerance_deg:.0}°):\n{}",
            failures.len(),
            total,
            failures.join("\n")
        );
    }

    println!(
        "All {total} noisy DOA tests passed \
         (SNR {snr_db:.0} dB, tolerance ±{tolerance_deg:.0}°)."
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

/// DOA estimation under 10 dB SNR.
///
/// At 10 dB the noise power is one-tenth of the signal power — a comfortable
/// operating regime for MUSIC. Tolerances are widened slightly compared with
/// the clean-signal baseline (±5°) to account for residual estimation error.
#[test]
fn test_single_source_doa_noisy_10db() {
    run_noisy_doa_test(10.0, 10.0);
}

/// DOA estimation under 5 dB SNR.
///
/// At 5 dB signal and noise are of comparable magnitude. MUSIC degrades
/// gracefully but some additional angular error is expected.
#[test]
fn test_single_source_doa_noisy_5db() {
    run_noisy_doa_test(5.0, 15.0);
}

/// DOA estimation under 1 dB SNR.
///
/// At 1 dB the noise is almost as strong as the signal. This tests the
/// lower bound of practical MUSIC performance with the available recording
/// length and array geometry.
#[test]
fn test_single_source_doa_noisy_1db() {
    run_noisy_doa_test(1.0, 20.0);
}
