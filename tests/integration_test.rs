// Integration tests for musicrs
//
// Tests DOA estimation against the pre-recorded single-source WAV test data.
// Each WAV file is a 4-channel, 24-bit PCM, 16 kHz recording whose filename
// encodes the source frequency, true DOA, and duration.

use std::fs;
use std::path::PathBuf;

use nalgebra::DMatrix;
use musicrs::{MusicEstimator, analytic_signal};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Parse metadata from a test-data filename.
/// Format: `single-source_<freq>Hz_<doa>deg_<duration>s.wav`
/// Returns `(frequency_hz, doa_degrees, duration_s)`.
fn parse_test_file_meta_params(path: &PathBuf) -> Option<Vec<&str>> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|file_name| {
            file_name
                .strip_suffix(".wav")
                .unwrap()
                .split('_')
                .collect::<Vec<&str>>()
        })
}

/// Read a 4-channel, 24-bit PCM WAV file with `hound` and return
/// `(sample_rate, data)` where `data` is a `(4 × n_samples)` real matrix
/// normalised to `[-1, 1]`.
fn read_wav_4ch(path: &PathBuf) -> (u32, DMatrix<f64>) {
    let mut reader = hound::WavReader::open(path).expect("Failed to open WAV file");
    let spec = reader.spec();
    assert_eq!(spec.channels, 4, "Expected 4-channel WAV");
    assert_eq!(spec.bits_per_sample, 24, "Expected 24-bit PCM");

    // hound reads 24-bit samples as i32 (zero-padded in the low bits)
    let raw_samples: Vec<i32> = reader
        .samples::<i32>()
        .map(|s| s.expect("Sample read error"))
        .collect();

    let n_channels = spec.channels as usize;
    let n_samples = raw_samples.len() / n_channels;
    let scale = 2.0_f64.powi(23); // 24-bit signed → [-1, 1]

    // De-interleave: WAV stores [ch0, ch1, ch2, ch3, ch0, ch1, ...]
    let mut data = DMatrix::<f64>::zeros(n_channels, n_samples);
    for (idx, &sample) in raw_samples.iter().enumerate() {
        let ch = idx % n_channels;
        let t = idx / n_channels;
        data[(ch, t)] = sample as f64 / scale;
    }

    (spec.sample_rate, data)
}

// ─────────────────────────────────────────────────────────────────────────────
// Integration test
// ─────────────────────────────────────────────────────────────────────────────

/// For every WAV file in `tests/data/single-source/`, estimate the DOA of the
/// single source and verify it is within ±5° of the ground-truth angle encoded
/// in the filename.
#[test]
fn test_single_source_doa() {
    let estimator = MusicEstimator::default();

    // Tolerance: ±5 degrees
    let tolerance_deg = 5.0_f64;

    let paths = fs::read_dir("./tests/data/single-source")
        .expect("Test data directory not found")
        .filter(|p| {
            let p = p.as_ref().unwrap().path();
            p.extension().and_then(|ext| ext.to_str()) == Some("wav")
        });

    let mut total = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for entry in paths {
        let path = entry.unwrap().path();
        let parts = parse_test_file_meta_params(&path).unwrap();

        let freq_hz: f64 = parts[1]
            .strip_suffix("Hz")
            .unwrap()
            .parse::<f64>()
            .unwrap();
        let true_doa_deg: f64 = parts[2]
            .strip_suffix("deg")
            .unwrap()
            .parse::<f64>()
            .unwrap();
        let _duration_s: f64 = parts[3]
            .strip_suffix("s")
            .unwrap()
            .parse::<f64>()
            .unwrap();

        // ── Read WAV and build complex signal matrix ──────────────────────────
        let (sample_rate, real_data) = read_wav_4ch(&path);
        let n_mics = real_data.nrows();
        let n_samples = real_data.ncols();

        // Convert each microphone channel to its analytic (complex) signal
        let mut complex_data =
            nalgebra::DMatrix::<num_complex::Complex<f64>>::zeros(n_mics, n_samples);
        for m in 0..n_mics {
            let row: Vec<f64> = real_data.row(m).iter().copied().collect();
            let analytic = analytic_signal(&row);
            for (t, c) in analytic.into_iter().enumerate() {
                complex_data[(m, t)] = c;
            }
        }

        // ── Estimate DOA ──────────────────────────────────────────────────────
        let estimated_angles = estimator.estimate_doa(&complex_data, 1, freq_hz);

        let estimated_deg = if estimated_angles.is_empty() {
            failures.push(format!(
                "{:?}: no peak found (expected {:.1}°)",
                path.file_name().unwrap(),
                true_doa_deg
            ));
            total += 1;
            continue;
        } else {
            estimated_angles[0].to_degrees()
        };

        // ── Check accuracy ────────────────────────────────────────────────────
        // A linear x-axis array has mirror symmetry: a(θ) = a(360°−θ).
        // At exactly Nyquist (f == sr/2), the real steering vector adds two more
        // equivalent angles: a(θ) = a(180°−θ) = a(180°+θ).
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
                "{}: expected {:.1}° (equivalents: {:?}), got {:.1}° (error {:.1}°)",
                path.file_name().unwrap().to_str().unwrap(),
                true_doa_deg,
                equivalent_angles,
                estimated_deg,
                diff_wrapped
            ));
        }
        total += 1;
    }

    if !failures.is_empty() {
        panic!(
            "{}/{} files failed DOA accuracy test (tolerance ±{:.0}°):\n{}",
            failures.len(),
            total,
            tolerance_deg,
            failures.join("\n")
        );
    }

    println!("All {total} single-source DOA tests passed (tolerance ±{tolerance_deg:.0}°).");
}

