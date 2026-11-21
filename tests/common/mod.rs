//! Helpers shared by the WAV-based integration tests.

#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::DMatrix;

pub const DATA_DIR: &str = "./tests/data/single-source";

/// Ground truth encoded in a test-data filename.
pub struct Meta {
    pub frequency_hz: f64,
    pub doa_deg: f64,
    pub duration_s: f64,
}

/// Parse `single-source_<freq>Hz_<doa>deg_<duration>s.wav`.
pub fn parse_meta(path: &Path) -> Meta {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("bad file name");
    let parts: Vec<&str> = stem.split('_').collect();
    let num = |s: &str, suffix: &str| -> f64 { s.strip_suffix(suffix).unwrap().parse().unwrap() };
    Meta {
        frequency_hz: num(parts[1], "Hz"),
        doa_deg: num(parts[2], "deg"),
        duration_s: num(parts[3], "s"),
    }
}

/// Path of the recording at `freq_hz` / `doa_deg` / `duration` (e.g. `"2.0s"`).
pub fn wav_path(freq_hz: u32, doa_deg: u32, duration: &str) -> PathBuf {
    Path::new(DATA_DIR).join(format!(
        "single-source_{freq_hz}Hz_{doa_deg}deg_{duration}.wav"
    ))
}

/// All WAV files in the single-source data directory, sorted by name.
pub fn wav_files() -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = fs::read_dir(DATA_DIR)
        .expect("Test data directory not found")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().and_then(|ext| ext.to_str()) == Some("wav"))
        .collect();
    paths.sort();
    paths
}

/// Read a 4-channel, 24-bit PCM WAV file with `hound` and return
/// `(sample_rate, data)` where `data` is a `(4 × n_samples)` real matrix
/// normalised to `[-1, 1]`.
pub fn read_wav_4ch(path: &Path) -> (u32, DMatrix<f64>) {
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
    let scale = 2.0_f64.powi(23); // 24-bit signed → [-1, 1]

    // De-interleave: WAV stores [ch0, ch1, ch2, ch3, ch0, ch1, ...]
    let mut data = DMatrix::<f64>::zeros(n_channels, n_samples);
    for (idx, &sample) in raw_samples.iter().enumerate() {
        data[(idx % n_channels, idx / n_channels)] = sample as f64 / scale;
    }

    (spec.sample_rate, data)
}

/// Smallest circular distance in degrees between `estimate_deg` and any of `truths_deg`.
pub fn angular_error(estimate_deg: f64, truths_deg: &[f64]) -> f64 {
    truths_deg
        .iter()
        .map(|&t| {
            let d = (estimate_deg - t).rem_euclid(360.0);
            d.min(360.0 - d)
        })
        .fold(f64::INFINITY, f64::min)
}

/// Angles indistinguishable from `doa_deg` for the Kinect line array (all mics on the
/// x axis): its mirror image `360° − θ`, plus `180° ± θ` at exactly Nyquist, where the
/// steering vector is real.
pub fn equivalent_angles(doa_deg: f64, frequency_hz: f64, sample_rate: f64) -> Vec<f64> {
    if (frequency_hz - sample_rate / 2.0).abs() < 0.5 {
        vec![doa_deg, 360.0 - doa_deg, 180.0 - doa_deg, 180.0 + doa_deg]
    } else {
        vec![doa_deg, 360.0 - doa_deg]
    }
}

/// Collects one angular error per case and checks them all at the end, so a failing run
/// lists every failing case, and a passing run prints the worst error (useful when tuning
/// tolerances).
pub struct ErrorLog {
    label: String,
    cases: Vec<(String, f64)>,
}

impl ErrorLog {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            cases: Vec::new(),
        }
    }

    pub fn record(&mut self, case: impl Into<String>, error_deg: f64) {
        self.cases.push((case.into(), error_deg));
    }

    /// Panics if there are no cases or any error exceeds `tolerance_deg`.
    pub fn check(&self, tolerance_deg: f64) {
        let label = &self.label;
        let (worst_case, worst) = self
            .cases
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or_else(|| panic!("{label}: no cases ran"));
        println!(
            "{label}: {} cases, worst error {worst:.2}° ({worst_case}), tolerance ±{tolerance_deg}°",
            self.cases.len()
        );
        let failures: Vec<String> = self
            .cases
            .iter()
            .filter(|(_, e)| *e > tolerance_deg)
            .map(|(c, e)| format!("  {c}: error {e:.2}°"))
            .collect();
        assert!(
            failures.is_empty(),
            "{label}: {}/{} cases exceed ±{tolerance_deg}°:\n{}",
            failures.len(),
            self.cases.len(),
            failures.join("\n")
        );
    }
}
