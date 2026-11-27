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

// ─── Far-field (plane-wave) model for 3-D arrays ─────────────────────────────

pub const SPEED_OF_SOUND: f64 = 343.2;

/// Unit vector pointing towards the source at `azimuth` (from +x towards +y) and
/// `elevation` (from the x-y plane towards +z), both in degrees.
pub fn unit_direction(azimuth_deg: f64, elevation_deg: f64) -> [f64; 3] {
    let (az, el) = (azimuth_deg.to_radians(), elevation_deg.to_radians());
    [el.cos() * az.cos(), el.cos() * az.sin(), el.sin()]
}

/// Time by which a plane wave from `(azimuth_deg, elevation_deg)` reaches `p` ahead of
/// the origin (far field: the wavefront is flat across the array).
pub fn advance_3d(p: [f64; 3], azimuth_deg: f64, elevation_deg: f64) -> f64 {
    let u = unit_direction(azimuth_deg, elevation_deg);
    (p[0] * u[0] + p[1] * u[1] + p[2] * u[2]) / SPEED_OF_SOUND
}

/// Great-circle distance in degrees between two directions given as
/// `(azimuth_deg, elevation_deg)`.
pub fn great_circle_deg(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (u, v) = (unit_direction(a.0, a.1), unit_direction(b.0, b.1));
    let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    dot.clamp(-1.0, 1.0).acos().to_degrees()
}

/// Deterministic pseudo-random values in [0, 1) (LCG).
pub fn uniform(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        })
        .collect()
}

/// A real-valued tone at `freq` from one far-field source, sampled at `sample_rate`.
pub fn plane_wave_real(
    positions: &[[f64; 3]],
    source_deg: (f64, f64),
    freq: f64,
    sample_rate: f64,
    n_samples: usize,
) -> DMatrix<f64> {
    DMatrix::from_fn(positions.len(), n_samples, |m, t| {
        let tau = advance_3d(positions[m], source_deg.0, source_deg.1);
        (2.0 * std::f64::consts::PI * freq * (t as f64 / sample_rate + tau)).cos()
    })
}

/// Complex (analytic) data of several uncorrelated far-field sources at `freq`, plus
/// complex white noise of amplitude `noise`. Every source but the first gets an
/// independent random phase per sample, which decorrelates it from the others.
pub fn plane_waves_complex(
    positions: &[[f64; 3]],
    sources_deg: &[(f64, f64)],
    freq: f64,
    sample_rate: f64,
    n_samples: usize,
    noise: f64,
    seed: u64,
) -> DMatrix<num_complex::Complex<f64>> {
    use num_complex::Complex;
    use std::f64::consts::PI;
    let jitter = uniform(sources_deg.len() * n_samples, seed);
    let white = uniform(2 * positions.len() * n_samples, seed + 1);
    DMatrix::from_fn(positions.len(), n_samples, |m, t| {
        let t_s = t as f64 / sample_rate;
        let signal: Complex<f64> = sources_deg
            .iter()
            .enumerate()
            .map(|(k, &(az, el))| {
                let phi = if k == 0 {
                    0.0
                } else {
                    2.0 * PI * jitter[k * n_samples + t]
                };
                let tau = advance_3d(positions[m], az, el);
                Complex::from_polar(1.0, 2.0 * PI * freq * (t_s + tau) + phi)
            })
            .sum();
        let i = 2 * (m * n_samples + t);
        signal + Complex::new(white[i] - 0.5, white[i + 1] - 0.5).scale(noise)
    })
}
