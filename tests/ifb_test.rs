// IFB-MUSIC on the single-source WAV data, using the one STFT bin that holds the tone.
//
// nfft = 160 at 16 kHz gives 100 Hz bins, so every test tone falls exactly on a bin.
// Tones at or above Nyquist are excluded: at Nyquist the DOA is ambiguous
// (θ, 180° ± θ), and above it the STFT only sees the aliased frequency.

mod common;

use std::path::Path;

use musicrs::MusicEstimator;
use nalgebra::DMatrix;

const NFFT: usize = 160;

#[test]
fn ifb_single_bin_doa() {
    let est = MusicEstimator::default();
    let tolerance_deg = 5.0_f64;
    let nyquist = est.sample_rate / 2.0;

    let mut total = 0usize;
    let mut failures = Vec::new();

    for path in common::wav_files() {
        let meta = common::parse_meta(&path);
        if meta.frequency_hz >= nyquist {
            continue;
        }
        let (sample_rate, data) = common::read_wav_4ch(&path);
        assert_eq!(sample_rate as f64, est.sample_rate);
        let bin = (meta.frequency_hz * NFFT as f64 / est.sample_rate).round() as usize;

        let (spectrum, angles) = est.ifb_music_spectrum(&data, NFFT, 1, Some(&[bin]));
        let argmax = (0..spectrum.len())
            .max_by(|&a, &b| spectrum[a].total_cmp(&spectrum[b]))
            .unwrap();
        let estimated_deg = angles[argmax].to_degrees();
        let error = common::angular_error(estimated_deg, &[meta.doa_deg, 360.0 - meta.doa_deg]);
        total += 1;
        if error > tolerance_deg {
            failures.push(format!(
                "{}: expected {:.1}°, got {estimated_deg:.1}° (error {error:.1}°)",
                path.file_name().unwrap().to_string_lossy(),
                meta.doa_deg
            ));
        }
    }

    assert!(total > 0, "no test files below Nyquist");
    assert!(
        failures.is_empty(),
        "{}/{total} files failed IFB DOA test (tolerance ±{tolerance_deg:.0}°):\n{}",
        failures.len(),
        failures.join("\n")
    );
    println!("All {total} IFB single-bin DOA tests passed (tolerance ±{tolerance_deg:.0}°).");
}

fn load(stem: &str) -> DMatrix<f64> {
    common::read_wav_4ch(&Path::new(common::DATA_DIR).join(format!("{stem}.wav"))).1
}

fn assert_same_spectrum(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= 1e-9 * y.abs(), "{x} vs {y}");
    }
}

/// Combining all 512 bins must not overflow: the spectrum is a geometric mean, so it stays
/// within the range of the per-bin spectra. Before taking the mean as a sum of logs, the raw
/// product overflowed to `inf` on this recording.
#[test]
fn ifb_all_bins_spectrum_is_finite() {
    let est = MusicEstimator::default();
    let (spectrum, _) =
        est.ifb_music_spectrum(&load("single-source_300Hz_80deg_1.0s"), 1024, 1, None);
    for (i, &v) in spectrum.iter().enumerate() {
        assert!(v.is_finite() && v > 0.0, "spectrum[{i}] = {v}");
    }
}

/// The geometric mean of a bin with itself is that bin's spectrum.
#[test]
fn ifb_repeated_bin_equals_single_bin() {
    let est = MusicEstimator::default();
    let data = load("single-source_1400Hz_40deg_1.0s");
    let (once, _) = est.ifb_music_spectrum(&data, NFFT, 1, Some(&[14]));
    let (thrice, _) = est.ifb_music_spectrum(&data, NFFT, 1, Some(&[14, 14, 14]));
    assert_same_spectrum(&thrice, &once);
}

/// Bins at or above `nfft / 2 + 1` are skipped.
#[test]
fn ifb_ignores_out_of_range_bins() {
    let est = MusicEstimator::default();
    let data = load("single-source_1400Hz_40deg_1.0s");
    let (only, _) = est.ifb_music_spectrum(&data, NFFT, 1, Some(&[14]));
    let (with_extra, _) = est.ifb_music_spectrum(&data, NFFT, 1, Some(&[14, NFFT / 2 + 1, 10_000]));
    assert_same_spectrum(&with_extra, &only);
}

/// Data shorter than one STFT window gives a flat spectrum over the search grid.
#[test]
fn ifb_data_shorter_than_window_is_flat() {
    let est = MusicEstimator::default();
    let (spectrum, angles) = est.ifb_music_spectrum(&DMatrix::zeros(4, NFFT - 1), NFFT, 1, None);
    assert_eq!(angles.len(), est.angle_steps_num);
    assert!(spectrum.iter().all(|&v| v == 1.0));
}
