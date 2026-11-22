// IFB-MUSIC on the single-source WAV data, using the one STFT bin that holds the tone.
//
// nfft = 160 at 16 kHz gives 100 Hz bins, so every test tone falls exactly on a bin.
// Tones at or above Nyquist are excluded: at Nyquist the DOA is ambiguous
// (θ, 180° ± θ), and above it the STFT only sees the aliased frequency.

mod common;

use std::path::Path;

use musicrs::{MusicEstimator, select_bins_by_energy};
use nalgebra::DMatrix;

const NFFT: usize = 160;

/// Every recording's estimate lands on its true angle (worst error 0.00° on the 0.5°
/// grid); allow one grid step.
const TOLERANCE_DEG: f64 = 0.5;

#[test]
fn ifb_single_bin_doa() {
    let est = MusicEstimator::default();
    let nyquist = est.sample_rate / 2.0;
    let mut log = common::ErrorLog::new("IFB single bin");

    for path in common::wav_files() {
        let meta = common::parse_meta(&path);
        if meta.frequency_hz >= nyquist {
            continue;
        }
        let (sample_rate, data) = common::read_wav_4ch(&path);
        assert_eq!(sample_rate as f64, est.sample_rate);
        let bin = (meta.frequency_hz * NFFT as f64 / est.sample_rate).round() as usize;

        let (spectrum, angles) = est
            .ifb_music_spectrum(&data, NFFT, 1, Some(&[bin]))
            .unwrap();
        let argmax = (0..spectrum.len())
            .max_by(|&a, &b| spectrum[a].total_cmp(&spectrum[b]))
            .unwrap();
        let error = common::angular_error(
            angles[argmax].to_degrees(),
            &[meta.doa_deg, 360.0 - meta.doa_deg],
        );
        log.record(path.file_name().unwrap().to_string_lossy(), error);
    }

    log.check(TOLERANCE_DEG);
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
    let (spectrum, _) = est
        .ifb_music_spectrum(&load("single-source_300Hz_80deg_1.0s"), 1024, 1, None)
        .unwrap();
    for (i, &v) in spectrum.iter().enumerate() {
        assert!(v.is_finite() && v > 0.0, "spectrum[{i}] = {v}");
    }
}

/// The geometric mean of a bin with itself is that bin's spectrum.
#[test]
fn ifb_repeated_bin_equals_single_bin() {
    let est = MusicEstimator::default();
    let data = load("single-source_1400Hz_40deg_1.0s");
    let (once, _) = est.ifb_music_spectrum(&data, NFFT, 1, Some(&[14])).unwrap();
    let (thrice, _) = est
        .ifb_music_spectrum(&data, NFFT, 1, Some(&[14, 14, 14]))
        .unwrap();
    assert_same_spectrum(&thrice, &once);
}

/// Repeated bins are counted once and the order of the bins does not matter: on two tones,
/// each carrying signal on its own bin, listing them as [b, a, a, b, b] must not weight
/// bin b more than bin a.
#[test]
fn ifb_duplicate_and_reordered_bins_are_ignored() {
    let est = MusicEstimator::default();
    let low = load("single-source_1400Hz_40deg_1.0s");
    let high = load("single-source_3600Hz_120deg_1.0s");
    let n = low.ncols().min(high.ncols());
    let data = low.columns(0, n) + high.columns(0, n);
    let (plain, _) = est
        .ifb_music_spectrum(&data, NFFT, 1, Some(&[14, 36]))
        .unwrap();
    let (repeated, _) = est
        .ifb_music_spectrum(&data, NFFT, 1, Some(&[36, 14, 14, 36, 36]))
        .unwrap();
    assert_eq!(repeated, plain);
}

/// Bins at or above `nfft / 2 + 1` are skipped.
#[test]
fn ifb_ignores_out_of_range_bins() {
    let est = MusicEstimator::default();
    let data = load("single-source_1400Hz_40deg_1.0s");
    let (only, _) = est.ifb_music_spectrum(&data, NFFT, 1, Some(&[14])).unwrap();
    let (with_extra, _) = est
        .ifb_music_spectrum(&data, NFFT, 1, Some(&[14, NFFT / 2 + 1, 10_000]))
        .unwrap();
    assert_same_spectrum(&with_extra, &only);
}

/// With bins picked by energy (within 20 dB of the strongest), IFB-MUSIC finds every
/// recording below Nyquist. At nfft = 160 every tone falls on a bin, so selection picks
/// that bin and the estimate is exact; with all bins (`None`) noise-only bins dominate and
/// estimates are off by up to ~130°.
#[test]
fn ifb_energy_selected_bins_doa() {
    let est = MusicEstimator::default();
    let mut log = common::ErrorLog::new("IFB energy-selected bins, nfft 160");

    for path in common::wav_files() {
        let meta = common::parse_meta(&path);
        if meta.frequency_hz >= est.sample_rate / 2.0 {
            continue;
        }
        let (_, data) = common::read_wav_4ch(&path);
        let bins = select_bins_by_energy(&data, NFFT, 20.0).unwrap();
        let (spectrum, angles) = est.ifb_music_spectrum(&data, NFFT, 1, Some(&bins)).unwrap();
        let argmax = (0..spectrum.len())
            .max_by(|&a, &b| spectrum[a].total_cmp(&spectrum[b]))
            .unwrap();
        let error = common::angular_error(
            angles[argmax].to_degrees(),
            &[meta.doa_deg, 360.0 - meta.doa_deg],
        );
        log.record(
            format!(
                "{} (bins {bins:?})",
                path.file_name().unwrap().to_string_lossy()
            ),
            error,
        );
    }

    log.check(TOLERANCE_DEG);
}
