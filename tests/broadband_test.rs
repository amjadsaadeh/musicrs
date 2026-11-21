//! Broadband single-source integration tests for IFB-MUSIC.
//!
//! Each test constructs a **composite-tone source** by superimposing three
//! single-frequency recordings from `tests/data/single-source/` that share
//! the same true DOA.  The resulting signal mimics a broadband source emitting
//! three simultaneous sinusoidal tones at a known direction.
//!
//! IFB-MUSIC (`ifb_music_spectrum`) is then run on the real-valued composite
//! data and the estimated DOA is checked against the ground-truth angle.
//!
//! Tolerances are set from the measured worst errors; see `TOL_TARGETED` and
//! `TOL_TRIPLES`.

mod common;

use musicrs::MusicEstimator;
use nalgebra::DMatrix;

// ─── Constants ────────────────────────────────────────────────────────────────

const SAMPLE_RATE: f64 = 16_000.0;
const N_MICS: usize = 4;

/// Default NFFT for IFB-MUSIC.  At 16 kHz this gives a bin width of
/// 16000 / 1024 ≈ 15.6 Hz, well below the lowest component tone (300 Hz).
const NFFT: usize = 1024;

/// Tolerance for the tests on a few fixed tone combinations (degrees). Their estimates
/// land on the true angle (worst error 0.00° on the 0.5° grid); allow one grid step.
const TOL_TARGETED: f64 = 0.5;

/// Tolerance for the 90 low/mid/high triples (degrees). Worst measured error is 2.5°
/// (0°, 1400 + 5800 + 6900 Hz).
const TOL_TRIPLES: f64 = 3.0;

/// All nine available test frequencies (Hz), matching the WAV file corpus.
const ALL_FREQS: [u32; 9] = [300, 1400, 2500, 3600, 4700, 5800, 6900, 8000, 9100];

/// All five ground-truth DOA angles (degrees) in the test corpus.
const ALL_ANGLES_DEG: [u32; 5] = [0, 40, 80, 120, 160];

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn load_composite_source(doa_deg: u32, freqs_hz: &[u32; 3], duration_s: &str) -> DMatrix<f64> {
    load_composite_source_n(doa_deg, freqs_hz, duration_s)
}

/// Load the single-frequency recordings at `doa_deg` and sum them channel-by-channel,
/// trimmed to the shortest recording.
fn load_composite_source_n(doa_deg: u32, freqs_hz: &[u32], duration_s: &str) -> DMatrix<f64> {
    assert!(!freqs_hz.is_empty(), "need at least one frequency");
    let signals: Vec<DMatrix<f64>> = freqs_hz
        .iter()
        .map(|&f| common::read_wav_4ch(&common::wav_path(f, doa_deg, duration_s)).1)
        .collect();

    let n_samples = signals.iter().map(|s| s.ncols()).min().unwrap();
    let mut composite = DMatrix::<f64>::zeros(N_MICS, n_samples);
    for sig in &signals {
        composite += sig.columns(0, n_samples);
    }
    composite
}

/// Angular error allowing for the line array's mirror image `360° − θ`.
fn min_equivalent_error_deg(estimated_deg: f64, true_deg: f64) -> f64 {
    common::angular_error(estimated_deg, &[true_deg, 360.0 - true_deg])
}

/// Find the DOA (in degrees) corresponding to the highest local-maximum peak
/// of an IFB-MUSIC spectrum.  Falls back to the global maximum if no strict
/// local maximum exists (e.g. plateau at the boundary).
fn find_peak_doa_deg(spectrum: &[f64], angles_rad: &[f64]) -> f64 {
    let len = spectrum.len();

    // Collect local maxima (strictly greater than both neighbours, wrapping).
    let mut peaks: Vec<(f64, f64)> = (0..len)
        .filter_map(|i| {
            let prev = spectrum[(i + len - 1) % len];
            let next = spectrum[(i + 1) % len];
            if spectrum[i] > prev && spectrum[i] > next {
                Some((spectrum[i], angles_rad[i].to_degrees()))
            } else {
                None
            }
        })
        .collect();

    if peaks.is_empty() {
        // Fallback: global maximum
        let (idx, _) = spectrum
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        return angles_rad[idx].to_degrees();
    }

    peaks.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    peaks[0].1
}

/// Compute the STFT bin index for a given frequency.
///
/// For a real-valued STFT the single-sided spectrum covers `[0, sr/2]`.
/// Frequencies above the Nyquist (`sr/2`) alias back into the passband at
/// `sr − freq_hz`, so the bin is computed on the aliased frequency instead.
/// The result is clamped to `[1, nfft/2]` (DC and Nyquist are edge cases).
fn freq_to_bin(freq_hz: f64, nfft: usize, sample_rate: f64) -> usize {
    let nyquist = sample_rate / 2.0;
    let effective = if freq_hz > nyquist {
        sample_rate - freq_hz
    } else {
        freq_hz
    };
    let bin = (effective * nfft as f64 / sample_rate).round() as usize;
    bin.clamp(1, nfft / 2)
}

/// Build a sorted, deduplicated list of STFT bin indices covering a
/// neighbourhood of `±half_width` bins around the centre bin of each
/// component frequency.
///
/// Bins are clamped to the valid range `[1, nfft/2]` so that DC and
/// above-Nyquist indices are never included.
fn neighbourhood_bins(
    freqs_hz: &[f64],
    nfft: usize,
    sample_rate: f64,
    half_width: usize,
) -> Vec<usize> {
    let max_bin = nfft / 2;
    let mut bins: Vec<usize> = freqs_hz
        .iter()
        .flat_map(|&f| {
            let centre = freq_to_bin(f, nfft, sample_rate);
            let lo = centre.saturating_sub(half_width).max(1);
            let hi = (centre + half_width).min(max_bin);
            lo..=hi
        })
        .collect();
    bins.sort_unstable();
    bins.dedup();
    bins
}

// ─── Tests ────────────────────────────────────────────────────────────────────

/// IFB-MUSIC estimates the correct DOA for a three-tone broadband source at
/// 40°, using a neighbourhood of ±2 STFT bins around each component frequency.
///
/// This "neighbourhood" approach models the realistic case where the signal
/// frequency bands are known approximately (e.g. from a coarse spectral
/// pre-scan) and a few bins on each side of the peaks are included to guard
/// against STFT frequency leakage.
///
/// Signal: 300 Hz + 1400 Hz + 2500 Hz, DOA = 40°.
#[test]
fn ifb_music_broadband_single_source_40deg_neighbourhood_bins() {
    let estimator = MusicEstimator::default();
    let data = load_composite_source(40, &[300, 1400, 2500], "2.0s");

    let neighbourhood_bins = neighbourhood_bins(&[300.0, 1400.0, 2500.0], NFFT, SAMPLE_RATE, 2);

    let (spectrum, angles) =
        estimator.ifb_music_spectrum(&data, NFFT, 1, Some(&neighbourhood_bins));

    let estimated_deg = find_peak_doa_deg(&spectrum, &angles);
    let err = min_equivalent_error_deg(estimated_deg, 40.0);

    assert!(
        err <= TOL_TARGETED,
        "40° composite (neighbourhood bins): expected ~40° (or 320°), got {estimated_deg:.1}° (error {err:.1}°)"
    );
}

/// IFB-MUSIC estimates the correct DOA for a three-tone broadband source at
/// 40°, using only the three STFT bins closest to the component frequencies.
///
/// Targeted bins should yield at least as good an estimate as using all bins,
/// since irrelevant bins can dilute the geometric-mean combination.
#[test]
fn ifb_music_broadband_single_source_40deg_targeted_bins() {
    let estimator = MusicEstimator::default();
    let data = load_composite_source(40, &[300, 1400, 2500], "2.0s");

    let component_freqs = [300.0_f64, 1400.0, 2500.0];
    let bins: Vec<usize> = component_freqs
        .iter()
        .map(|&f| freq_to_bin(f, NFFT, SAMPLE_RATE))
        .collect();

    let (spectrum, angles) = estimator.ifb_music_spectrum(&data, NFFT, 1, Some(&bins));

    let estimated_deg = find_peak_doa_deg(&spectrum, &angles);
    let err = min_equivalent_error_deg(estimated_deg, 40.0);

    assert!(
        err <= TOL_TARGETED,
        "40° composite (targeted bins {bins:?}): expected ~40°, got {estimated_deg:.1}° (error {err:.1}°)"
    );
}

/// IFB-MUSIC broadband DOA estimation passes for all five ground-truth angles
/// (0°, 40°, 80°, 120°, 160°) using a three-tone composite source and a
/// neighbourhood of ±2 bins around each component frequency.
///
/// Any individual failure is collected and reported together at the end.
#[test]
fn ifb_music_broadband_single_source_all_angles_neighbourhood_bins() {
    let estimator = MusicEstimator::default();
    let freqs: [u32; 3] = [300, 1400, 2500];
    let nbins = neighbourhood_bins(&[300.0, 1400.0, 2500.0], NFFT, SAMPLE_RATE, 2);
    let mut log = common::ErrorLog::new("IFB 300+1400+2500 Hz, neighbourhood bins");

    for doa_deg in ALL_ANGLES_DEG {
        let data = load_composite_source(doa_deg, &freqs, "2.0s");
        let (spectrum, angles) = estimator.ifb_music_spectrum(&data, NFFT, 1, Some(&nbins));
        let estimated_deg = find_peak_doa_deg(&spectrum, &angles);
        log.record(
            format!("{doa_deg}°"),
            min_equivalent_error_deg(estimated_deg, doa_deg as f64),
        );
    }

    log.check(TOL_TARGETED);
}

/// IFB-MUSIC broadband DOA estimation passes for all five ground-truth angles
/// using a three-tone composite source and targeted frequency bins.
#[test]
fn ifb_music_broadband_single_source_all_angles_targeted_bins() {
    let estimator = MusicEstimator::default();
    let freqs: [u32; 3] = [300, 1400, 2500];
    let bins: Vec<usize> = freqs
        .iter()
        .map(|&f| freq_to_bin(f as f64, NFFT, SAMPLE_RATE))
        .collect();
    let mut log = common::ErrorLog::new("IFB 300+1400+2500 Hz, targeted bins");

    for doa_deg in ALL_ANGLES_DEG {
        let data = load_composite_source(doa_deg, &freqs, "2.0s");
        let (spectrum, angles) = estimator.ifb_music_spectrum(&data, NFFT, 1, Some(&bins));
        let estimated_deg = find_peak_doa_deg(&spectrum, &angles);
        log.record(
            format!("{doa_deg}°"),
            min_equivalent_error_deg(estimated_deg, doa_deg as f64),
        );
    }

    log.check(TOL_TARGETED);
}

/// IFB-MUSIC broadband DOA estimation with a mid-to-high frequency composite
/// (1400 + 2500 + 3600 Hz) at 80°.  Verifies that a different spectral
/// composition still resolves the correct direction.
#[test]
fn ifb_music_broadband_mid_high_freqs_80deg() {
    let estimator = MusicEstimator::default();
    let data = load_composite_source(80, &[1400, 2500, 3600], "2.0s");

    let component_freqs = [1400.0_f64, 2500.0, 3600.0];
    let bins: Vec<usize> = component_freqs
        .iter()
        .map(|&f| freq_to_bin(f, NFFT, SAMPLE_RATE))
        .collect();

    let (spectrum, angles) = estimator.ifb_music_spectrum(&data, NFFT, 1, Some(&bins));
    let estimated_deg = find_peak_doa_deg(&spectrum, &angles);
    let err = min_equivalent_error_deg(estimated_deg, 80.0);

    assert!(
        err <= TOL_TARGETED,
        "80° mid-high composite (targeted bins {bins:?}): expected ~80°, got {estimated_deg:.1}° (error {err:.1}°)"
    );
}

/// IFB-MUSIC broadband DOA estimation with a high-frequency composite
/// (3600 + 4700 + 5800 Hz) at 120°.
#[test]
fn ifb_music_broadband_high_freqs_120deg() {
    let estimator = MusicEstimator::default();
    let data = load_composite_source(120, &[3600, 4700, 5800], "2.0s");

    let component_freqs = [3600.0_f64, 4700.0, 5800.0];
    let bins: Vec<usize> = component_freqs
        .iter()
        .map(|&f| freq_to_bin(f, NFFT, SAMPLE_RATE))
        .collect();

    let (spectrum, angles) = estimator.ifb_music_spectrum(&data, NFFT, 1, Some(&bins));
    let estimated_deg = find_peak_doa_deg(&spectrum, &angles);
    let err = min_equivalent_error_deg(estimated_deg, 120.0);

    assert!(
        err <= TOL_TARGETED,
        "120° high-freq composite (targeted bins {bins:?}): expected ~120° (or 240°), got {estimated_deg:.1}° (error {err:.1}°)"
    );
}

// ─── IFB-MUSIC structural / contract tests ────────────────────────────────────

/// `ifb_music_spectrum` output lengths match the estimator's `angle_steps_num`.
#[test]
fn ifb_music_spectrum_output_length_matches_angle_steps() {
    let estimator = MusicEstimator::default(); // 720 angle steps
    let data = load_composite_source(40, &[300, 1400, 2500], "1.0s");

    let (spectrum, angles) = estimator.ifb_music_spectrum(&data, NFFT, 1, None);

    assert_eq!(spectrum.len(), estimator.angle_steps_num);
    assert_eq!(angles.len(), estimator.angle_steps_num);
}

/// `ifb_music_spectrum` angles cover exactly the estimator's `search_interval`.
#[test]
fn ifb_music_spectrum_angles_span_search_interval() {
    use std::f64::consts::PI;
    let estimator = MusicEstimator::default(); // [0, 2π)
    let data = load_composite_source(40, &[300, 1400, 2500], "1.0s");

    let (_, angles) = estimator.ifb_music_spectrum(&data, NFFT, 1, None);

    let (start, end) = estimator.search_interval;
    assert!(
        (angles[0] - start).abs() < 1e-9,
        "First angle should equal search_interval start"
    );
    // Last angle is end − step (exclusive upper bound)
    assert!(
        angles.last().unwrap() < &end,
        "Angles should not exceed search_interval end"
    );
    // All angles are within [start, end)
    for &a in &angles {
        assert!(
            a >= start && a < end,
            "Angle {a:.4} rad outside [{start:.4}, {end:.4})"
        );
    }
    let _ = PI; // suppress unused import warning
}

/// Passing an empty `bins` slice returns a flat unit spectrum (no bins processed).
#[test]
fn ifb_music_spectrum_empty_bins_returns_flat() {
    let estimator = MusicEstimator::default();
    let data = load_composite_source(40, &[300, 1400, 2500], "1.0s");

    let (spectrum, angles) = estimator.ifb_music_spectrum(&data, NFFT, 1, Some(&[]));

    assert_eq!(spectrum.len(), estimator.angle_steps_num);
    assert_eq!(angles.len(), estimator.angle_steps_num);

    // All values should be 1.0 (the flat initialiser value)
    for &v in &spectrum {
        assert!(
            (v - 1.0).abs() < 1e-12,
            "Expected flat spectrum (all 1.0), got {v:.6}"
        );
    }
}

// ─── Broadband integration tests ─────────────────────────────────────────────
//
// These tests mirror the structure of `integration_test.rs` but exercise
// IFB-MUSIC rather than narrowband MUSIC.  The composite source at each angle
// is formed by superimposing *all* available single-frequency recordings at
// that DOA, making the signal genuinely broadband across the entire test
// frequency range (300 Hz – 9.1 kHz).

/// IFB-MUSIC correctly estimates the DOA of a nine-tone broadband single source
/// at every available ground-truth angle.
///
/// For each angle in {0°, 40°, 80°, 120°, 160°} the nine single-frequency
/// recordings at that angle (300 – 9100 Hz) are summed to form a composite
/// signal.  IFB-MUSIC is run with targeted STFT bins at each component
/// frequency.  The estimated DOA must be within `TOL_TARGETED` of the true angle
/// (accounting for the linear-array mirror ambiguity).
///
/// All failures are collected and reported together at the end.
#[test]
fn test_broadband_single_source_doa_nine_tone() {
    let estimator = MusicEstimator::default();
    // 9100 Hz aliases onto the 6900 Hz bin; dedup so that bin is not counted twice.
    let mut bins: Vec<usize> = ALL_FREQS
        .iter()
        .map(|&f| freq_to_bin(f as f64, NFFT, SAMPLE_RATE))
        .collect();
    bins.sort_unstable();
    bins.dedup();
    let mut log = common::ErrorLog::new("IFB nine-tone");

    for doa_deg in ALL_ANGLES_DEG {
        let data = load_composite_source_n(doa_deg, &ALL_FREQS, "2.0s");
        let (spectrum, angles) = estimator.ifb_music_spectrum(&data, NFFT, 1, Some(&bins));
        let estimated_deg = find_peak_doa_deg(&spectrum, &angles);
        log.record(
            format!("{doa_deg}°"),
            min_equivalent_error_deg(estimated_deg, doa_deg as f64),
        );
    }

    log.check(TOL_TARGETED);
}

/// IFB-MUSIC correctly estimates the DOA of composite broadband sources formed
/// from every distinct low / mid / high frequency triple available in the test
/// corpus, across all five ground-truth angles.
///
/// The frequencies are divided into three sub-Nyquist spectral bands (all ≤ 8 kHz;
/// 9100 Hz is excluded because it lies above the 8 kHz Nyquist and its energy
/// appears at the aliased bin 6900 Hz in the single-sided STFT, making the
/// steering-vector frequency ambiguous — that case is covered by the narrowband
/// `test_single_source_doa` test which uses the correct sign-flip convention):
///  - Low : 300, 1400, 2500 Hz
///  - Mid : 3600, 4700, 5800 Hz
///  - High: 6900, 8000 Hz
///
/// Tested triples are all 18 combinations (one frequency from each band), giving
/// 18 × 5 = 90 (angle, triple) pairs.  IFB-MUSIC is run with three targeted
/// STFT bins — one per component.  Tolerance: `TOL_TRIPLES`.
///
/// All failures are collected and reported together at the end.
#[test]
#[ignore = "slow"]
fn test_broadband_single_source_doa_frequency_triples() {
    let estimator = MusicEstimator::default();
    let low_band: [u32; 3] = [300, 1400, 2500];
    let mid_band: [u32; 3] = [3600, 4700, 5800];
    // 9100 Hz is above the 8 kHz Nyquist; limit high band to sub-Nyquist only.
    let high_band: [u32; 2] = [6900, 8000];
    let mut log = common::ErrorLog::new("IFB frequency triples");

    for fl in low_band {
        for fm in mid_band {
            for fh in high_band {
                let triple = [fl, fm, fh];
                let bins: Vec<usize> = triple
                    .iter()
                    .map(|&f| freq_to_bin(f as f64, NFFT, SAMPLE_RATE))
                    .collect();
                for doa_deg in ALL_ANGLES_DEG {
                    let data = load_composite_source(doa_deg, &triple, "2.0s");
                    let (spectrum, angles) =
                        estimator.ifb_music_spectrum(&data, NFFT, 1, Some(&bins));
                    let estimated_deg = find_peak_doa_deg(&spectrum, &angles);
                    log.record(
                        format!("{doa_deg}°, {fl}+{fm}+{fh} Hz"),
                        min_equivalent_error_deg(estimated_deg, doa_deg as f64),
                    );
                }
            }
        }
    }

    log.check(TOL_TRIPLES);
}

/// Spectrum values are all finite and positive (using targeted bins).
#[test]
fn ifb_music_spectrum_values_finite_positive() {
    let estimator = MusicEstimator::default();
    let data = load_composite_source(80, &[1400, 2500, 3600], "1.0s");

    let bins: Vec<usize> = [1400.0_f64, 2500.0, 3600.0]
        .iter()
        .map(|&f| freq_to_bin(f, NFFT, SAMPLE_RATE))
        .collect();
    let (spectrum, _) = estimator.ifb_music_spectrum(&data, NFFT, 1, Some(&bins));

    for (i, &v) in spectrum.iter().enumerate() {
        assert!(v.is_finite(), "spectrum[{i}] = {v} is not finite");
        assert!(v > 0.0, "spectrum[{i}] = {v} is not positive");
    }
}
