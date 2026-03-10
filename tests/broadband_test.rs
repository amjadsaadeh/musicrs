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
//! Tolerance: ±10° (relaxed slightly versus the narrowband ±5° because the
//! geometric-mean combination spans bins that may carry little signal energy
//! when `bins = None`; targeted-bin tests use the tighter ±5° tolerance).

use std::path::PathBuf;

use musicrs::MusicEstimator;
use nalgebra::DMatrix;

// ─── Constants ────────────────────────────────────────────────────────────────

const SAMPLE_RATE: f64 = 16_000.0;
const N_MICS: usize = 4;

/// Default NFFT for IFB-MUSIC.  At 16 kHz this gives a bin width of
/// 16000 / 1024 ≈ 15.6 Hz, well below the lowest component tone (300 Hz).
const NFFT: usize = 1024;

/// Tolerance for all-bin IFB-MUSIC tests (degrees).
const TOL_ALL_BINS: f64 = 10.0;

/// Tolerance for targeted-bin IFB-MUSIC tests (degrees).
const TOL_TARGETED: f64 = 5.0;

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn wav_path(freq_hz: u32, doa_deg: u32, duration_s: &str) -> PathBuf {
    PathBuf::from(format!(
        "./tests/data/single-source/single-source_{}Hz_{}deg_{}.wav",
        freq_hz, doa_deg, duration_s
    ))
}

/// Read a 4-channel, 24-bit PCM WAV file and return a `(4 × n_samples)` matrix
/// with amplitudes normalised to `[−1, 1]`.
fn read_wav_4ch(path: &PathBuf) -> DMatrix<f64> {
    let mut reader =
        hound::WavReader::open(path).unwrap_or_else(|e| panic!("Failed to open {:?}: {}", path, e));
    let spec = reader.spec();
    assert_eq!(spec.channels, 4, "Expected 4-channel WAV");
    assert_eq!(spec.bits_per_sample, 24, "Expected 24-bit PCM");

    let raw: Vec<i32> = reader
        .samples::<i32>()
        .map(|s| s.expect("Sample read error"))
        .collect();
    let n_samples = raw.len() / N_MICS;
    let scale = 2.0_f64.powi(23);

    let mut data = DMatrix::<f64>::zeros(N_MICS, n_samples);
    for (idx, &s) in raw.iter().enumerate() {
        let ch = idx % N_MICS;
        let t = idx / N_MICS;
        data[(ch, t)] = s as f64 / scale;
    }
    data
}

/// Load three single-frequency WAV files at the same `doa_deg` and sum them
/// channel-by-channel, producing a composite broadband recording.
///
/// All files are trimmed to the shortest available recording length before
/// superimposition so that the matrix dimensions are consistent.
fn load_composite_source(doa_deg: u32, freqs_hz: &[u32; 3], duration_s: &str) -> DMatrix<f64> {
    load_composite_source_n(doa_deg, freqs_hz, duration_s)
}

/// Load an arbitrary number of single-frequency WAV files at the same `doa_deg`
/// and sum them channel-by-channel.  All recordings are trimmed to the shortest
/// length before superimposition.
fn load_composite_source_n(doa_deg: u32, freqs_hz: &[u32], duration_s: &str) -> DMatrix<f64> {
    assert!(!freqs_hz.is_empty(), "need at least one frequency");
    let signals: Vec<DMatrix<f64>> = freqs_hz
        .iter()
        .map(|&f| read_wav_4ch(&wav_path(f, doa_deg, duration_s)))
        .collect();

    let n_samples = signals.iter().map(|s| s.ncols()).min().unwrap();
    let mut composite = DMatrix::<f64>::zeros(N_MICS, n_samples);
    for sig in &signals {
        composite += sig.columns(0, n_samples);
    }
    composite
}

/// Compute the nearest equivalent angular error (in degrees) taking into
/// account the mirror-symmetry ambiguity of a linear microphone array
/// (`a(θ) = a(360° − θ)`).
fn min_equivalent_error_deg(estimated_deg: f64, true_deg: f64) -> f64 {
    [true_deg, 360.0 - true_deg]
        .iter()
        .map(|&t| {
            let d = (estimated_deg - t).abs();
            if d > 180.0 { 360.0 - d } else { d }
        })
        .fold(f64::INFINITY, f64::min)
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
        estimator.ifb_music_spectrum(&data, NFFT, 1, SAMPLE_RATE, Some(&neighbourhood_bins));

    let estimated_deg = find_peak_doa_deg(&spectrum, &angles);
    let err = min_equivalent_error_deg(estimated_deg, 40.0);

    assert!(
        err <= TOL_ALL_BINS,
        "40° composite (neighbourhood bins): expected ~40° (or 320°), got {:.1}° (error {:.1}°)",
        estimated_deg,
        err
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

    let (spectrum, angles) = estimator.ifb_music_spectrum(&data, NFFT, 1, SAMPLE_RATE, Some(&bins));

    let estimated_deg = find_peak_doa_deg(&spectrum, &angles);
    let err = min_equivalent_error_deg(estimated_deg, 40.0);

    assert!(
        err <= TOL_TARGETED,
        "40° composite (targeted bins {:?}): expected ~40°, got {:.1}° (error {:.1}°)",
        bins,
        estimated_deg,
        err
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
    let angles_under_test = [0u32, 40, 80, 120, 160];
    let nbins = neighbourhood_bins(&[300.0, 1400.0, 2500.0], NFFT, SAMPLE_RATE, 2);

    let mut failures: Vec<String> = Vec::new();

    for &doa_deg in &angles_under_test {
        let data = load_composite_source(doa_deg, &freqs, "2.0s");
        let (spectrum, angles) =
            estimator.ifb_music_spectrum(&data, NFFT, 1, SAMPLE_RATE, Some(&nbins));
        let estimated_deg = find_peak_doa_deg(&spectrum, &angles);
        let err = min_equivalent_error_deg(estimated_deg, doa_deg as f64);

        if err > TOL_ALL_BINS {
            failures.push(format!(
                "DOA {}° (composite 300+1400+2500 Hz, neighbourhood bins): got {:.1}° (error {:.1}°)",
                doa_deg, estimated_deg, err
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "{}/{} angles failed IFB-MUSIC neighbourhood-bin broadband test (tolerance ±{:.0}°):\n{}",
            failures.len(),
            angles_under_test.len(),
            TOL_ALL_BINS,
            failures.join("\n")
        );
    }
}

/// IFB-MUSIC broadband DOA estimation passes for all five ground-truth angles
/// using a three-tone composite source and targeted frequency bins.
#[test]
fn ifb_music_broadband_single_source_all_angles_targeted_bins() {
    let estimator = MusicEstimator::default();
    let freqs: [u32; 3] = [300, 1400, 2500];
    let angles_under_test = [0u32, 40, 80, 120, 160];

    let bins: Vec<usize> = [300.0_f64, 1400.0, 2500.0]
        .iter()
        .map(|&f| freq_to_bin(f, NFFT, SAMPLE_RATE))
        .collect();

    let mut failures: Vec<String> = Vec::new();

    for &doa_deg in &angles_under_test {
        let data = load_composite_source(doa_deg, &freqs, "2.0s");
        let (spectrum, angles) =
            estimator.ifb_music_spectrum(&data, NFFT, 1, SAMPLE_RATE, Some(&bins));
        let estimated_deg = find_peak_doa_deg(&spectrum, &angles);
        let err = min_equivalent_error_deg(estimated_deg, doa_deg as f64);

        if err > TOL_TARGETED {
            failures.push(format!(
                "DOA {}° (composite 300+1400+2500 Hz, targeted bins {:?}): got {:.1}° (error {:.1}°)",
                doa_deg, bins, estimated_deg, err
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "{}/{} angles failed IFB-MUSIC targeted-bin broadband test (tolerance ±{:.0}°):\n{}",
            failures.len(),
            angles_under_test.len(),
            TOL_TARGETED,
            failures.join("\n")
        );
    }
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

    let (spectrum, angles) = estimator.ifb_music_spectrum(&data, NFFT, 1, SAMPLE_RATE, Some(&bins));
    let estimated_deg = find_peak_doa_deg(&spectrum, &angles);
    let err = min_equivalent_error_deg(estimated_deg, 80.0);

    assert!(
        err <= TOL_TARGETED,
        "80° mid-high composite (targeted bins {:?}): expected ~80°, got {:.1}° (error {:.1}°)",
        bins,
        estimated_deg,
        err
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

    let (spectrum, angles) = estimator.ifb_music_spectrum(&data, NFFT, 1, SAMPLE_RATE, Some(&bins));
    let estimated_deg = find_peak_doa_deg(&spectrum, &angles);
    let err = min_equivalent_error_deg(estimated_deg, 120.0);

    assert!(
        err <= TOL_TARGETED,
        "120° high-freq composite (targeted bins {:?}): expected ~120° (or 240°), got {:.1}° (error {:.1}°)",
        bins,
        estimated_deg,
        err
    );
}

// ─── IFB-MUSIC structural / contract tests ────────────────────────────────────

/// `ifb_music_spectrum` output lengths match the estimator's `angle_steps_num`.
#[test]
fn ifb_music_spectrum_output_length_matches_angle_steps() {
    let estimator = MusicEstimator::default(); // 720 angle steps
    let data = load_composite_source(40, &[300, 1400, 2500], "1.0s");

    let (spectrum, angles) = estimator.ifb_music_spectrum(&data, NFFT, 1, SAMPLE_RATE, None);

    assert_eq!(spectrum.len(), estimator.angle_steps_num);
    assert_eq!(angles.len(), estimator.angle_steps_num);
}

/// `ifb_music_spectrum` angles cover exactly the estimator's `search_interval`.
#[test]
fn ifb_music_spectrum_angles_span_search_interval() {
    use std::f64::consts::PI;
    let estimator = MusicEstimator::default(); // [0, 2π)
    let data = load_composite_source(40, &[300, 1400, 2500], "1.0s");

    let (_, angles) = estimator.ifb_music_spectrum(&data, NFFT, 1, SAMPLE_RATE, None);

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
            "Angle {:.4} rad outside [{:.4}, {:.4})",
            a,
            start,
            end
        );
    }
    let _ = PI; // suppress unused import warning
}

/// Passing an empty `bins` slice returns a flat unit spectrum (no bins processed).
#[test]
fn ifb_music_spectrum_empty_bins_returns_flat() {
    let estimator = MusicEstimator::default();
    let data = load_composite_source(40, &[300, 1400, 2500], "1.0s");

    let (spectrum, angles) = estimator.ifb_music_spectrum(&data, NFFT, 1, SAMPLE_RATE, Some(&[]));

    assert_eq!(spectrum.len(), estimator.angle_steps_num);
    assert_eq!(angles.len(), estimator.angle_steps_num);

    // All values should be 1.0 (the flat initialiser value)
    for &v in &spectrum {
        assert!(
            (v - 1.0).abs() < 1e-12,
            "Expected flat spectrum (all 1.0), got {:.6}",
            v
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

/// All nine available test frequencies (Hz), matching the WAV file corpus.
const ALL_FREQS: [u32; 9] = [300, 1400, 2500, 3600, 4700, 5800, 6900, 8000, 9100];

/// All five ground-truth DOA angles (degrees) in the test corpus.
const ALL_ANGLES_DEG: [u32; 5] = [0, 40, 80, 120, 160];

/// IFB-MUSIC correctly estimates the DOA of a nine-tone broadband single source
/// at every available ground-truth angle.
///
/// For each angle in {0°, 40°, 80°, 120°, 160°} the nine single-frequency
/// recordings at that angle (300 – 9100 Hz) are summed to form a composite
/// signal.  IFB-MUSIC is run with targeted STFT bins at each component
/// frequency.  The estimated DOA must be within ±5° of the true angle
/// (accounting for the linear-array mirror ambiguity).
///
/// All failures are collected and reported together at the end.
#[test]
fn test_broadband_single_source_doa_nine_tone() {
    let estimator = MusicEstimator::default();
    let tolerance_deg = TOL_TARGETED;

    let bins: Vec<usize> = ALL_FREQS
        .iter()
        .map(|&f| freq_to_bin(f as f64, NFFT, SAMPLE_RATE))
        .collect();

    let mut failures: Vec<String> = Vec::new();

    for &doa_deg in &ALL_ANGLES_DEG {
        let data = load_composite_source_n(doa_deg, &ALL_FREQS, "2.0s");

        let (spectrum, angles) =
            estimator.ifb_music_spectrum(&data, NFFT, 1, SAMPLE_RATE, Some(&bins));
        let estimated_deg = find_peak_doa_deg(&spectrum, &angles);
        let err = min_equivalent_error_deg(estimated_deg, doa_deg as f64);

        if err > tolerance_deg {
            failures.push(format!(
                "DOA {}°: expected {:.1}° (or {:.1}°), got {:.1}° (error {:.1}°)",
                doa_deg,
                doa_deg as f64,
                360.0 - doa_deg as f64,
                estimated_deg,
                err
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "{}/{} angles failed nine-tone IFB-MUSIC test (tolerance ±{:.0}°):\n{}",
            failures.len(),
            ALL_ANGLES_DEG.len(),
            tolerance_deg,
            failures.join("\n")
        );
    }

    println!(
        "All {} nine-tone broadband DOA tests passed (tolerance ±{:.0}°).",
        ALL_ANGLES_DEG.len(),
        tolerance_deg
    );
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
/// STFT bins — one per component.  Tolerance: ±5°.
///
/// All failures are collected and reported together at the end.
#[test]
fn test_broadband_single_source_doa_frequency_triples() {
    let estimator = MusicEstimator::default();
    let tolerance_deg = TOL_TARGETED;

    let low_band: [u32; 3] = [300, 1400, 2500];
    let mid_band: [u32; 3] = [3600, 4700, 5800];
    // 9100 Hz is above the 8 kHz Nyquist; limit high band to sub-Nyquist only.
    let high_band: [u32; 2] = [6900, 8000];

    let mut failures: Vec<String> = Vec::new();
    let mut total = 0usize;

    for &fl in &low_band {
        for &fm in &mid_band {
            for &fh in &high_band {
                let triple = [fl, fm, fh];
                let bins: Vec<usize> = triple
                    .iter()
                    .map(|&f| freq_to_bin(f as f64, NFFT, SAMPLE_RATE))
                    .collect();

                for &doa_deg in &ALL_ANGLES_DEG {
                    let data = load_composite_source(doa_deg, &triple, "2.0s");

                    let (spectrum, angles) =
                        estimator.ifb_music_spectrum(&data, NFFT, 1, SAMPLE_RATE, Some(&bins));
                    let estimated_deg = find_peak_doa_deg(&spectrum, &angles);
                    let err = min_equivalent_error_deg(estimated_deg, doa_deg as f64);

                    if err > tolerance_deg {
                        failures.push(format!(
                            "DOA {}°, triple {}+{}+{} Hz (bins {:?}): got {:.1}° (error {:.1}°)",
                            doa_deg, fl, fm, fh, bins, estimated_deg, err
                        ));
                    }
                    total += 1;
                }
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "{}/{} (angle, triple) pairs failed IFB-MUSIC frequency-triple test \
             (tolerance ±{:.0}°):\n{}",
            failures.len(),
            total,
            tolerance_deg,
            failures.join("\n")
        );
    }

    println!(
        "All {total} (angle, triple) broadband DOA tests passed (tolerance ±{tolerance_deg:.0}°)."
    );
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
    let (spectrum, _) = estimator.ifb_music_spectrum(&data, NFFT, 1, SAMPLE_RATE, Some(&bins));

    for (i, &v) in spectrum.iter().enumerate() {
        assert!(v.is_finite(), "spectrum[{}] = {} is not finite", i, v);
        assert!(v > 0.0, "spectrum[{}] = {} is not positive", i, v);
    }
}
