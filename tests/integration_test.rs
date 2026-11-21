// Integration tests for musicrs
//
// Tests DOA estimation against the pre-recorded single-source WAV test data.
// Each WAV file is a 4-channel, 24-bit PCM, 16 kHz recording whose filename
// encodes the source frequency, true DOA, and duration.

mod common;

use musicrs::MusicEstimator;

/// For every WAV file in `tests/data/single-source/`, estimate the DOA of the
/// single source and verify it is within ±5° of the ground-truth angle encoded
/// in the filename.
#[test]
fn test_single_source_doa() {
    let estimator = MusicEstimator::default();
    let tolerance_deg = 5.0_f64;

    let mut total = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for path in common::wav_files() {
        let meta = common::parse_meta(&path);
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let (sample_rate, real_data) = common::read_wav_4ch(&path);
        assert_eq!(sample_rate as f64, estimator.sample_rate);
        total += 1;

        let estimated_angles = estimator.estimate_doa_from_real(&real_data, 1, meta.frequency_hz);
        let Some(estimate) = estimated_angles.first() else {
            failures.push(format!(
                "{name}: no peak found (expected {:.1}°)",
                meta.doa_deg
            ));
            continue;
        };
        let estimated_deg = estimate.to_degrees();

        // A linear x-axis array has mirror symmetry: a(θ) = a(360°−θ).
        // At exactly Nyquist (f == sr/2), the real steering vector adds two more
        // equivalent angles: a(θ) = a(180°−θ) = a(180°+θ).
        let nyquist = sample_rate as f64 / 2.0;
        let truth = meta.doa_deg;
        let equivalent_angles: Vec<f64> = if (meta.frequency_hz - nyquist).abs() < 0.5 {
            vec![truth, 360.0 - truth, 180.0 - truth, 180.0 + truth]
        } else {
            vec![truth, 360.0 - truth]
        };
        let error = common::angular_error(estimated_deg, &equivalent_angles);

        if error > tolerance_deg {
            failures.push(format!(
                "{name}: expected {truth:.1}° (equivalents: {equivalent_angles:?}), got {estimated_deg:.1}° (error {error:.1}°)"
            ));
        }
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
