// IFB-MUSIC on the single-source WAV data, using the one STFT bin that holds the tone.
//
// nfft = 160 at 16 kHz gives 100 Hz bins, so every test tone falls exactly on a bin.
// Tones at or above Nyquist are excluded: at Nyquist the DOA is ambiguous
// (θ, 180° ± θ), and above it the STFT only sees the aliased frequency.

mod common;

use musicrs::MusicEstimator;

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
