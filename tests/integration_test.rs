// Narrowband DOA on every single-source WAV recording.
//
// Each WAV file is a 4-channel, 24-bit PCM, 16 kHz recording whose filename
// encodes the source frequency, true DOA, and duration.

mod common;

use musicrs::MusicEstimator;

/// Every recording's estimate lands on its true angle (worst error 0.00° on the 0.5°
/// grid); allow one grid step.
const TOLERANCE_DEG: f64 = 0.5;

#[test]
#[ignore = "slow"]
fn test_single_source_doa() {
    let estimator = MusicEstimator::default();
    let mut log = common::ErrorLog::new("narrowband WAV");

    for path in common::wav_files() {
        let meta = common::parse_meta(&path);
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let (sample_rate, real_data) = common::read_wav_4ch(&path);
        assert_eq!(sample_rate as f64, estimator.sample_rate);

        let estimate = estimator.estimate_doa_from_real(&real_data, 1, meta.frequency_hz);
        let truths =
            common::equivalent_angles(meta.doa_deg, meta.frequency_hz, estimator.sample_rate);
        let error = estimate.first().map_or(f64::INFINITY, |a| {
            common::angular_error(a.to_degrees(), &truths)
        });
        log.record(name, error);
    }

    log.check(TOLERANCE_DEG);
}
