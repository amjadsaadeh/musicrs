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

/// With the grid shifted by half a step, a 0° (endfire) source falls between the grid
/// points ±0.25°, which score the same because the line array cannot tell θ from −θ.
/// The estimate must still be 0°, the midpoint, rather than a side lobe.
#[test]
fn endfire_recordings_on_grid_straddling_zero() {
    let half_step = 0.25_f64.to_radians();
    let estimator = MusicEstimator {
        search_interval: (-half_step, 2.0 * std::f64::consts::PI - half_step),
        ..MusicEstimator::default()
    };
    let mut log = common::ErrorLog::new("narrowband WAV at 0°, shifted grid");

    for path in common::wav_files() {
        let meta = common::parse_meta(&path);
        if meta.doa_deg != 0.0 || meta.frequency_hz >= estimator.sample_rate / 2.0 {
            continue;
        }
        let (_, real_data) = common::read_wav_4ch(&path);
        let estimate = estimator.estimate_doa_from_real(&real_data, 1, meta.frequency_hz);
        let error = estimate.first().map_or(f64::INFINITY, |a| {
            common::angular_error(a.to_degrees(), &[0.0])
        });
        log.record(path.file_name().unwrap().to_string_lossy(), error);
    }

    log.check(1e-9);
}
