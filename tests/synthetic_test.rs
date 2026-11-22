// DOA estimation on synthetic plane waves with arrays other than the Kinect line.
//
// The signals are built from the array geometry directly (not with
// `MusicEstimator::steering_vector`), so the tests check the physical model rather than
// the estimator's agreement with itself. A 2-D array has no θ / 360° − θ mirror
// ambiguity, so every estimate must match the true angle itself.

mod common;

use std::f64::consts::PI;

use musicrs::{MusicError, MusicEstimator, select_bins_by_energy};
use nalgebra::DMatrix;
use num_complex::Complex;

const SPEED_OF_SOUND: f64 = 343.2;
const SAMPLE_RATE: f64 = 16000.0;

/// Square array, 10 cm side, centred on the origin.
fn square_array() -> MusicEstimator {
    MusicEstimator::new(
        vec![[0.05, 0.05], [-0.05, 0.05], [-0.05, -0.05], [0.05, -0.05]],
        720,
        (0.0, 2.0 * PI),
        SAMPLE_RATE,
    )
}

/// Time by which a plane wave from `doa` reaches `p` ahead of the origin.
fn advance(p: [f64; 2], doa: f64) -> f64 {
    (p[0] * doa.cos() + p[1] * doa.sin()) / SPEED_OF_SOUND
}

/// Deterministic pseudo-random values in [0, 1) (LCG).
fn uniform(n: usize, seed: u64) -> Vec<f64> {
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

fn assert_angles(found: &[f64], truths_deg: &[f64], tolerance_deg: f64) {
    let found_deg: Vec<f64> = found.iter().map(|a| a.to_degrees()).collect();
    assert_eq!(found.len(), truths_deg.len(), "got {found_deg:?}");
    for &truth in truths_deg {
        let error = found_deg
            .iter()
            .map(|&f| common::angular_error(f, &[truth]))
            .fold(f64::INFINITY, f64::min);
        assert!(
            error <= tolerance_deg,
            "expected {truth}° among {found_deg:?} (error {error:.2}°)"
        );
    }
}

/// A real-valued tone through `estimate_doa_from_real` is located at its true angle
/// all around the circle, with no mirror image.
#[test]
fn square_array_single_real_source() {
    let est = square_array();
    let freq = 1000.0;
    for doa_deg in [0.0, 50.0, 135.0, 200.0, 300.0] {
        let doa = f64::to_radians(doa_deg);
        let data = DMatrix::from_fn(4, 4000, |m, t| {
            let p = est.antenna_positions[m];
            (2.0 * PI * freq * (t as f64 / SAMPLE_RATE + advance(p, doa))).cos()
        });
        assert_angles(
            &est.estimate_doa_from_real(&data, 1, freq).unwrap(),
            &[doa_deg],
            1.0,
        );
    }
}

/// Two uncorrelated sources at the same frequency, plus a little noise, are both found.
#[test]
fn square_array_two_sources() {
    let est = square_array();
    let freq = 1500.0;
    let n_samples = 4000;
    let doas_deg = [50.0_f64, 200.0];
    let jitter = uniform(n_samples, 3);
    let noise = uniform(8 * n_samples, 5);
    let data = DMatrix::from_fn(4, n_samples, |m, t| {
        let p = est.antenna_positions[m];
        let t_s = t as f64 / SAMPLE_RATE;
        // Source 2 gets a random phase per sample, which decorrelates it from source 1
        let phases = [0.0, 2.0 * PI * jitter[t]];
        let sources: Complex<f64> = doas_deg
            .iter()
            .zip(phases)
            .map(|(&d, phi)| {
                Complex::from_polar(
                    1.0,
                    2.0 * PI * freq * (t_s + advance(p, d.to_radians())) + phi,
                )
            })
            .sum();
        let i = 2 * (m * n_samples + t);
        sources + Complex::new(noise[i] - 0.5, noise[i + 1] - 0.5).scale(0.05)
    });
    assert_angles(&est.estimate_doa(&data, 2, freq).unwrap(), &doas_deg, 1.0);
}

/// Three sources on a five-element array (the most an n-element array resolves is n − 1).
#[test]
fn pentagon_array_three_sources() {
    let positions: Vec<[f64; 2]> = (0..5)
        .map(|k| {
            let a = 2.0 * PI * k as f64 / 5.0;
            [0.06 * a.cos(), 0.06 * a.sin()]
        })
        .collect();
    let est = MusicEstimator::new(positions, 720, (0.0, 2.0 * PI), SAMPLE_RATE);
    let freq = 1200.0;
    let n_samples = 6000;
    let doas_deg = [30.0_f64, 150.0, 260.0];
    let jitter = uniform(2 * n_samples, 9);
    let noise = uniform(10 * n_samples, 13);
    let data = DMatrix::from_fn(5, n_samples, |m, t| {
        let p = est.antenna_positions[m];
        let t_s = t as f64 / SAMPLE_RATE;
        let phases = [0.0, 2.0 * PI * jitter[t], 2.0 * PI * jitter[n_samples + t]];
        let sources: Complex<f64> = doas_deg
            .iter()
            .zip(phases)
            .map(|(&d, phi)| {
                Complex::from_polar(
                    1.0,
                    2.0 * PI * freq * (t_s + advance(p, d.to_radians())) + phi,
                )
            })
            .sum();
        let i = 2 * (m * n_samples + t);
        sources + Complex::new(noise[i] - 0.5, noise[i + 1] - 0.5).scale(0.05)
    });
    assert_angles(&est.estimate_doa(&data, 3, freq).unwrap(), &doas_deg, 1.0);
}

/// The Kinect line array's spectrum is mirror-symmetric (θ ≡ 360° − θ). A source at
/// endfire that falls between two grid points therefore scores exactly the same on both,
/// and must still be found, at the midpoint between them.
#[test]
fn line_array_endfire_between_grid_points() {
    let half_step = 0.25_f64.to_radians();
    let cases = [
        // 719 steps: 180° lies midway between two grid points
        (
            MusicEstimator {
                angle_steps_num: 719,
                ..MusicEstimator::default()
            },
            180.0,
        ),
        // Grid shifted by half a step: 0° lies midway between -0.25° and 0.25°
        (
            MusicEstimator {
                search_interval: (-half_step, 2.0 * PI - half_step),
                ..MusicEstimator::default()
            },
            0.0,
        ),
    ];
    let freq = 1000.0;
    for (est, doa_deg) in cases {
        let doa = f64::to_radians(doa_deg);
        let data = DMatrix::from_fn(4, 4000, |m, t| {
            let p = est.antenna_positions[m];
            (2.0 * PI * freq * (t as f64 / SAMPLE_RATE + advance(p, doa))).cos()
        });
        assert_angles(
            &est.estimate_doa_from_real(&data, 1, freq).unwrap(),
            &[doa_deg],
            0.01,
        );
    }
}

/// A Kinect-array plane wave at `freq` from `doa_deg`, sampled at the estimator's rate.
fn kinect_tone(est: &MusicEstimator, freq: f64, doa_deg: f64, n_samples: usize) -> DMatrix<f64> {
    let doa = f64::to_radians(doa_deg);
    DMatrix::from_fn(4, n_samples, |m, t| {
        let p = est.antenna_positions[m];
        (2.0 * PI * freq * (t as f64 / est.sample_rate + advance(p, doa))).cos()
    })
}

/// Very loud and very quiet recordings give the same answer: the data is rescaled before
/// its energy and covariance are computed, which would otherwise overflow (or underflow).
#[test]
fn extreme_amplitudes() {
    let est = MusicEstimator {
        search_interval: (0.0, PI),
        angle_steps_num: 360,
        ..MusicEstimator::default()
    };
    let data = kinect_tone(&est, 1000.0, 40.0, 4000);
    let argmax = |s: &[f64]| (0..s.len()).max_by(|&a, &b| s[a].total_cmp(&s[b])).unwrap();
    for amplitude in [1.0, 1e160, 1e300, 1e-160, 1e-300] {
        let scaled = &data * amplitude;
        let found = est.estimate_doa_from_real(&scaled, 1, 1000.0).unwrap();
        assert_eq!(found.len(), 1, "amplitude {amplitude}");
        assert!(
            (found[0].to_degrees() - 40.0).abs() <= 0.5,
            "amplitude {amplitude}: got {found:?}"
        );
        let (spectrum, angles) = est
            .ifb_music_spectrum(&scaled, 160, 1, Some(&[10]))
            .unwrap();
        let peak = angles[argmax(&spectrum)].to_degrees();
        assert!(
            (peak - 40.0).abs() <= 0.5,
            "amplitude {amplitude}: IFB peak {peak}°"
        );
    }
}

/// Zero sources and frequency 0 (every steering vector is the same, so the spectrum is
/// flat) have nothing to find and return no angles. An empty search grid is an error.
#[test]
fn degenerate_configurations_find_nothing() {
    let est = MusicEstimator::default();
    let data = kinect_tone(&est, 1000.0, 40.0, 1000);
    let no_grid = MusicEstimator {
        angle_steps_num: 0,
        ..MusicEstimator::default()
    };
    assert!(matches!(
        no_grid.estimate_doa_from_real(&data, 1, 1000.0),
        Err(MusicError::InvalidAngleGrid { .. })
    ));
    assert_eq!(est.estimate_doa_from_real(&data, 0, 1000.0), Ok(vec![]));
    assert_eq!(est.estimate_doa_from_real(&data, 1, 0.0), Ok(vec![]));
}

/// Very short recordings: two or three samples still give an estimate. A single sample
/// is silent, because removing its mean leaves nothing.
#[test]
fn very_short_data() {
    let est = MusicEstimator::default();
    let doa = |n| est.estimate_doa_from_real(&kinect_tone(&est, 1000.0, 40.0, n), 1, 1000.0);
    assert_eq!(doa(1), Err(MusicError::SilentData));
    for n in 2..=3 {
        assert!(doa(n).is_ok(), "n = {n}: {:?}", doa(n));
    }
}

/// IFB-MUSIC and bin selection at 48 kHz: bin frequencies follow `sample_rate`.
/// nfft = 480 gives 100 Hz bins, so the 1 kHz tone is bin 10.
#[test]
fn ifb_at_48_khz() {
    let est = MusicEstimator {
        sample_rate: 48000.0,
        ..MusicEstimator::default()
    };
    let data = kinect_tone(&est, 1000.0, 40.0, 48000);
    let bins = select_bins_by_energy(&data, 480, 20.0).unwrap();
    assert_eq!(bins, [10]);
    let (spectrum, angles) = est.ifb_music_spectrum(&data, 480, 1, Some(&bins)).unwrap();
    let argmax = (0..spectrum.len())
        .max_by(|&a, &b| spectrum[a].total_cmp(&spectrum[b]))
        .unwrap();
    let error = common::angular_error(angles[argmax].to_degrees(), &[40.0, 320.0]);
    assert!(error <= 0.5, "got {}°", angles[argmax].to_degrees());
}
