// DOA estimation on synthetic plane waves with arrays other than the Kinect line.
//
// The signals are built from the array geometry directly (not with
// `MusicEstimator::steering_vector`), so the tests check the physical model rather than
// the estimator's agreement with itself. A 2-D array has no θ / 360° − θ mirror
// ambiguity, so every estimate must match the true angle itself.

mod common;

use std::f64::consts::PI;

use musicrs::MusicEstimator;
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
        assert_angles(&est.estimate_doa_from_real(&data, 1, freq), &[doa_deg], 1.0);
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
    assert_angles(&est.estimate_doa(&data, 2, freq), &doas_deg, 1.0);
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
    assert_angles(&est.estimate_doa(&data, 3, freq), &doas_deg, 1.0);
}

#[test]
#[should_panic(expected = "data must have one row per antenna")]
fn estimate_doa_from_real_rejects_wrong_row_count() {
    square_array().estimate_doa_from_real(&DMatrix::zeros(3, 100), 1, 1000.0);
}
