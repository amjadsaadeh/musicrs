// Every error the public API returns, one test per function.

use musicrs::{MusicError, MusicEstimator, select_bins_by_energy, stft};
use nalgebra::DMatrix;
use num_complex::Complex;
use std::f64::consts::PI;

const NFFT: usize = 160;

/// A 1 kHz tone, the same on all four Kinect microphones.
fn tone(n_samples: usize) -> DMatrix<f64> {
    DMatrix::from_fn(4, n_samples, |m, t| {
        (2.0 * PI * 1000.0 * t as f64 / 16000.0 + m as f64).cos()
    })
}

fn with_nan(mut data: DMatrix<f64>) -> DMatrix<f64> {
    data[(2, 10)] = f64::NAN;
    data
}

const TOO_MANY: MusicError = MusicError::TooManySources {
    n_sources: 4,
    n_antennas: 4,
};

#[test]
fn estimate_doa_from_real_errors() {
    let est = MusicEstimator::default();
    let doa = |data: &DMatrix<f64>, n_sources| est.estimate_doa_from_real(data, n_sources, 1000.0);
    assert_eq!(
        doa(&DMatrix::zeros(3, 100), 1),
        Err(MusicError::RowCountMismatch {
            expected: 4,
            got: 3
        })
    );
    assert_eq!(doa(&DMatrix::zeros(4, 0), 1), Err(MusicError::EmptyData));
    assert_eq!(
        doa(&with_nan(tone(1000)), 1),
        Err(MusicError::NonFiniteData)
    );
    let mut inf = tone(1000);
    inf[(0, 0)] = f64::INFINITY;
    assert_eq!(doa(&inf, 1), Err(MusicError::NonFiniteData));
    assert_eq!(
        doa(&DMatrix::zeros(4, 1000), 1),
        Err(MusicError::SilentData)
    );
    // A constant offset is removed with the mean, leaving silence
    assert_eq!(
        doa(&DMatrix::from_element(4, 1000, 0.3), 1),
        Err(MusicError::SilentData)
    );
    assert_eq!(doa(&tone(1000), 4), Err(TOO_MANY));
}

#[test]
fn estimate_doa_errors() {
    let est = MusicEstimator::default();
    let doa = |data: &DMatrix<Complex<f64>>| est.estimate_doa(data, 1, 1000.0);
    assert_eq!(
        doa(&DMatrix::from_element(3, 100, Complex::new(1.0, 0.5))),
        Err(MusicError::RowCountMismatch {
            expected: 4,
            got: 3
        })
    );
    assert_eq!(doa(&DMatrix::zeros(4, 0)), Err(MusicError::EmptyData));
    let mut nan = DMatrix::from_element(4, 100, Complex::new(1.0, 0.5));
    nan[(1, 1)] = Complex::new(0.0, f64::NAN);
    assert_eq!(doa(&nan), Err(MusicError::NonFiniteData));
    assert_eq!(doa(&DMatrix::zeros(4, 100)), Err(MusicError::SilentData));
}

#[test]
fn noise_subspace_and_spectrum_errors() {
    let est = MusicEstimator::default();
    assert_eq!(
        est.noise_subspace(&DMatrix::identity(3, 3), 1),
        Err(MusicError::RowCountMismatch {
            expected: 4,
            got: 3
        })
    );
    assert_eq!(
        est.noise_subspace(&DMatrix::identity(4, 3), 1),
        Err(MusicError::RowCountMismatch {
            expected: 4,
            got: 3
        })
    );
    let mut nan = DMatrix::identity(4, 4);
    nan[(0, 1)] = Complex::new(f64::NAN, 0.0);
    assert_eq!(est.noise_subspace(&nan, 1), Err(MusicError::NonFiniteData));
    assert_eq!(
        est.noise_subspace(&DMatrix::identity(4, 4), 4),
        Err(TOO_MANY)
    );
    assert_eq!(
        est.spectrum(&DMatrix::identity(3, 2), 1000.0),
        Err(MusicError::RowCountMismatch {
            expected: 4,
            got: 3
        })
    );
}

/// An unusable angle grid is an error wherever the grid is searched. `new` does not check
/// it, because the fields are public and can change after construction.
#[test]
fn invalid_angle_grid_errors() {
    let data = tone(1024);
    let n = MusicEstimator::default().n_antennas();
    for (steps, interval) in [
        (0, (0.0, PI)),
        (360, (PI, 0.0)),
        (360, (1.0, 1.0)),
        (360, (0.0, f64::NAN)),
        (360, (f64::NEG_INFINITY, 0.0)),
    ] {
        let est = MusicEstimator::new(
            MusicEstimator::default().antenna_positions,
            steps,
            interval,
            16000.0,
        );
        let is_grid_err = |e: MusicError| matches!(e, MusicError::InvalidAngleGrid { .. });
        assert!(is_grid_err(
            est.estimate_doa_from_real(&data, 1, 1000.0).unwrap_err()
        ));
        assert!(is_grid_err(
            est.ifb_music_spectrum(&data, NFFT, 1, None).unwrap_err()
        ));
        assert!(is_grid_err(
            est.spectrum(&DMatrix::identity(n, n - 1), 1000.0)
                .unwrap_err()
        ));
    }
}

#[test]
fn ifb_music_spectrum_errors() {
    let est = MusicEstimator::default();
    let data = tone(1600);
    let ifb = |data: &DMatrix<f64>, nfft, n_sources, bins: Option<&[usize]>| {
        est.ifb_music_spectrum(data, nfft, n_sources, bins)
            .map(|_| ())
    };
    assert_eq!(
        ifb(&DMatrix::zeros(3, 1600), NFFT, 1, None),
        Err(MusicError::RowCountMismatch {
            expected: 4,
            got: 3
        })
    );
    assert_eq!(
        ifb(&DMatrix::zeros(0, 1600), NFFT, 1, None),
        Err(MusicError::RowCountMismatch {
            expected: 4,
            got: 0
        })
    );
    assert_eq!(
        ifb(&DMatrix::zeros(4, 0), NFFT, 1, None),
        Err(MusicError::EmptyData)
    );
    assert_eq!(
        ifb(&with_nan(data.clone()), NFFT, 1, None),
        Err(MusicError::NonFiniteData)
    );
    assert_eq!(ifb(&data, NFFT, 4, None), Err(TOO_MANY));
    assert_eq!(ifb(&data, 0, 1, None), Err(MusicError::InvalidNfft));
    assert_eq!(
        ifb(&tone(NFFT - 1), NFFT, 1, None),
        Err(MusicError::DataShorterThanWindow {
            nfft: NFFT,
            samples: NFFT - 1
        })
    );
    // nfft = 1 has only the DC bin, which `None` skips
    assert_eq!(ifb(&data, 1, 1, None), Err(MusicError::NoUsableBins));
    assert_eq!(
        ifb(&data, NFFT, 1, Some(&[])),
        Err(MusicError::NoUsableBins)
    );
    assert_eq!(
        ifb(&data, NFFT, 1, Some(&[NFFT / 2 + 1, 10_000])),
        Err(MusicError::NoUsableBins)
    );
    assert_eq!(
        ifb(&DMatrix::zeros(4, 1600), NFFT, 1, None),
        Err(MusicError::SilentData)
    );
    // The tone is bin 10; bin 30 holds no power
    assert_eq!(
        ifb(&data, NFFT, 1, Some(&[30])),
        Err(MusicError::SilentData)
    );
}

/// Bins without power are skipped rather than mixed into the geometric mean.
#[test]
fn ifb_music_spectrum_skips_silent_bins() {
    let est = MusicEstimator::default();
    let data = tone(1600);
    let (only, _) = est.ifb_music_spectrum(&data, NFFT, 1, Some(&[10])).unwrap();
    let (with_silent, _) = est
        .ifb_music_spectrum(&data, NFFT, 1, Some(&[10, 30, 50]))
        .unwrap();
    assert_eq!(only, with_silent);
}

#[test]
fn stft_errors() {
    assert_eq!(stft(&[0.0; 8], 0, 0), Err(MusicError::InvalidNfft));
    assert_eq!(
        stft(&[0.0; 8], 4, 5),
        Err(MusicError::InvalidOverlap {
            no_overlap: 5,
            nfft: 4
        })
    );
}

#[test]
fn select_bins_by_energy_errors() {
    assert_eq!(
        select_bins_by_energy(&tone(1600), NFFT, -3.0),
        Err(MusicError::InvalidThreshold(-3.0))
    );
    assert_eq!(
        select_bins_by_energy(&DMatrix::zeros(4, 0), NFFT, 20.0),
        Err(MusicError::EmptyData)
    );
}

/// Errors work with `?` and `Box<dyn Error>`, and say what went wrong.
#[test]
fn errors_display_and_convert() {
    fn run() -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        Ok(MusicEstimator::default().estimate_doa_from_real(&DMatrix::zeros(3, 10), 1, 1000.0)?)
    }
    assert_eq!(
        run().unwrap_err().to_string(),
        "expected 4 rows (one per antenna), got 3"
    );
    assert_eq!(
        TOO_MANY.to_string(),
        "4 sources requested, but at most 3 fit 4 antennas"
    );
}
