// Every error the public API returns, one test per function.

use musicrs::{
    Direction, MusicError, MusicEstimator, MusicEstimator3d, select_bins_by_energy, stft,
};
use nalgebra::DMatrix;
use num_complex::Complex;
use std::f64::consts::FRAC_PI_2;
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
    assert_eq!(doa(&tone(1000), 0), Err(MusicError::NoSources));
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
    for (rows, cols) in [(3, 3), (4, 3), (3, 4)] {
        assert_eq!(
            est.noise_subspace(&DMatrix::identity(rows, cols), 1),
            Err(MusicError::CovarianceShape {
                n_antennas: 4,
                rows,
                cols
            })
        );
    }
    assert_eq!(
        est.noise_subspace(&DMatrix::identity(4, 3), 1)
            .unwrap_err()
            .to_string(),
        "covariance matrix is 4 × 3, expected 4 × 4 (one row and column per antenna)"
    );
    let mut nan = DMatrix::identity(4, 4);
    nan[(0, 1)] = Complex::new(f64::NAN, 0.0);
    assert_eq!(est.noise_subspace(&nan, 1), Err(MusicError::NonFiniteData));
    assert_eq!(
        est.noise_subspace(&DMatrix::identity(4, 4), 4),
        Err(TOO_MANY)
    );
    assert_eq!(
        est.noise_subspace(&DMatrix::identity(4, 4), 0),
        Err(MusicError::NoSources)
    );
    assert_eq!(
        est.spectrum(&DMatrix::identity(3, 2), 1000.0),
        Err(MusicError::RowCountMismatch {
            expected: 4,
            got: 3
        })
    );
}

/// Constant data is silent once its mean is removed, whatever the constant and length:
/// the rounding residue stays below the silence threshold.
#[test]
fn constant_data_is_silent() {
    let est = MusicEstimator::default();
    for c in [0.3, -0.1, 1.0 / 3.0, PI * 1e5, 1e6 + 0.3, 7e-9, 1e12 / 7.0] {
        for n in [2, 3, 1000, 1601, 16000] {
            let data = DMatrix::from_element(4, n, c);
            assert_eq!(
                est.estimate_doa_from_real(&data, 1, 1000.0),
                Err(MusicError::SilentData),
                "c = {c}, n = {n}"
            );
            if n >= NFFT {
                assert_eq!(
                    est.ifb_music_spectrum(&data, NFFT, 1, None),
                    Err(MusicError::SilentData),
                    "IFB: c = {c}, n = {n}"
                );
            }
        }
    }
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
    assert_eq!(ifb(&data, NFFT, 0, None), Err(MusicError::NoSources));
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

/// The estimator's own fields are checked where they are used: the sample rate and
/// antenna positions by every function that builds steering vectors.
#[test]
fn invalid_estimator_errors() {
    let data = tone(1600);
    let noise = DMatrix::identity(4, 3);
    let check = |est: &MusicEstimator, expected: MusicError| {
        assert_eq!(
            est.estimate_doa_from_real(&data, 1, 1000.0),
            Err(expected.clone())
        );
        let complex = data.map(|x| Complex::new(x, 0.0));
        assert_eq!(est.estimate_doa(&complex, 1, 1000.0), Err(expected.clone()));
        assert_eq!(
            est.ifb_music_spectrum(&data, NFFT, 1, None).map(|_| ()),
            Err(expected.clone())
        );
        assert_eq!(est.spectrum(&noise, 1000.0).map(|_| ()), Err(expected));
    };
    for sr in [0.0, -16000.0, f64::NAN, f64::INFINITY] {
        let est = MusicEstimator {
            sample_rate: sr,
            ..MusicEstimator::default()
        };
        // NaN != NaN, so compare the error's text for it
        if sr.is_nan() {
            assert_eq!(
                est.estimate_doa_from_real(&data, 1, 1000.0)
                    .unwrap_err()
                    .to_string(),
                "sample rate must be finite and positive, got NaN Hz"
            );
        } else {
            check(&est, MusicError::InvalidSampleRate(sr));
        }
    }
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut est = MusicEstimator::default();
        est.antenna_positions[2][1] = bad;
        check(&est, MusicError::InvalidAntennaPosition { index: 2 });
    }
    // A search interval wider than 2π visits directions twice; exactly 2π is fine
    let wide = MusicEstimator {
        search_interval: (0.0, 4.0 * PI),
        angle_steps_num: 1440,
        ..MusicEstimator::default()
    };
    check(
        &wide,
        MusicError::InvalidAngleGrid {
            angle_steps_num: 1440,
            search_interval: (0.0, 4.0 * PI),
        },
    );
    let shifted = MusicEstimator {
        search_interval: (-PI, PI),
        ..MusicEstimator::default()
    };
    assert!(shifted.estimate_doa_from_real(&data, 1, 1000.0).is_ok());
}

/// The signal frequency must be finite and positive, for every function that takes one.
#[test]
fn invalid_frequency_errors() {
    let est = MusicEstimator::default();
    let data = tone(1600);
    let complex = data.map(|x| Complex::new(x, 0.0));
    for freq in [0.0, -1000.0, f64::INFINITY, f64::NEG_INFINITY] {
        let expected = Err(MusicError::InvalidFrequency(freq));
        assert_eq!(est.estimate_doa_from_real(&data, 1, freq), expected);
        assert_eq!(est.estimate_doa(&complex, 1, freq), expected);
        assert_eq!(
            est.spectrum(&DMatrix::identity(4, 3), freq).map(|_| ()),
            expected.map(|_| ())
        );
    }
    assert!(matches!(
        est.estimate_doa_from_real(&data, 1, f64::NAN),
        Err(MusicError::InvalidFrequency(f)) if f.is_nan()
    ));
}

/// IFB never uses the DC bin: it has no direction information.
#[test]
fn ifb_music_spectrum_ignores_dc_bin() {
    let est = MusicEstimator::default();
    let data = tone(1600).map(|x| x + 0.5);
    assert_eq!(
        est.ifb_music_spectrum(&data, NFFT, 1, Some(&[0]))
            .map(|_| ()),
        Err(MusicError::NoUsableBins)
    );
    assert_eq!(
        est.ifb_music_spectrum(&data, NFFT, 1, Some(&[0, 10])),
        est.ifb_music_spectrum(&data, NFFT, 1, Some(&[10]))
    );
}

/// The complex entry point checks the source count and the angle grid itself.
#[test]
fn estimate_doa_parameter_errors() {
    let complex = tone(1000).map(|x| Complex::new(x, 0.0));
    assert_eq!(
        MusicEstimator::default().estimate_doa(&complex, 4, 1000.0),
        Err(TOO_MANY)
    );
    assert_eq!(
        MusicEstimator::default().estimate_doa(&complex, 0, 1000.0),
        Err(MusicError::NoSources)
    );
    let no_grid = MusicEstimator {
        angle_steps_num: 0,
        ..MusicEstimator::default()
    };
    assert!(matches!(
        no_grid.estimate_doa(&complex, 1, 1000.0),
        Err(MusicError::InvalidAngleGrid { .. })
    ));
}

/// An estimator without antennas has room for no source.
#[test]
fn no_antennas() {
    let est = MusicEstimator {
        antenna_positions: vec![],
        ..MusicEstimator::default()
    };
    let none = |n_sources| MusicError::TooManySources {
        n_sources,
        n_antennas: 0,
    };
    let data = DMatrix::zeros(0, 1000);
    assert_eq!(est.estimate_doa_from_real(&data, 0, 1000.0), Err(none(0)));
    assert_eq!(est.estimate_doa_from_real(&data, 1, 1000.0), Err(none(1)));
    assert_eq!(
        est.ifb_music_spectrum(&data, NFFT, 0, None).map(|_| ()),
        Err(none(0))
    );
    assert_eq!(
        none(0).to_string(),
        "0 sources requested, but at most 0 fit 0 antennas"
    );
}

/// A noise subspace that annihilates every steering vector (all zeros, or no columns)
/// scores every angle as infinitely likely, instead of dividing by zero into NaN.
#[test]
fn spectrum_of_zero_noise_subspace_is_infinite() {
    let est = MusicEstimator::default();
    for noise in [DMatrix::zeros(4, 3), DMatrix::zeros(4, 0)] {
        let (spectrum, angles) = est.spectrum(&noise, 1000.0).unwrap();
        assert_eq!(angles.len(), 720);
        assert!(spectrum.iter().all(|&s| s == f64::INFINITY));
    }
}

/// Every error's message names the problem and the offending values.
#[test]
fn every_error_message() {
    let cases = [
        (
            MusicError::RowCountMismatch {
                expected: 4,
                got: 3,
            },
            "expected 4 rows (one per antenna), got 3",
        ),
        (
            MusicError::CovarianceShape {
                n_antennas: 4,
                rows: 4,
                cols: 3,
            },
            "covariance matrix is 4 × 3, expected 4 × 4 (one row and column per antenna)",
        ),
        (MusicError::EmptyData, "data has no samples"),
        (
            MusicError::DataShorterThanWindow {
                nfft: 160,
                samples: 100,
            },
            "data has 100 samples, fewer than one STFT window of 160",
        ),
        (
            MusicError::NonFiniteData,
            "data contains NaN or infinite values",
        ),
        (
            MusicError::SilentData,
            "data is silent: there is no signal to locate",
        ),
        (
            TOO_MANY,
            "4 sources requested, but at most 3 fit 4 antennas",
        ),
        (MusicError::NoSources, "n_sources must be at least 1"),
        (MusicError::InvalidNfft, "nfft must be at least 1"),
        (
            MusicError::InvalidOverlap {
                no_overlap: 5,
                nfft: 4,
            },
            "overlap of 5 samples must be less than nfft = 4",
        ),
        (
            MusicError::InvalidThreshold(-3.0),
            "threshold must be finite and non-negative, got -3 dB",
        ),
        (
            MusicError::NoUsableBins,
            "none of the requested frequency bins is usable (each must be in 1..=nfft/2)",
        ),
        (
            MusicError::InvalidAngleGrid {
                angle_steps_num: 0,
                search_interval: (0.0, 1.5),
            },
            "invalid angle grid: 0 steps over (0, 1.5); need at least one step and a finite \
             interval with start < end spanning at most 2π",
        ),
        (
            MusicError::InvalidElevationGrid {
                elevation_steps: 0,
                elevation_interval: (0.0, 2.0),
            },
            "invalid elevation grid: 0 steps over (0, 2); need at least one step and a \
             finite interval with start < end within [-π/2, π/2]",
        ),
        (
            MusicError::InvalidFrequency(-1000.0),
            "frequency must be finite and positive, got -1000 Hz",
        ),
        (
            MusicError::InvalidSampleRate(0.0),
            "sample rate must be finite and positive, got 0 Hz",
        ),
        (
            MusicError::InvalidAntennaPosition { index: 2 },
            "antenna 2 has a NaN or infinite position",
        ),
    ];
    for (error, message) in cases {
        assert_eq!(error.to_string(), message);
    }
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
    // Checked before the O(nfft) FFT plan: a huge nfft returns at once.
    assert_eq!(
        stft(&[0.0; 4], 1 << 40, 0),
        Err(MusicError::DataShorterThanWindow {
            nfft: 1 << 40,
            samples: 4
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

// ─── MusicEstimator3d ────────────────────────────────────────────────────────

fn tetrahedron() -> MusicEstimator3d {
    let a = 0.03;
    MusicEstimator3d::new(
        vec![[a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]],
        16000.0,
    )
}

#[test]
fn estimator_3d_data_errors() {
    let est = tetrahedron();
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
    assert_eq!(
        doa(&DMatrix::zeros(4, 1000), 1),
        Err(MusicError::SilentData)
    );
    assert_eq!(
        doa(&DMatrix::from_element(4, 1000, 0.3), 1),
        Err(MusicError::SilentData)
    );
    assert_eq!(doa(&tone(1000), 4), Err(TOO_MANY));
    assert_eq!(doa(&tone(1000), 0), Err(MusicError::NoSources));
    assert_eq!(
        est.estimate_doa_from_real(&tone(1000), 1, 0.0),
        Err(MusicError::InvalidFrequency(0.0))
    );

    let complex = |data: &DMatrix<Complex<f64>>| est.estimate_doa(data, 1, 1000.0);
    assert_eq!(
        complex(&DMatrix::from_element(3, 100, Complex::new(1.0, 0.5))),
        Err(MusicError::RowCountMismatch {
            expected: 4,
            got: 3
        })
    );
    assert_eq!(complex(&DMatrix::zeros(4, 0)), Err(MusicError::EmptyData));
    let mut nan = DMatrix::from_element(4, 100, Complex::new(1.0, 0.5));
    nan[(1, 1)] = Complex::new(0.0, f64::NAN);
    assert_eq!(complex(&nan), Err(MusicError::NonFiniteData));
    assert_eq!(
        complex(&DMatrix::zeros(4, 100)),
        Err(MusicError::SilentData)
    );
    assert!(matches!(
        est.estimate_doa(&DMatrix::from_element(4, 100, Complex::new(1.0, 0.5)), 1, f64::NAN),
        Err(MusicError::InvalidFrequency(f)) if f.is_nan()
    ));
}

#[test]
fn estimator_3d_subspace_and_spectrum_errors() {
    let est = tetrahedron();
    assert_eq!(
        est.noise_subspace(&DMatrix::zeros(3, 3), 1),
        Err(MusicError::CovarianceShape {
            n_antennas: 4,
            rows: 3,
            cols: 3
        })
    );
    assert_eq!(
        est.noise_subspace(&DMatrix::identity(4, 4), 4),
        Err(TOO_MANY)
    );
    let mut nan = DMatrix::<Complex<f64>>::identity(4, 4);
    nan[(2, 1)] = Complex::new(f64::NAN, 0.0);
    assert_eq!(est.noise_subspace(&nan, 1), Err(MusicError::NonFiniteData));
    assert_eq!(
        est.spectrum(&DMatrix::zeros(3, 2), 1000.0).map(|_| ()),
        Err(MusicError::RowCountMismatch {
            expected: 4,
            got: 3
        })
    );
    assert_eq!(
        est.spectrum(&DMatrix::zeros(4, 2), -5.0).map(|_| ()),
        Err(MusicError::InvalidFrequency(-5.0))
    );
}

#[test]
fn estimator_3d_array_errors() {
    let data = tone(1000);
    let mut bad_z = tetrahedron();
    bad_z.antenna_positions[2][2] = f64::NAN;
    assert_eq!(
        bad_z.estimate_doa_from_real(&data, 1, 1000.0),
        Err(MusicError::InvalidAntennaPosition { index: 2 })
    );
    let mut bad_x = tetrahedron();
    bad_x.antenna_positions[0][0] = f64::INFINITY;
    assert_eq!(
        bad_x.estimate_doa_from_real(&data, 1, 1000.0),
        Err(MusicError::InvalidAntennaPosition { index: 0 })
    );
    for sr in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let est = MusicEstimator3d {
            sample_rate: sr,
            ..tetrahedron()
        };
        assert!(matches!(
            est.estimate_doa_from_real(&data, 1, 1000.0),
            Err(MusicError::InvalidSampleRate(_))
        ));
    }
}

/// Azimuth intervals spanning more than 2π, elevation intervals reaching past the poles,
/// reversed or non-finite intervals and empty grids are rejected; intervals that end
/// exactly on ±π/2 or span exactly 2π (or a rounding error more) are accepted.
#[test]
fn estimator_3d_angle_overflow() {
    let data = tone(1000);
    let azimuth = |steps: usize, interval: (f64, f64)| {
        MusicEstimator3d {
            azimuth_steps: steps,
            azimuth_interval: interval,
            ..tetrahedron()
        }
        .estimate_doa_from_real(&data, 1, 1000.0)
    };
    let elevation = |steps: usize, interval: (f64, f64)| {
        MusicEstimator3d {
            elevation_steps: steps,
            elevation_interval: interval,
            ..tetrahedron()
        }
        .estimate_doa_from_real(&data, 1, 1000.0)
    };

    for (steps, interval) in [
        (0, (0.0, 2.0 * PI)),
        (360, (0.0, 2.0 * PI + 0.01)),
        (360, (-PI, 3.0 * PI)),
        (360, (1.0, 1.0)),
        (360, (2.0, 1.0)),
        (360, (f64::NAN, 1.0)),
        (360, (0.0, f64::INFINITY)),
        (360, (f64::NEG_INFINITY, 0.0)),
    ] {
        // Compared bit for bit, so that a NaN end matches itself
        let got = azimuth(steps, interval);
        assert!(
            matches!(got, Err(MusicError::InvalidAngleGrid {
                angle_steps_num,
                search_interval: (s, e),
            }) if angle_steps_num == steps
                && s.to_bits() == interval.0.to_bits()
                && e.to_bits() == interval.1.to_bits()),
            "azimuth {steps} over {interval:?}: {got:?}"
        );
    }
    for (steps, interval) in [
        (0, (-FRAC_PI_2, FRAC_PI_2)),
        (180, (-FRAC_PI_2 - 0.01, FRAC_PI_2)),
        (180, (-FRAC_PI_2, FRAC_PI_2 + 0.01)),
        (180, (0.0, PI)),
        (180, (-PI, 0.0)),
        (180, (0.5, 0.5)),
        (180, (FRAC_PI_2, -FRAC_PI_2)),
        (180, (f64::NAN, 0.0)),
        (180, (0.0, f64::INFINITY)),
    ] {
        let got = elevation(steps, interval);
        assert!(
            matches!(got, Err(MusicError::InvalidElevationGrid {
                elevation_steps,
                elevation_interval: (s, e),
            }) if elevation_steps == steps
                && s.to_bits() == interval.0.to_bits()
                && e.to_bits() == interval.1.to_bits()),
            "elevation {steps} over {interval:?}: {got:?}"
        );
    }

    // Exactly the full circle and both poles, and the same off by a rounding error
    assert!(azimuth(360, (-PI, PI)).is_ok());
    assert!(azimuth(360, (0.0, 2.0 * PI + 1e-12)).is_ok());
    assert!(elevation(180, (-FRAC_PI_2, FRAC_PI_2)).is_ok());
    assert!(elevation(180, (-FRAC_PI_2 - 1e-12, FRAC_PI_2 + 1e-12)).is_ok());
}

#[test]
fn estimator_3d_steering_vector() {
    let est = tetrahedron();
    let sv = est.steering_vector(
        Direction {
            azimuth: 0.3,
            elevation: -0.4,
        },
        1000.0,
    );
    assert_eq!(sv.len(), 4);
    assert!(sv.iter().all(|c| (c.norm() - 1.0).abs() < 1e-12));
}
