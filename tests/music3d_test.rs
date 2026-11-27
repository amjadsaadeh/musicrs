// 3-D (azimuth + elevation) DOA estimation on synthetic far-field plane waves.
//
// There are no 3-D recordings, so the data is generated from the array geometry with the
// far-field model in `common` (a flat wavefront: microphone p hears the source
// `p · u / c` seconds early, u the unit vector towards the source). It does not use the
// estimator's steering vector, so the tests check the physical model rather than the
// estimator's agreement with itself. The link to the validated 2-D estimator is
// `matches_2d_estimator_on_recordings`: on a planar array with a one-row elevation grid at
// 0°, the 3-D estimator must give exactly the 2-D spectrum.

mod common;

use std::f64::consts::{FRAC_PI_2, PI};

use common::{great_circle_deg, plane_wave_real, plane_waves_complex};
use musicrs::{Direction, MusicEstimator, MusicEstimator3d, select_bins_by_energy};
use nalgebra::DMatrix;

const SAMPLE_RATE: f64 = 16000.0;

/// Regular tetrahedron, 5.2 cm from the centre to each microphone (8.5 cm edges).
fn tetrahedron() -> Vec<[f64; 3]> {
    let a = 0.03;
    vec![[a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]]
}

/// Octahedron: one microphone 5 cm along each of ±x, ±y, ±z.
fn octahedron() -> Vec<[f64; 3]> {
    let r = 0.05;
    vec![
        [r, 0.0, 0.0],
        [-r, 0.0, 0.0],
        [0.0, r, 0.0],
        [0.0, -r, 0.0],
        [0.0, 0.0, r],
        [0.0, 0.0, -r],
    ]
}

/// Square in the x-y plane, 10 cm side: a planar array.
fn square() -> Vec<[f64; 3]> {
    vec![
        [0.05, 0.05, 0.0],
        [-0.05, 0.05, 0.0],
        [-0.05, -0.05, 0.0],
        [0.05, -0.05, 0.0],
    ]
}

fn degrees(d: &Direction) -> (f64, f64) {
    (d.azimuth.to_degrees(), d.elevation.to_degrees())
}

/// Every true direction has an estimate within `tolerance_deg` (great-circle), and there
/// are as many estimates as truths.
fn assert_directions(found: &[Direction], truths: &[(f64, f64)], tolerance_deg: f64) {
    let found_deg: Vec<(f64, f64)> = found.iter().map(degrees).collect();
    assert_eq!(found.len(), truths.len(), "got {found_deg:?}");
    for &truth in truths {
        let error = found_deg
            .iter()
            .map(|&f| great_circle_deg(f, truth))
            .fold(f64::INFINITY, f64::min);
        assert!(
            error <= tolerance_deg,
            "expected {truth:?} among {found_deg:?} (error {error:.2}°)"
        );
    }
}

/// A real tone on a non-planar array is located all over the sphere, including at the
/// zenith and nadir, where every azimuth is the same direction, and next to the azimuth
/// seam at 0° / 360°.
#[test]
fn tetrahedron_single_real_source() {
    let est = MusicEstimator3d::new(tetrahedron(), SAMPLE_RATE);
    let freq = 1000.0;
    let mut log = common::ErrorLog::new("tetrahedron single source");
    for source in [
        (0.0, 0.0),
        (45.0, 30.0),
        (135.0, -60.0),
        (250.0, 10.0),
        (359.7, 45.0),
        (0.3, -20.0),
        (10.0, 85.0),
        (200.0, -85.0),
        (0.0, 90.0),
        (77.0, -90.0),
    ] {
        let data = plane_wave_real(&tetrahedron(), source, freq, SAMPLE_RATE, 4000);
        let found = est.estimate_doa_from_real(&data, 1, freq).unwrap();
        assert_eq!(found.len(), 1, "{source:?}: {found:?}");
        log.record(
            format!("{source:?}"),
            great_circle_deg(degrees(&found[0]), source),
        );
    }
    // 1° grid: the nearest grid point is at most ~0.7° away
    log.check(1.0);
}

/// Two and three uncorrelated sources (with a little noise) on a six-element array.
#[test]
fn octahedron_multiple_sources() {
    let est = MusicEstimator3d::new(octahedron(), SAMPLE_RATE);
    let freq = 1200.0;
    for sources in [
        vec![(80.0, 0.0), (200.0, 45.0)],
        vec![(30.0, 20.0), (150.0, -40.0), (260.0, 60.0)],
    ] {
        let data = plane_waves_complex(&octahedron(), &sources, freq, SAMPLE_RATE, 6000, 0.05, 7);
        let found = est.estimate_doa(&data, sources.len(), freq).unwrap();
        assert_directions(&found, &sources, 1.0);
    }
}

/// A real tone with white noise at about 10 dB SNR.
#[test]
fn tetrahedron_noisy_source() {
    let est = MusicEstimator3d::new(tetrahedron(), SAMPLE_RATE);
    let freq = 1500.0;
    let source = (120.0, 35.0);
    let noise = common::uniform(4 * 8000, 21);
    let data = plane_wave_real(&tetrahedron(), source, freq, SAMPLE_RATE, 8000)
        .map_with_location(|m, t, x| x + 0.55 * (noise[m * 8000 + t] - 0.5) * 2.0);
    let found = est.estimate_doa_from_real(&data, 1, freq).unwrap();
    assert_directions(&found, &[source], 2.0);
}

/// On a planar array with the elevation grid fixed at 0°, the 3-D estimator is the 2-D
/// estimator: the same spectrum, bit for bit, and the same DOAs. A few recordings,
/// including the ones at (8000 Hz) and above (9100 Hz) Nyquist.
#[test]
fn matches_2d_estimator_on_recordings() {
    let files: Vec<_> = [(300, 0), (1400, 40), (4700, 120), (8000, 80), (9100, 160)]
        .iter()
        .map(|&(f, d)| common::wav_path(f, d, "1.0s"))
        .collect();
    check_matches_2d(&files);
}

/// `matches_2d_estimator_on_recordings` on every recording.
#[test]
#[ignore = "slow"]
fn matches_2d_estimator_on_all_recordings() {
    check_matches_2d(&common::wav_files());
}

fn check_matches_2d(files: &[std::path::PathBuf]) {
    let est_2d = MusicEstimator::default();
    let half = 0.25_f64.to_radians();
    let est_3d = MusicEstimator3d {
        antenna_positions: est_2d
            .antenna_positions
            .iter()
            .map(|p| [p[0], p[1], 0.0])
            .collect(),
        azimuth_steps: est_2d.angle_steps_num,
        azimuth_interval: est_2d.search_interval,
        elevation_steps: 1,
        elevation_interval: (-half, half),
        sample_rate: est_2d.sample_rate,
    };
    assert!(!files.is_empty());
    for path in files {
        assert!(path.exists(), "{path:?}");
        let meta = common::parse_meta(path);
        let (_, data) = common::read_wav_4ch(path);
        let freq = meta.frequency_hz;

        let doa_2d = est_2d.estimate_doa_from_real(&data, 1, freq).unwrap();
        let doa_3d = est_3d.estimate_doa_from_real(&data, 1, freq).unwrap();
        assert_eq!(doa_3d.len(), doa_2d.len(), "{path:?}");
        for (d3, d2) in doa_3d.iter().zip(&doa_2d) {
            assert_eq!(d3.elevation, 0.0, "{path:?}");
            assert!((d3.azimuth - d2).abs() < 1e-12, "{path:?}: {d3:?} vs {d2}");
        }

        // The spectra, from the same noise subspace
        let rows: Vec<Vec<num_complex::Complex<f64>>> = (0..4)
            .map(|m| musicrs::analytic_signal(&data.row(m).iter().copied().collect::<Vec<f64>>()))
            .collect();
        let analytic = DMatrix::from_fn(4, data.ncols(), |m, t| rows[m][t]);
        let cov = &analytic * analytic.adjoint();
        let noise = est_2d.noise_subspace(&cov, 1).unwrap();
        let (spec_2d, angles_2d) = est_2d.spectrum(&noise, freq).unwrap();
        let spec_3d = est_3d.spectrum(&noise, freq).unwrap();
        assert_eq!(spec_3d.elevations, vec![0.0]);
        assert_eq!(spec_3d.azimuths, angles_2d, "{path:?}");
        assert_eq!(spec_3d.values.shape(), (1, spec_2d.len()));
        assert_eq!(
            spec_3d.values.row(0).iter().copied().collect::<Vec<f64>>(),
            spec_2d,
            "{path:?}"
        );
    }
}

/// A planar array cannot tell a source above its plane from its mirror image below: the
/// spectrum is symmetric in elevation. Searching only the upper hemisphere gives a single
/// answer.
#[test]
fn planar_array_mirrors_elevation() {
    let freq = 1000.0;
    let source = (60.0, 35.0);
    let data = plane_wave_real(&square(), source, freq, SAMPLE_RATE, 4000);

    let full = MusicEstimator3d::new(square(), SAMPLE_RATE);
    let cov = {
        let c = plane_waves_complex(&square(), &[source], freq, SAMPLE_RATE, 4000, 0.0, 1);
        &c * c.adjoint()
    };
    let spec = full
        .spectrum(&full.noise_subspace(&cov, 1).unwrap(), freq)
        .unwrap();
    let n_el = spec.elevations.len();
    let max = spec.values.max();
    for i in 0..n_el {
        assert!((spec.elevations[i] + spec.elevations[n_el - 1 - i]).abs() < 1e-12);
        for j in 0..spec.azimuths.len() {
            let (a, b) = (spec.values[(i, j)], spec.values[(n_el - 1 - i, j)]);
            assert!((a - b).abs() <= 1e-9 * max, "({i}, {j}): {a} vs {b}");
        }
    }

    let upper = MusicEstimator3d {
        elevation_steps: 90,
        elevation_interval: (0.0, FRAC_PI_2),
        ..MusicEstimator3d::new(square(), SAMPLE_RATE)
    };
    let found = upper.estimate_doa_from_real(&data, 1, freq).unwrap();
    assert_directions(&found, &[source], 1.0);
}

// ─── Angle ranges and overflow ───────────────────────────────────────────────

/// The azimuth interval may start anywhere (here at −180°, 360° and −720°): estimates are
/// reported inside the interval, and a full-circle interval still wraps around.
#[test]
fn shifted_azimuth_intervals() {
    let freq = 1000.0;
    for source in [(350.0, 20.0), (5.0, -30.0), (180.0, 0.0)] {
        let data = plane_wave_real(&tetrahedron(), source, freq, SAMPLE_RATE, 4000);
        for interval in [(-PI, PI), (2.0 * PI, 4.0 * PI), (-4.0 * PI, -2.0 * PI)] {
            let est = MusicEstimator3d {
                azimuth_interval: interval,
                ..MusicEstimator3d::new(tetrahedron(), SAMPLE_RATE)
            };
            let found = est.estimate_doa_from_real(&data, 1, freq).unwrap();
            assert_directions(&found, &[source], 1.0);
            let az = found[0].azimuth;
            assert!(
                interval.0 <= az && az < interval.1,
                "{source:?} in {interval:?}: azimuth {az} outside the interval"
            );
        }
    }
}

/// A source between the last and the first grid azimuth (359.75° on a 0.5° grid over
/// [0°, 360°)) is found across the seam, and reported inside [0, 2π).
#[test]
fn source_on_azimuth_seam() {
    let freq = 1000.0;
    let est = MusicEstimator3d {
        azimuth_steps: 720,
        ..MusicEstimator3d::new(tetrahedron(), SAMPLE_RATE)
    };
    for source in [(359.75, 10.0), (359.9, -45.0), (0.1, 60.0)] {
        let data = plane_wave_real(&tetrahedron(), source, freq, SAMPLE_RATE, 4000);
        let found = est.estimate_doa_from_real(&data, 1, freq).unwrap();
        assert_directions(&found, &[source], 1.0);
        assert!((0.0..2.0 * PI).contains(&found[0].azimuth), "{found:?}");
    }
}

/// The grids stay inside their intervals: azimuths from the start (included) to the end
/// (excluded), elevations at the centres of equal cells, so the poles ±90° are never on
/// the grid and every elevation lies strictly within ±90°.
#[test]
fn grids_stay_inside_intervals() {
    let est = MusicEstimator3d {
        azimuth_steps: 8,
        azimuth_interval: (-PI, PI),
        elevation_steps: 4,
        elevation_interval: (-FRAC_PI_2, FRAC_PI_2),
        ..MusicEstimator3d::new(tetrahedron(), SAMPLE_RATE)
    };
    let cov = {
        let c = plane_waves_complex(
            &tetrahedron(),
            &[(10.0, 10.0)],
            1000.0,
            SAMPLE_RATE,
            500,
            0.01,
            3,
        );
        &c * c.adjoint()
    };
    let spec = est
        .spectrum(&est.noise_subspace(&cov, 1).unwrap(), 1000.0)
        .unwrap();
    let az_expected: Vec<f64> = (0..8).map(|i| -PI + 2.0 * PI * i as f64 / 8.0).collect();
    assert_eq!(spec.azimuths, az_expected);
    let el_deg: Vec<f64> = spec.elevations.iter().map(|e| e.to_degrees()).collect();
    for (got, want) in el_deg.iter().zip([-67.5, -22.5, 22.5, 67.5]) {
        assert!((got - want).abs() < 1e-9, "{el_deg:?}");
    }
    assert_eq!(spec.values.shape(), (4, 8));
}

/// A partial azimuth range is open at both ends: a source at its start is still found
/// there.
#[test]
fn partial_azimuth_range_peaks_at_its_edge() {
    let est = MusicEstimator3d {
        azimuth_steps: 180,
        azimuth_interval: (0.0, PI),
        ..MusicEstimator3d::new(tetrahedron(), SAMPLE_RATE)
    };
    let source = (0.0, 20.0);
    let data = plane_wave_real(&tetrahedron(), source, 1000.0, SAMPLE_RATE, 4000);
    let found = est.estimate_doa_from_real(&data, 1, 1000.0).unwrap();
    assert_directions(&found, &[source], 1.0);
}

/// A one-point grid on an open azimuth range reports its only direction.
#[test]
fn one_point_grid() {
    let est = MusicEstimator3d {
        azimuth_steps: 1,
        azimuth_interval: (0.0, PI),
        elevation_steps: 1,
        elevation_interval: (0.0, 0.2),
        ..MusicEstimator3d::new(tetrahedron(), SAMPLE_RATE)
    };
    let data = plane_wave_real(&tetrahedron(), (40.0, 10.0), 1000.0, SAMPLE_RATE, 1000);
    let found = est.estimate_doa_from_real(&data, 1, 1000.0).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].azimuth, 0.0);
    assert!((found[0].elevation - 0.1).abs() < 1e-15);
}

/// Very loud and very quiet data give the same answer (the data is rescaled first).
#[test]
fn extreme_amplitudes() {
    let est = MusicEstimator3d::new(tetrahedron(), SAMPLE_RATE);
    let source = (100.0, -25.0);
    let data = plane_wave_real(&tetrahedron(), source, 1000.0, SAMPLE_RATE, 4000);
    for amplitude in [1e300, 1e-300] {
        let found = est
            .estimate_doa_from_real(&(&data * amplitude), 1, 1000.0)
            .unwrap();
        assert_directions(&found, &[source], 1.0);
    }
}

// ─── IFB-MUSIC (wideband) ────────────────────────────────────────────────────

/// Sum of real tones at `freqs`, all from `source` (far field), each with its own phase.
fn multi_tone(positions: &[[f64; 3]], source: (f64, f64), freqs: &[f64], n: usize) -> DMatrix<f64> {
    freqs
        .iter()
        .enumerate()
        .map(|(k, &f)| {
            plane_wave_real(positions, source, f, SAMPLE_RATE, n)
                .map(|x| x * (1.0 + 0.3 * k as f64))
        })
        .fold(DMatrix::zeros(positions.len(), n), |acc, x| acc + x)
}

/// IFB-MUSIC on the bin of a single tone (nfft 160: 100 Hz bins, 1000 Hz is bin 10).
#[test]
fn ifb_single_tone_on_its_bin() {
    let est = MusicEstimator3d::new(tetrahedron(), SAMPLE_RATE);
    let mut log = common::ErrorLog::new("IFB 3-D single tone");
    for source in [(70.0, -35.0), (300.0, 10.0), (0.2, 75.0)] {
        let data = plane_wave_real(&tetrahedron(), source, 1000.0, SAMPLE_RATE, 4000);
        let spectrum = est.ifb_music_spectrum(&data, 160, 1, Some(&[10])).unwrap();
        assert_eq!(spectrum.values.shape(), (180, 360));
        let found = est.find_peaks(&spectrum, 1);
        assert_eq!(found.len(), 1);
        log.record(
            format!("{source:?}"),
            great_circle_deg(degrees(&found[0]), source),
        );
    }
    log.check(1.0);
}

/// A multi-tone source, with the bins picked by energy: all spectrum values are finite and
/// positive, and the highest peak is the source.
#[test]
fn ifb_broadband_energy_selected_bins() {
    let est = MusicEstimator3d::new(tetrahedron(), SAMPLE_RATE);
    let source = (210.0, 50.0);
    let data = multi_tone(&tetrahedron(), source, &[600.0, 1100.0, 1700.0], 8000);
    let bins = select_bins_by_energy(&data, 160, 20.0).unwrap();
    assert_eq!(bins, [6, 11, 17]);
    let spectrum = est.ifb_music_spectrum(&data, 160, 1, Some(&bins)).unwrap();
    assert!(spectrum.values.iter().all(|v| v.is_finite() && *v > 0.0));
    assert_directions(&est.find_peaks(&spectrum, 1), &[source], 1.0);
}

/// Two sources at different frequencies and directions, one bin each: both are among the
/// four highest peaks of the merged spectrum (as in the 2-D broadband test).
#[test]
fn ifb_two_sources_at_different_frequencies() {
    let est = MusicEstimator3d::new(octahedron(), SAMPLE_RATE);
    let (a, b) = ((40.0, 20.0), (230.0, -50.0));
    let data = plane_wave_real(&octahedron(), a, 800.0, SAMPLE_RATE, 8000)
        + plane_wave_real(&octahedron(), b, 1500.0, SAMPLE_RATE, 8000);
    let spectrum = est
        .ifb_music_spectrum(&data, 160, 1, Some(&[8, 15]))
        .unwrap();
    let found: Vec<(f64, f64)> = est.find_peaks(&spectrum, 4).iter().map(degrees).collect();
    for truth in [a, b] {
        let error = found
            .iter()
            .map(|&f| great_circle_deg(f, truth))
            .fold(f64::INFINITY, f64::min);
        assert!(
            error <= 1.5,
            "{truth:?} not among {found:?} (error {error:.2}°)"
        );
    }
}

/// On a planar array with one elevation row at 0°, the 3-D IFB spectrum is the 2-D one, bit
/// for bit, with chosen bins and with all bins.
#[test]
fn ifb_matches_2d_estimator_on_recording() {
    let est_2d = MusicEstimator::default();
    let half = 0.25_f64.to_radians();
    let est_3d = MusicEstimator3d {
        antenna_positions: est_2d
            .antenna_positions
            .iter()
            .map(|p| [p[0], p[1], 0.0])
            .collect(),
        azimuth_steps: est_2d.angle_steps_num,
        azimuth_interval: est_2d.search_interval,
        elevation_steps: 1,
        elevation_interval: (-half, half),
        sample_rate: est_2d.sample_rate,
    };
    let (_, data) = common::read_wav_4ch(&common::wav_path(1400, 40, "1.0s"));
    for bins in [Some(&[14][..]), Some(&[13, 14, 15][..]), None] {
        let (spec_2d, angles_2d) = est_2d.ifb_music_spectrum(&data, 160, 1, bins).unwrap();
        let spec_3d = est_3d.ifb_music_spectrum(&data, 160, 1, bins).unwrap();
        assert_eq!(spec_3d.azimuths, angles_2d);
        assert_eq!(spec_3d.elevations, vec![0.0]);
        assert_eq!(
            spec_3d.values.row(0).iter().copied().collect::<Vec<f64>>(),
            spec_2d,
            "bins {bins:?}"
        );
    }
}

/// `find_peaks` on a narrowband spectrum gives what `estimate_doa` gives.
#[test]
fn find_peaks_matches_estimate_doa() {
    let est = MusicEstimator3d::new(octahedron(), SAMPLE_RATE);
    let sources = [(80.0, 0.0), (200.0, 45.0)];
    let data = plane_waves_complex(&octahedron(), &sources, 1200.0, SAMPLE_RATE, 6000, 0.05, 7);
    let cov = &data * data.adjoint();
    let spectrum = est
        .spectrum(&est.noise_subspace(&cov, 2).unwrap(), 1200.0)
        .unwrap();
    assert_eq!(
        est.find_peaks(&spectrum, 2),
        est.estimate_doa(&data, 2, 1200.0).unwrap()
    );
}
