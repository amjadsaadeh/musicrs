// Parity with the original pyMUSIC implementation.
//
// The CSVs in tests/golden/ were produced by running pyMUSIC on the WAV test data
// (see tests/golden/generate_golden.py), in two variants:
//   * `*.eigh.csv` – pyMUSIC with `np.linalg.eig` swapped for the Hermitian `eigh`.
//     musicrs must match these value-by-value.
//   * `*.eig.csv`  – pyMUSIC as published. On this nearly noiseless data `eig`
//     returns a non-orthogonal noise basis, so the values are basis-dependent; only
//     the peak location is compared.
//   * `*.hilbert_n<N>.scipy.csv` – `scipy.signal.hilbert` of one channel, compared
//     with `analytic_signal` value by value.
//
// At and above Nyquist (8000 Hz and 9100 Hz here) musicrs deliberately differs from
// pyMUSIC (real steering vector at f == sr/2, conjugated one above it; see CLAUDE.md
// "Sign convention"), so there is no pyMUSIC reference. Instead
// `*.narrowband.musicrs.csv` are snapshots of musicrs's OWN output, not pyMUSIC
// references: they pin that behaviour so it cannot change silently. Regenerate them
// (after an intended change only) with
//     MUSICRS_UPDATE_SNAPSHOTS=1 cargo test --test golden_test

mod common;

use std::f64::consts::PI;
use std::fs;
use std::path::Path;

use musicrs::{MusicEstimator, analytic_signal};
use nalgebra::DMatrix;

const GOLDEN_DIR: &str = "./tests/golden";
const NFFT: usize = 160;
// Noise eigenvalues are ~1e-15 and pyMUSIC builds steering vectors in float128,
// so agreement is ~1e-7..1e-6 rather than machine precision.
const REL_TOLERANCE: f64 = 1e-4;

const CASES: [(&str, f64); 15] = [
    ("single-source_300Hz_80deg_1.0s", 300.0),
    ("single-source_1400Hz_40deg_1.0s", 1400.0),
    ("single-source_4700Hz_120deg_1.0s", 4700.0),
    ("single-source_2500Hz_0deg_2.0s", 2500.0),
    ("single-source_5800Hz_160deg_2.0s", 5800.0),
    ("single-source_6900Hz_120deg_1.0s", 6900.0),
    // All five angles at 3600 Hz and 6900 Hz, alternating 1 s / 2 s recordings
    ("single-source_3600Hz_0deg_1.0s", 3600.0),
    ("single-source_3600Hz_40deg_2.0s", 3600.0),
    ("single-source_3600Hz_80deg_1.0s", 3600.0),
    ("single-source_3600Hz_120deg_2.0s", 3600.0),
    ("single-source_3600Hz_160deg_1.0s", 3600.0),
    ("single-source_6900Hz_0deg_2.0s", 6900.0),
    ("single-source_6900Hz_40deg_1.0s", 6900.0),
    ("single-source_6900Hz_80deg_2.0s", 6900.0),
    ("single-source_6900Hz_160deg_1.0s", 6900.0),
];

/// Multi-bin IFB cases: `(frequencies in Hz, DOA in degrees, duration)`; the recordings
/// at the same DOA are summed and IFB uses one bin per tone.
const COMPOSITE_CASES: [(&[u32], u32, &str); 2] = [
    (&[300, 1400, 2500], 40, "2.0s"),
    (&[3600, 4700, 5800], 120, "1.0s"),
];

fn read_golden(name: &str) -> (Vec<f64>, Vec<f64>) {
    let text = fs::read_to_string(Path::new(GOLDEN_DIR).join(name))
        .unwrap_or_else(|e| panic!("cannot read golden file {name}: {e}"));
    text.lines()
        .skip(1)
        .map(|line| {
            let (a, s) = line.split_once(',').unwrap();
            (a.parse::<f64>().unwrap(), s.parse::<f64>().unwrap())
        })
        .unzip()
}

fn argmax(v: &[f64]) -> usize {
    (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap()
}

/// Compare a musicrs spectrum against both golden variants of `prefix`.
fn check(prefix: &str, angles: &[f64], spectrum: &[f64]) {
    // Value-by-value against pyMUSIC with a Hermitian eigensolver
    let (g_angles, g_spectrum) = read_golden(&format!("{prefix}.eigh.csv"));
    assert_eq!(angles.len(), g_angles.len(), "{prefix}: grid size differs");
    for (a, g) in angles.iter().zip(&g_angles) {
        assert!(
            (a - g).abs() < 1e-9,
            "{prefix}: angle grid differs ({a} vs {g})"
        );
    }
    let err = spectrum
        .iter()
        .zip(&g_spectrum)
        .map(|(s, g)| (s - g).abs() / g.abs())
        .fold(0.0, f64::max);
    println!("{prefix}: max relative error vs eigh {err:.3e}");
    assert!(
        err < REL_TOLERANCE,
        "{prefix}: deviates from pyMUSIC (eigh) by {err:.3e}"
    );

    // Peak location against pyMUSIC as published (within one grid step, circularly). The
    // line array's spectrum has equal peaks at θ and 360° − θ (grid index n − i), and
    // rounding decides which one is the argmax, so either counts.
    let (_, orig) = read_golden(&format!("{prefix}.eig.csv"));
    let (ours, theirs) = (argmax(spectrum), argmax(&orig));
    let n = spectrum.len();
    let circular_steps = |a: usize, b: usize| {
        let d = (a + n - b) % n;
        d.min(n - d)
    };
    let steps = circular_steps(ours, theirs).min(circular_steps(ours, (n - theirs) % n));
    println!(
        "{prefix}: peak {:.1}° vs pyMUSIC (eig) {:.1}°",
        angles[ours].to_degrees(),
        angles[theirs].to_degrees()
    );
    assert!(
        steps <= 1,
        "{prefix}: peak differs from pyMUSIC (eig) by {steps} grid steps"
    );
}

/// Narrowband spectrum of recording `stem` at `freq`, computed the way pyMUSIC's
/// `getSourceAngles` does (analytic signal, covariance, one source). Returns
/// `(spectrum, angles)`.
fn narrowband_spectrum(est: &MusicEstimator, stem: &str, freq: f64) -> (Vec<f64>, Vec<f64>) {
    let (_, data) = common::read_wav_4ch(&Path::new(common::DATA_DIR).join(format!("{stem}.wav")));
    narrowband_spectrum_of(est, &data, freq, 1)
}

/// Narrowband spectrum of `data` with `n_sources` sources, without the mean removal
/// `estimate_doa_from_real` does (pyMUSIC does not remove it).
fn narrowband_spectrum_of(
    est: &MusicEstimator,
    data: &DMatrix<f64>,
    freq: f64,
    n_sources: usize,
) -> (Vec<f64>, Vec<f64>) {
    let analytic: Vec<Vec<_>> = (0..data.nrows())
        .map(|m| analytic_signal(&data.row(m).iter().copied().collect::<Vec<_>>()))
        .collect();
    let x = DMatrix::from_fn(data.nrows(), data.ncols(), |m, t| analytic[m][t]);
    let cov = (&x * x.adjoint()).scale(1.0 / data.ncols() as f64);
    let noise = est.noise_subspace(&cov, n_sources).unwrap();
    est.spectrum(&noise, freq).unwrap()
}

#[test]
fn narrowband_spectrum_matches_pymusic() {
    let est = MusicEstimator::default();
    for (stem, freq) in CASES {
        let (spectrum, angles) = narrowband_spectrum(&est, stem, freq);
        check(&format!("{stem}.narrowband"), &angles, &spectrum);
    }
}

#[test]
fn ifb_spectrum_matches_pymusic() {
    let est = MusicEstimator::default();
    for (stem, freq) in CASES {
        let (_, data) =
            common::read_wav_4ch(&Path::new(common::DATA_DIR).join(format!("{stem}.wav")));
        let bin = (freq * NFFT as f64 / est.sample_rate).round() as usize;
        let (spectrum, angles) = est
            .ifb_music_spectrum(&data, NFFT, 1, Some(&[bin]))
            .unwrap();
        check(
            &format!("{stem}.ifb_nfft{NFFT}_bin{bin}"),
            &angles,
            &spectrum,
        );
    }
}

/// Multi-bin IFB (geometric mean over one bin per tone) on sums of recordings at one DOA.
#[test]
fn ifb_multi_bin_spectrum_matches_pymusic() {
    let est = MusicEstimator::default();
    for (freqs, doa, duration) in COMPOSITE_CASES {
        let parts: Vec<DMatrix<f64>> = freqs
            .iter()
            .map(|&f| common::read_wav_4ch(&common::wav_path(f, doa, duration)).1)
            .collect();
        let n = parts.iter().map(|p| p.ncols()).min().unwrap();
        let data = parts
            .iter()
            .fold(DMatrix::zeros(4, n), |acc, p| acc + p.columns(0, n));
        let bins: Vec<usize> = freqs
            .iter()
            .map(|&f| f as usize * NFFT / est.sample_rate as usize)
            .collect();
        let (spectrum, angles) = est.ifb_music_spectrum(&data, NFFT, 1, Some(&bins)).unwrap();
        let tones: Vec<String> = freqs.iter().map(u32::to_string).collect();
        let bin_names: Vec<String> = bins.iter().map(usize::to_string).collect();
        check(
            &format!(
                "composite_{}Hz_{doa}deg_{duration}.ifb_nfft{NFFT}_bins{}",
                tones.join("+"),
                bin_names.join("-")
            ),
            &angles,
            &spectrum,
        );
    }
}

/// Narrowband on half the circle (360 steps, so still a 0.5° grid). On `(π, 2π)` the
/// 120° source shows up as its mirror image, 240°.
#[test]
fn narrowband_partial_interval_matches_pymusic() {
    let cases = [
        (
            "single-source_1400Hz_40deg_1.0s",
            1400.0,
            "0-180deg",
            (0.0, PI),
        ),
        (
            "single-source_4700Hz_120deg_1.0s",
            4700.0,
            "180-360deg",
            (PI, 2.0 * PI),
        ),
    ];
    for (stem, freq, interval_name, interval) in cases {
        let est = MusicEstimator {
            angle_steps_num: 360,
            search_interval: interval,
            ..MusicEstimator::default()
        };
        let (spectrum, angles) = narrowband_spectrum(&est, stem, freq);
        check(
            &format!("{stem}.narrowband_{interval_name}"),
            &angles,
            &spectrum,
        );
    }
}

/// IFB at STFT sizes other than 160. At nfft 1024 the tone falls between the two bins.
#[test]
fn ifb_other_nfft_matches_pymusic() {
    let cases: [(&str, usize, &[usize]); 4] = [
        ("single-source_1400Hz_40deg_1.0s", 320, &[28]),
        ("single-source_4700Hz_120deg_1.0s", 320, &[94]),
        ("single-source_1400Hz_40deg_1.0s", 1024, &[89, 90]),
        ("single-source_4700Hz_120deg_1.0s", 1024, &[300, 301]),
    ];
    let est = MusicEstimator::default();
    for (stem, nfft, bins) in cases {
        let (_, data) =
            common::read_wav_4ch(&Path::new(common::DATA_DIR).join(format!("{stem}.wav")));
        let (spectrum, angles) = est.ifb_music_spectrum(&data, nfft, 1, Some(bins)).unwrap();
        let bin_names: Vec<String> = bins.iter().map(usize::to_string).collect();
        check(
            &format!("{stem}.ifb_nfft{nfft}_bins{}", bin_names.join("-")),
            &angles,
            &spectrum,
        );
    }
}

/// Narrowband with two sources: 1400 Hz from 40° plus 2500 Hz from 120°, analysed at
/// 1400 Hz. The covariance has rank 2, so the two-dimensional noise subspace is well
/// defined (two same-frequency recordings would be coherent, i.e. rank 1).
#[test]
fn narrowband_two_sources_matches_pymusic() {
    let est = MusicEstimator::default();
    let a = common::read_wav_4ch(&common::wav_path(1400, 40, "1.0s")).1;
    let b = common::read_wav_4ch(&common::wav_path(2500, 120, "1.0s")).1;
    let n = a.ncols().min(b.ncols());
    let data = a.columns(0, n) + b.columns(0, n);
    let (spectrum, angles) = narrowband_spectrum_of(&est, &data, 1400.0, 2);
    check(
        "composite_1400Hz_40deg+2500Hz_120deg_1.0s.narrowband_1400Hz_2src",
        &angles,
        &spectrum,
    );
}

/// IFB with two sources at different angles: 1400 Hz from 40° plus 2500 Hz from 120°, one
/// bin per tone. Each bin holds one source, so IFB runs with one source per bin.
#[test]
fn ifb_two_sources_matches_pymusic() {
    let est = MusicEstimator::default();
    let a = common::read_wav_4ch(&common::wav_path(1400, 40, "1.0s")).1;
    let b = common::read_wav_4ch(&common::wav_path(2500, 120, "1.0s")).1;
    let n = a.ncols().min(b.ncols());
    let data = a.columns(0, n) + b.columns(0, n);
    let (spectrum, angles) = est
        .ifb_music_spectrum(&data, NFFT, 1, Some(&[14, 25]))
        .unwrap();
    check(
        "composite_1400Hz_40deg+2500Hz_120deg_1.0s.ifb_nfft160_bins14-25",
        &angles,
        &spectrum,
    );
}

/// `analytic_signal` against `scipy.signal.hilbert`, value by value, for an even and an
/// odd length (odd lengths have no Nyquist bin).
#[test]
fn analytic_signal_matches_scipy() {
    let (_, data) = common::read_wav_4ch(&common::wav_path(1400, 40, "1.0s"));
    for n in [2048, 2047] {
        let name = format!("single-source_1400Hz_40deg_1.0s.ch0.hilbert_n{n}.scipy.csv");
        let (re, im) = read_golden(&name);
        assert_eq!(re.len(), n, "{name}: length");
        let input: Vec<f64> = data.row(0).iter().take(n).copied().collect();
        let z = analytic_signal(&input);
        let err = z
            .iter()
            .zip(re.iter().zip(&im))
            .map(|(c, (r, i))| (c.re - r).abs().max((c.im - i).abs()))
            .fold(0.0, f64::max);
        println!("{name}: max abs error vs scipy {err:.3e}");
        assert!(err < 1e-12, "{name}: deviates from scipy by {err:.3e}");
    }
}

// ── musicrs snapshots at / above Nyquist (not pyMUSIC references) ────────────────

/// Recordings at exactly Nyquist (real steering vector) and above it (conjugated).
const SNAPSHOT_CASES: [(&str, f64); 4] = [
    ("single-source_8000Hz_40deg_1.0s", 8000.0),
    ("single-source_8000Hz_120deg_2.0s", 8000.0),
    ("single-source_9100Hz_80deg_1.0s", 9100.0),
    ("single-source_9100Hz_160deg_2.0s", 9100.0),
];
/// Snapshots compare musicrs with itself, so only floating-point noise across
/// platforms/toolchains should remain.
const SNAPSHOT_REL_TOLERANCE: f64 = 1e-9;

#[test]
fn narrowband_spectrum_matches_musicrs_snapshot_at_and_above_nyquist() {
    let update = std::env::var("MUSICRS_UPDATE_SNAPSHOTS").is_ok_and(|v| v == "1");
    let est = MusicEstimator::default();
    for (stem, freq) in SNAPSHOT_CASES {
        let (spectrum, angles) = narrowband_spectrum(&est, stem, freq);
        let name = format!("{stem}.narrowband.musicrs.csv");
        if update {
            let mut csv = String::from("angle_rad,spectrum\n");
            for (a, s) in angles.iter().zip(&spectrum) {
                csv.push_str(&format!("{a:?},{s:?}\n"));
            }
            fs::write(Path::new(GOLDEN_DIR).join(&name), csv).unwrap();
            println!("{name}: snapshot rewritten");
            continue;
        }
        let (s_angles, s_spectrum) = read_golden(&name);
        assert_eq!(angles, s_angles, "{name}: angle grid differs");
        let err = spectrum
            .iter()
            .zip(&s_spectrum)
            .map(|(s, g)| (s - g).abs() / g.abs())
            .fold(0.0, f64::max);
        println!("{name}: max relative error vs snapshot {err:.3e}");
        assert!(
            err < SNAPSHOT_REL_TOLERANCE,
            "{name}: musicrs output changed by {err:.3e} relative to its snapshot; if \
             intended, regenerate with MUSICRS_UPDATE_SNAPSHOTS=1 cargo test --test golden_test"
        );
    }
}
