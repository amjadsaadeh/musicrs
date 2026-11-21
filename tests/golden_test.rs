// Parity with the original pyMUSIC implementation.
//
// The CSVs in tests/golden/ were produced by running pyMUSIC on the WAV test data
// (see tests/golden/generate_golden.py), in two variants:
//   * `*.eigh.csv` – pyMUSIC with `np.linalg.eig` swapped for the Hermitian `eigh`.
//     musicrs must match these value-by-value.
//   * `*.eig.csv`  – pyMUSIC as published. On this nearly noiseless data `eig`
//     returns a non-orthogonal noise basis, so the values are basis-dependent; only
//     the peak location is compared.

mod common;

use std::fs;
use std::path::Path;

use musicrs::{MusicEstimator, analytic_signal};
use nalgebra::DMatrix;

const GOLDEN_DIR: &str = "./tests/golden";
const NFFT: usize = 160;
// Noise eigenvalues are ~1e-15 and pyMUSIC builds steering vectors in float128,
// so agreement is ~1e-7..1e-6 rather than machine precision.
const REL_TOLERANCE: f64 = 1e-4;

const CASES: [(&str, f64); 6] = [
    ("single-source_300Hz_80deg_1.0s", 300.0),
    ("single-source_1400Hz_40deg_1.0s", 1400.0),
    ("single-source_4700Hz_120deg_1.0s", 4700.0),
    ("single-source_2500Hz_0deg_2.0s", 2500.0),
    ("single-source_5800Hz_160deg_2.0s", 5800.0),
    ("single-source_6900Hz_120deg_1.0s", 6900.0),
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

#[test]
fn narrowband_spectrum_matches_pymusic() {
    let est = MusicEstimator::default();
    for (stem, freq) in CASES {
        let (_, data) =
            common::read_wav_4ch(&Path::new(common::DATA_DIR).join(format!("{stem}.wav")));
        let analytic: Vec<Vec<_>> = (0..data.nrows())
            .map(|m| analytic_signal(&data.row(m).iter().copied().collect::<Vec<_>>()))
            .collect();
        let x = DMatrix::from_fn(data.nrows(), data.ncols(), |m, t| analytic[m][t]);
        let cov = (&x * x.adjoint()).scale(1.0 / data.ncols() as f64);
        let noise = est.noise_subspace(&cov, 1);
        let (spectrum, angles) = est.spectrum(&noise, freq);
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
        let (spectrum, angles) = est.ifb_music_spectrum(&data, NFFT, 1, Some(&[bin]));
        check(
            &format!("{stem}.ifb_nfft{NFFT}_bin{bin}"),
            &angles,
            &spectrum,
        );
    }
}

/// Multi-bin IFB (geometric mean over one bin per tone) on the sum of three recordings at 40°.
#[test]
fn ifb_multi_bin_spectrum_matches_pymusic() {
    let est = MusicEstimator::default();
    let freqs = [300, 1400, 2500];
    let parts: Vec<DMatrix<f64>> = freqs
        .iter()
        .map(|&f| common::read_wav_4ch(&common::wav_path(f, 40, "2.0s")).1)
        .collect();
    let n = parts.iter().map(|p| p.ncols()).min().unwrap();
    let data = parts
        .iter()
        .fold(DMatrix::zeros(4, n), |acc, p| acc + p.columns(0, n));
    let bins: Vec<usize> = freqs
        .iter()
        .map(|&f| f as usize * NFFT / est.sample_rate as usize)
        .collect();
    let (spectrum, angles) = est.ifb_music_spectrum(&data, NFFT, 1, Some(&bins));
    check(
        "composite_300+1400+2500Hz_40deg_2.0s.ifb_nfft160_bins3-14-25",
        &angles,
        &spectrum,
    );
}
