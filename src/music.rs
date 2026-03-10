//! MUSIC (MUltiple SIgnal Classification) algorithm for Direction-of-Arrival estimation.

use std::f64::consts::PI;

use nalgebra::DMatrix;
use nalgebra::DVector;
use num_complex::Complex;

use crate::stft::stft;

/// Speed of sound in air at ~20 °C, in metres per second.
const SPEED_OF_SOUND: f64 = 343.2;

/// Default Kinect microphone array positions `(x, y)` in metres.
const DEFAULT_ANTENNA_POSITIONS: [[f64; 2]; 4] =
    [[0.113, 0.0], [-0.036, 0.0], [-0.076, 0.0], [-0.113, 0.0]];

// ─────────────────────────────────────────────────────────────────────────────
// Type aliases
// ─────────────────────────────────────────────────────────────────────────────
type Cplx = Complex<f64>;
type CMatrix = DMatrix<Cplx>;
type CVec = DVector<Cplx>;

// ─────────────────────────────────────────────────────────────────────────────
// Hermitian eigendecomposition (pure nalgebra, no LAPACK)
// ─────────────────────────────────────────────────────────────────────────────

/// Eigendecomposition of a complex Hermitian n×n matrix using faer's dedicated
/// complex selfadjoint solver, which is numerically more stable than the real
/// block-matrix trick for near-degenerate noise subspaces.
///
/// Returns `(eigenvalues, eigenvectors)` sorted by ascending eigenvalue.
/// `eigenvalues[k]` is real; `eigenvectors.column(k)` is the unit-norm eigenvector.
fn hermitian_eigen(mat: &CMatrix) -> (Vec<f64>, CMatrix) {
    use faer::complex_native::c64;
    use faer::{Mat as FaerMat, Side};

    let n = mat.nrows();

    // Convert nalgebra complex matrix → faer complex matrix
    let faer_mat: FaerMat<c64> = FaerMat::from_fn(n, n, |i, j| {
        let c = mat[(i, j)];
        c64::new(c.re, c.im)
    });

    // Numerically stable Hermitian eigendecomposition
    // Note: faer does not guarantee a sort order, so we sort ascending afterward.
    let evd = faer_mat.selfadjoint_eigendecomposition(Side::Lower);

    // Eigenvalues are real but stored as c64 with im ≈ 0; extract via .re
    let s_col = evd.s().column_vector();
    let u_mat = evd.u();

    // Sort by ascending (real) eigenvalue
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| s_col.read(a).re.partial_cmp(&s_col.read(b).re).unwrap());

    let eigenvalues: Vec<f64> = order.iter().map(|&i| s_col.read(i).re).collect();
    // faer's selfadjoint_eigendecomposition returns eigenvectors with conjugated
    // imaginary parts relative to the signal-processing convention.  Negate im to
    // recover the eigenvectors in the standard A·v = λ·v convention expected by MUSIC.
    let eigenvectors: CMatrix = CMatrix::from_fn(n, n, |row, col| {
        let src_col = order[col];
        let c = u_mat.read(row, src_col);
        Cplx::new(c.re, c.im)
    });

    (eigenvalues, eigenvectors)
}

// ─────────────────────────────────────────────────────────────────────────────
// Peak detection
// ─────────────────────────────────────────────────────────────────────────────

/// Returns the angles corresponding to the `n` largest local maxima in `scores`,
/// sorted by descending peak height.
fn find_peaks(scores: &[f64], angles: &[f64], n: usize) -> Vec<f64> {
    let len = scores.len();
    if len == 0 || n == 0 {
        return Vec::new();
    }

    let mut peaks: Vec<(f64, f64)> = (0..len)
        .filter_map(|i| {
            let prev = scores[(i + len - 1) % len];
            let next = scores[(i + 1) % len];
            if scores[i] > prev && scores[i] > next {
                Some((scores[i], angles[i]))
            } else {
                None
            }
        })
        .collect();

    peaks.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    peaks.into_iter().take(n).map(|(_, a)| a).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// MusicEstimator
// ─────────────────────────────────────────────────────────────────────────────

/// Direction-of-Arrival estimator based on the MUSIC algorithm (R. O. Schmidt, 1986).
///
/// The MUSIC algorithm decomposes the sensor-array covariance matrix into signal and
/// noise subspaces.  Angles where the steering vector is orthogonal to the noise
/// subspace produce peaks in the resulting pseudo-spectrum.
///
/// # Examples
/// ```
/// use musicrs::MusicEstimator;
/// let est = MusicEstimator::default(); // Kinect 4-mic array, 720 angle steps, [0, 2π)
/// ```
#[derive(Debug, Clone)]
pub struct MusicEstimator {
    /// Antenna (microphone) positions as `[x, y]` pairs in metres.
    pub antenna_positions: Vec<[f64; 2]>,
    /// Number of angle steps in the angular search grid.
    pub angle_steps_num: usize,
    /// Angular search range `(start_rad, end_rad)`.
    pub search_interval: (f64, f64),
    /// Sample rate of the input signal in Hz, used to determine the sign convention
    /// for the steering vector (positive exponent for f < sr/2, negative for f ≥ sr/2).
    pub sample_rate: f64,
}

impl Default for MusicEstimator {
    /// Returns an estimator configured for the 4-element Kinect microphone array
    /// with 720 angle steps over the full circle [0, 2π).
    fn default() -> Self {
        Self {
            antenna_positions: DEFAULT_ANTENNA_POSITIONS.to_vec(),
            angle_steps_num: 720,
            search_interval: (0.0, 2.0 * PI),
            sample_rate: 16000.0,
        }
    }
}

impl MusicEstimator {
    /// Creates a new `MusicEstimator`.
    ///
    /// # Arguments
    /// * `antenna_positions` – Slice of `[x, y]` positions in metres.
    /// * `angle_steps_num`   – Number of discrete angles to evaluate.
    /// * `search_interval`   – `(start_rad, end_rad)` search range.
    /// * `sample_rate`       – Sample rate in Hz (used for aliasing-aware steering).
    pub fn new(
        antenna_positions: Vec<[f64; 2]>,
        angle_steps_num: usize,
        search_interval: (f64, f64),
        sample_rate: f64,
    ) -> Self {
        Self {
            antenna_positions,
            angle_steps_num,
            search_interval,
            sample_rate,
        }
    }

    /// Number of antennas in the array.
    #[inline]
    pub fn n_antennas(&self) -> usize {
        self.antenna_positions.len()
    }

    // ─── Core algorithm ───────────────────────────────────────────────────────

    /// Computes the steering vector for a plane wave arriving from `angle` (radians)
    /// at `frequency` (Hz).
    ///
    /// Element *j* is `exp(−i·2π·f·τⱼ)` where
    /// `τⱼ = (pⱼ · d̂) / c` is the propagation delay to antenna *j*.
    ///
    /// # Examples
    /// ```
    /// use musicrs::MusicEstimator;
    /// let est = MusicEstimator::default();
    /// let sv = est.steering_vector(0.0, 300.0);
    /// assert_eq!(sv.len(), 4);
    /// // All elements have unit magnitude
    /// for c in sv.iter() {
    ///     assert!((c.norm() - 1.0).abs() < 1e-12);
    /// }
    /// ```
    pub fn steering_vector(&self, angle: f64, frequency: f64) -> CVec {
        let dir = [angle.cos(), angle.sin()];
        // For f < Nyquist (sr/2): analytic signal has positive phase → use +exponent.
        // For f > Nyquist: signal aliases in the one-sided FFT; dominant eigenvector
        // phase is conjugated → use −exponent.
        // At exactly Nyquist (f == sr/2): analytic signal is real; noise subspace is
        // also real → use cos(phase) as a real steering vector.
        let nyquist = self.sample_rate / 2.0;
        let phase_sign = if frequency > nyquist {
            -1.0_f64
        } else {
            1.0_f64
        };
        let at_nyquist = (frequency - nyquist).abs() < 0.5;
        CVec::from_iterator(
            self.n_antennas(),
            self.antenna_positions.iter().map(|p| {
                let delay = (p[0] * dir[0] + p[1] * dir[1]) / SPEED_OF_SOUND;
                let phase = phase_sign * 2.0 * PI * frequency * delay;
                if at_nyquist {
                    // Real steering vector: imaginary part is zero
                    Cplx::new(phase.cos(), 0.0)
                } else {
                    Cplx::from_polar(1.0, phase)
                }
            }),
        )
    }

    /// Extracts the noise subspace from the Hermitian covariance matrix `cov`.
    ///
    /// Eigendecomposes `cov`, sorts eigenvectors by ascending eigenvalue magnitude,
    /// and returns the `n − n_sources` columns spanning the noise subspace.
    ///
    /// # Arguments
    /// * `cov`       – Hermitian covariance matrix (n × n).
    /// * `n_sources` – Number of signal sources (must be < n).
    ///
    /// # Panics
    /// Panics if `n_sources >= n_antennas`.
    pub fn noise_subspace(&self, cov: &CMatrix, n_sources: usize) -> CMatrix {
        let n = cov.nrows();
        assert!(
            n_sources < n,
            "n_sources must be less than the number of antennas"
        );

        // faer returns eigenvalues in ascending order; the first n_noise columns
        // are the noise subspace (smallest eigenvalues).
        let (_eigenvalues, eigenvectors) = hermitian_eigen(cov);
        let n_noise = n - n_sources;
        eigenvectors.columns(0, n_noise).into()
    }

    /// Generates the MUSIC pseudo-spectrum.
    ///
    /// For each angle θ in the search grid, the score is `1 / ‖Eₙᴴ · a(θ)‖²`.
    /// High values indicate signal directions of arrival.
    ///
    /// # Arguments
    /// * `noise_mat` – Noise subspace matrix (n × n_noise) from [`noise_subspace`].
    /// * `frequency` – Signal frequency in Hz.
    ///
    /// # Returns
    /// `(spectrum, angles)` — corresponding vectors of the same length.
    pub fn spectrum(&self, noise_mat: &CMatrix, frequency: f64) -> (Vec<f64>, Vec<f64>) {
        let (start, end) = self.search_interval;
        let steps = self.angle_steps_num;

        let angles: Vec<f64> = (0..steps)
            .map(|i| start + (end - start) * i as f64 / steps as f64)
            .collect();

        let scores: Vec<f64> = angles
            .iter()
            .map(|&theta| {
                let a = self.steering_vector(theta, frequency);
                let en_a = noise_mat.adjoint() * &a; // (n_noise × 1)
                let denom = en_a.norm_squared();
                if denom == 0.0 {
                    f64::INFINITY
                } else {
                    1.0 / denom
                }
            })
            .collect();

        (scores, angles)
    }

    /// Estimates the directions of arrival for `n_sources` narrowband sources.
    ///
    /// # Arguments
    /// * `data`      – Complex array data (n_antennas × n_samples).
    /// * `n_sources` – Number of sources to detect.
    /// * `frequency` – Signal frequency in Hz.
    ///
    /// # Returns
    /// Up to `n_sources` estimated angles (radians), sorted by descending peak height.
    pub fn estimate_doa(&self, data: &CMatrix, n_sources: usize, frequency: f64) -> Vec<f64> {
        let n_samples = data.ncols() as f64;
        let cov = (data * data.adjoint()).scale(1.0 / n_samples);
        let noise_mat = self.noise_subspace(&cov, n_sources);
        let (scores, angles) = self.spectrum(&noise_mat, frequency);
        find_peaks(&scores, &angles, n_sources)
    }

    /// Integrated Frequency-Band (IFB) MUSIC spectrum.
    ///
    /// Computes a narrowband MUSIC spectrum at each frequency bin and combines them
    /// via geometric mean for improved wideband robustness.
    ///
    /// # Arguments
    /// * `data`          – Real-valued microphone data (n_antennas × n_samples).
    /// * `nfft`          – STFT window size.
    /// * `n_sources`     – Number of sources.
    /// * `sampling_rate` – Sample rate in Hz.
    /// * `bins`          – Frequency bin indices to use; `None` uses all bins except DC.
    ///
    /// # Returns
    /// `(spectrum, angles)` — the IFB-MUSIC spectrum and corresponding angles.
    pub fn ifb_music_spectrum(
        &self,
        data: &DMatrix<f64>,
        nfft: usize,
        n_sources: usize,
        sampling_rate: f64,
        bins: Option<&[usize]>,
    ) -> (Vec<f64>, Vec<f64>) {
        let n_mics = data.nrows();
        let freq_bins = nfft / 2 + 1;

        // STFT per microphone channel
        let stfts: Vec<Vec<Vec<Cplx>>> = (0..n_mics)
            .map(|m| {
                let row: Vec<f64> = data.row(m).iter().copied().collect();
                stft(&row, nfft, 0)
            })
            .collect();

        let n_windows = stfts[0][0].len();

        let make_flat_spectrum = || {
            let (start, end) = self.search_interval;
            let steps = self.angle_steps_num;
            let angles: Vec<f64> = (0..steps)
                .map(|i| start + (end - start) * i as f64 / steps as f64)
                .collect();
            (vec![1.0; steps], angles)
        };

        if n_windows == 0 {
            return make_flat_spectrum();
        }

        let default_bins: Vec<usize> = (1..freq_bins).collect();
        let active_bins: &[usize] = bins.unwrap_or(&default_bins);

        let mut product_spectrum: Option<Vec<f64>> = None;
        let mut angles_out: Vec<f64> = Vec::new();
        let mut n_processed = 0usize;

        for &bin in active_bins {
            if bin >= freq_bins {
                continue;
            }
            let frequency = bin as f64 * sampling_rate / nfft as f64;

            // Cross-spectral matrix averaged over time windows
            // Pre-collect per-mic slices for this bin to avoid a range-indexed loop.
            let bin_slices: Vec<&[Cplx]> = stfts.iter().map(|s| s[bin].as_slice()).collect();
            let mut cov = CMatrix::zeros(n_mics, n_mics);
            for col in
                (0..n_windows).map(|w| CVec::from_iterator(n_mics, bin_slices.iter().map(|s| s[w])))
            {
                cov += &col * col.adjoint();
            }
            let scale = Cplx::new(1.0 / n_windows as f64, 0.0);
            cov *= scale;

            let noise_mat = self.noise_subspace(&cov, n_sources);
            let (spec, angles) = self.spectrum(&noise_mat, frequency);

            match &mut product_spectrum {
                None => {
                    product_spectrum = Some(spec);
                    angles_out = angles;
                }
                Some(prod) => {
                    for (p, s) in prod.iter_mut().zip(spec.iter()) {
                        *p *= s;
                    }
                }
            }
            n_processed += 1;
        }

        match product_spectrum {
            None => make_flat_spectrum(),
            Some(prod) => {
                let exp = 1.0 / n_processed as f64;
                let merged: Vec<f64> = prod.iter().map(|&v| v.powf(exp)).collect();
                (merged, angles_out)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use nalgebra::DVector as DVec;

    fn default_est() -> MusicEstimator {
        MusicEstimator::default()
    }

    // ── hermitian_eigen ───────────────────────────────────────────────────────

    /// Verify faer EVD eigenvalues on a known rank-1 4×4 complex Hermitian matrix.
    /// The rank-1 matrix a·aᴴ has eigenvalues [0, 0, 0, ‖a‖²].
    #[test]
    fn hermitian_eigen_rank1_eigenvalues() {
        let est = default_est();
        let sv = est.steering_vector(PI / 4.0, 500.0); // unit-magnitude elements
        let cov = &sv * sv.adjoint(); // rank-1, ‖sv‖² = 4
        let (mut ev, _) = hermitian_eigen(&cov);
        ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Three near-zero eigenvalues, one equal to ‖sv‖² ≈ 4
        assert_abs_diff_eq!(ev[0], 0.0, epsilon = 1e-8);
        assert_abs_diff_eq!(ev[1], 0.0, epsilon = 1e-8);
        assert_abs_diff_eq!(ev[2], 0.0, epsilon = 1e-8);
        assert_abs_diff_eq!(ev[3], 4.0, epsilon = 1e-8);
    }

    /// MUSIC from a rank-1 covariance built with NOISE: the noise subspace
    /// should be orthogonal to the steering vector, so `estimate_doa` returns the true angle.
    /// This test uses pyMUSIC sign convention: z_i = a(θ)_i × exp(iωt).
    #[test]
    fn estimate_doa_from_rank1_covariance() {
        let est = default_est();
        let true_angle = 0.3_f64; // radians
        let freq = 1000.0_f64;
        let n = est.n_antennas();

        // Build signal aligned with the steering vector (pyMUSIC convention)
        let a = est.steering_vector(true_angle, freq);
        // Covariance = signal subspace + small noise floor
        let cov = &a * a.adjoint() + CMatrix::identity(n, n).scale(0.01_f64);

        let noise = est.noise_subspace(&cov, 1);
        let (scores, angles) = est.spectrum(&noise, freq);
        let peaks = find_peaks(&scores, &angles, 1);
        assert_eq!(peaks.len(), 1);
        let diff = (peaks[0] - true_angle).abs();
        let diff_wrapped = diff.min(2.0 * PI - diff);
        assert!(
            diff_wrapped < 2.0_f64.to_radians(),
            "Expected ~{:.1}°, got {:.1}°",
            true_angle.to_degrees(),
            peaks[0].to_degrees()
        );
    }

    /// Eigenvalues of a diagonal real matrix should be the diagonal entries.
    #[test]
    fn hermitian_eigen_diagonal_real() {
        let mat = CMatrix::from_diagonal(&DVec::from_vec(vec![
            Cplx::new(1.0, 0.0),
            Cplx::new(2.0, 0.0),
            Cplx::new(3.0, 0.0),
        ]));
        let (eigenvalues, _) = hermitian_eigen(&mat);
        let mut ev = eigenvalues.clone();
        ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_abs_diff_eq!(ev[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ev[1], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ev[2], 3.0, epsilon = 1e-10);
    }

    /// Eigenvectors should be unitary: EᴴE ≈ I.
    #[test]
    fn hermitian_eigen_unitary_eigenvectors() {
        let n = 4;
        let est = default_est();
        // Build a rank-1 Hermitian matrix from a steering vector
        let sv = est.steering_vector(0.5, 400.0);
        let cov = &sv * sv.adjoint(); // rank-1
        let (_, vecs) = hermitian_eigen(&cov);
        let g = vecs.adjoint() * &vecs;
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(g[(i, j)].re, expected, epsilon = 1e-9);
                assert_abs_diff_eq!(g[(i, j)].im, 0.0, epsilon = 1e-9);
            }
        }
    }

    // ── steering_vector ───────────────────────────────────────────────────────

    #[test]
    fn steering_vector_length() {
        let est = default_est();
        assert_eq!(est.steering_vector(0.0, 300.0).len(), 4);
    }

    /// All steering-vector elements have unit magnitude.
    #[test]
    fn steering_vector_unit_magnitude() {
        let est = default_est();
        for angle in [0.0, PI / 6.0, PI / 4.0, PI / 2.0, PI, 3.0 * PI / 2.0] {
            for c in est.steering_vector(angle, 500.0).iter() {
                assert_abs_diff_eq!(c.norm(), 1.0, epsilon = 1e-12);
            }
        }
    }

    /// Single element at origin → zero delay → phase = 1+0j for all angles.
    #[test]
    fn steering_vector_origin_element() {
        let est = MusicEstimator::new(vec![[0.0, 0.0]], 360, (0.0, 2.0 * PI), 16000.0);
        let sv = est.steering_vector(1.23, 1000.0);
        assert_abs_diff_eq!(sv[0].re, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(sv[0].im, 0.0, epsilon = 1e-12);
    }

    // ── noise_subspace ────────────────────────────────────────────────────────

    /// Identity covariance + 1 source → noise subspace of shape (4 × 3).
    #[test]
    fn noise_subspace_shape() {
        let est = default_est();
        let n = est.n_antennas();
        let cov = CMatrix::identity(n, n);
        let noise = est.noise_subspace(&cov, 1);
        assert_eq!(noise.nrows(), n);
        assert_eq!(noise.ncols(), n - 1);
    }

    /// Noise-subspace columns should be orthonormal: EₙᴴEₙ ≈ I.
    #[test]
    fn noise_subspace_columns_orthonormal() {
        let est = default_est();
        let n = est.n_antennas();
        let cov = CMatrix::identity(n, n);
        let noise = est.noise_subspace(&cov, 1);
        let g = noise.adjoint() * &noise;
        for i in 0..g.nrows() {
            for j in 0..g.ncols() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(g[(i, j)].re, expected, epsilon = 1e-9);
                assert_abs_diff_eq!(g[(i, j)].im, 0.0, epsilon = 1e-9);
            }
        }
    }

    // ── spectrum ──────────────────────────────────────────────────────────────

    /// For a two-element array with identity noise subspace, spectrum is flat.
    #[test]
    fn spectrum_flat_for_full_noise_space() {
        let est = MusicEstimator::new(vec![[0.0, 0.0], [0.1, 0.0]], 360, (0.0, 2.0 * PI), 16000.0);
        let n = est.n_antennas();
        let noise = CMatrix::identity(n, n);
        let (scores, _) = est.spectrum(&noise, 300.0);
        let first = scores[0];
        for &s in &scores[1..] {
            assert_abs_diff_eq!(s, first, epsilon = 1e-9);
        }
    }

    // ── find_peaks ────────────────────────────────────────────────────────────

    #[test]
    fn find_peaks_returns_top_n() {
        let scores = vec![0.1, 1.0, 0.1, 0.1, 2.0, 0.1, 0.1, 3.0, 0.1];
        let angles: Vec<f64> = (0..scores.len()).map(|i| i as f64).collect();
        let peaks = find_peaks(&scores, &angles, 2);
        assert_eq!(peaks.len(), 2);
        assert_abs_diff_eq!(peaks[0], 7.0, epsilon = 1e-12); // highest peak first
    }

    // ── estimate_doa (synthetic noiseless plane wave) ─────────────────────────

    #[test]
    fn estimate_doa_synthetic_source() {
        let est = default_est();
        let true_angle = PI / 4.0; // 45°
        let frequency = 500.0_f64;
        let n_samples = 2048usize;
        let n = est.n_antennas();

        let steering = est.steering_vector(true_angle, frequency);
        let mut data = CMatrix::zeros(n, n_samples);
        for t in 0..n_samples {
            let s = Cplx::from_polar(1.0, 2.0 * PI * frequency * t as f64 / 16000.0);
            for i in 0..n {
                data[(i, t)] = steering[i] * s;
            }
        }

        let angles = est.estimate_doa(&data, 1, frequency);
        assert_eq!(angles.len(), 1);
        let diff = (angles[0] - true_angle).abs();
        let diff_wrapped = diff.min(2.0 * PI - diff);
        assert!(
            diff_wrapped < 3.0_f64.to_radians(),
            "Expected ~{:.1}°, got {:.1}°",
            true_angle.to_degrees(),
            angles[0].to_degrees()
        );
    }
}
