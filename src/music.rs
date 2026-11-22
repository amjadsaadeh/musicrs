//! MUSIC (MUltiple SIgnal Classification) algorithm for Direction-of-Arrival estimation.

use std::f64::consts::PI;

use nalgebra::DMatrix;
use nalgebra::DVector;
use num_complex::Complex;

use crate::error::{MusicError, Result};
use crate::stft::{analytic_signal, stft};

/// Speed of sound in air at ~20 °C, in metres per second.
const SPEED_OF_SOUND: f64 = 343.2;

/// Power ratio (−200 dB) below which a signal counts as silent: far below any real noise
/// floor, but above the rounding residue left by removing a constant offset or by FFT
/// leakage from an on-bin tone.
const SILENCE_RATIO: f64 = 1e-20;

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
    order.sort_unstable_by(|&a, &b| s_col.read(a).re.total_cmp(&s_col.read(b).re));

    let eigenvalues: Vec<f64> = order.iter().map(|&i| s_col.read(i).re).collect();
    // Copy the eigenvectors (reordered by ascending eigenvalue) back into nalgebra.
    // faer's eigenvectors already satisfy A·v = λ·v, so no conjugation is applied;
    // the steering-vector sign convention in `steering_vector` is matched to this.
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

/// Returns the angles of the `n` highest peaks in `scores`, sorted by descending peak
/// height.
///
/// A peak is a run of one or more equal neighbouring scores that is strictly higher than
/// the scores on either side of it; it is reported at the run's midpoint. Runs of equal
/// scores happen on symmetric spectra: a line array's spectrum is mirror-symmetric, so a
/// source at endfire between two grid points scores exactly the same on both.
///
/// With `circular` (the grid covers the full circle), the first and last points are
/// neighbours and a run may wrap around. Otherwise they are the two ends of an open range
/// with only one neighbour each, so a spectrum still rising at an end of the range peaks
/// there. A spectrum that is constant over a grid of two or more points has no peak; a
/// one-point open grid peaks at its only point.
fn find_peaks(scores: &[f64], angles: &[f64], n: usize, circular: bool) -> Vec<f64> {
    let len = scores.len();
    if len == 0 || n == 0 {
        return Vec::new();
    }

    // Runs of equal scores as inclusive index ranges.
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut run_start = 0;
    for i in 1..=len {
        if i == len || scores[i] != scores[run_start] {
            runs.push((run_start, i - 1));
            run_start = i;
        }
    }
    if runs.len() == 1 {
        // A single point on an open range has no neighbours, so it is a peak (as for
        // `angle_steps_num = 1`); a longer constant spectrum has none.
        return if len == 1 && !circular {
            vec![angles[0]]
        } else {
            Vec::new()
        };
    }
    // On the circle, a run at the start continues the run at the end. Its indices then
    // go past `len` and are taken modulo `len`.
    if circular && scores[0] == scores[len - 1] {
        let (_, first_end) = runs.remove(0);
        runs.last_mut().unwrap().1 = first_end + len;
    }

    let score = |i: usize| scores[i % len];
    let midpoint = |(a, b): (usize, usize)| {
        if b < len {
            (angles[a] + angles[b]) / 2.0
        } else {
            let step = angles[1] - angles[0];
            let mid = (a + b) as f64 / 2.0;
            if mid >= len as f64 {
                angles[0] + step * (mid - len as f64)
            } else {
                angles[a] + step * (mid - a as f64)
            }
        }
    };

    let r = runs.len();
    let mut peaks: Vec<(f64, f64)> = (0..r)
        .filter_map(|k| {
            let (a, b) = runs[k];
            let prev = match k {
                0 if circular => Some(score(runs[r - 1].1)),
                0 => None,
                _ => Some(score(runs[k - 1].1)),
            };
            let next = match k + 1 {
                j if j < r => Some(score(runs[j].0)),
                _ if circular => Some(score(runs[0].0)),
                _ => None,
            };
            let v = score(a);
            (prev.is_none_or(|p| v > p) && next.is_none_or(|q| v > q))
                .then(|| (v, midpoint((a, b))))
        })
        .collect();

    peaks.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    peaks.into_iter().take(n).map(|(_, a)| a).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Bin selection for IFB-MUSIC
// ─────────────────────────────────────────────────────────────────────────────

/// Picks the STFT bins that carry signal, for [`MusicEstimator::ifb_music_spectrum`].
///
/// IFB-MUSIC weights every bin equally, so bins holding only noise point the merged
/// spectrum in arbitrary directions. This keeps the bins whose power is within
/// `threshold_db` of the strongest bin. A bin's power is the trace of its cross-spectral
/// matrix: `|X|²` summed over microphones and averaged over windows, using the same
/// non-overlapping `nfft`-sample STFT windows as `ifb_music_spectrum`.
///
/// The DC bin is never selected (as with `bins = None`). Returns bin indices in ascending
/// order.
///
/// A tone between two bins leaks into its neighbours, and each selected bin is steered at
/// the bin's frequency rather than the tone's. The mismatch shifts the estimate, most
/// near endfire (0° / 180° for a line array): on the test recordings at `nfft = 1024`
/// (15.6 Hz bins), a single tone is off by up to 5.5° at 0° and 1.5° elsewhere. Choose
/// `nfft` so the tones fall on bins where possible.
///
/// # Errors
/// - [`MusicError::InvalidThreshold`] if `threshold_db` is negative or not finite.
/// - [`MusicError::InvalidNfft`] if `nfft` is 0; [`MusicError::NoUsableBins`] if it is
///   1 (the only bin is DC).
/// - [`MusicError::EmptyData`], [`MusicError::NonFiniteData`] or
///   [`MusicError::DataShorterThanWindow`] for unusable data.
/// - [`MusicError::SilentData`] if every bin except DC has zero power.
///
/// # Examples
/// ```
/// use musicrs::{MusicEstimator, select_bins_by_energy};
/// use nalgebra::DMatrix;
/// use std::f64::consts::PI;
///
/// // A 1 kHz tone: with nfft = 160 at 16 kHz the bins are 100 Hz apart, so it is bin 10.
/// let est = MusicEstimator::default();
/// let data = DMatrix::from_fn(4, 16000, |_, t| (2.0 * PI * 1000.0 * t as f64 / 16000.0).cos());
/// let bins = select_bins_by_energy(&data, 160, 20.0)?;
/// assert_eq!(bins, [10]);
/// let (spectrum, angles) = est.ifb_music_spectrum(&data, 160, 1, Some(&bins))?;
/// # let _ = (spectrum, angles);
/// # Ok::<(), musicrs::MusicError>(())
/// ```
pub fn select_bins_by_energy(
    data: &DMatrix<f64>,
    nfft: usize,
    threshold_db: f64,
) -> Result<Vec<usize>> {
    if !(threshold_db.is_finite() && threshold_db >= 0.0) {
        return Err(MusicError::InvalidThreshold(threshold_db));
    }
    if nfft == 0 {
        return Err(MusicError::InvalidNfft);
    }
    if nfft == 1 {
        return Err(MusicError::NoUsableBins);
    }
    check_real_samples(data)?;
    if data.ncols() < nfft {
        return Err(MusicError::DataShorterThanWindow {
            nfft,
            samples: data.ncols(),
        });
    }
    let mut power = vec![0.0; nfft / 2 + 1];
    for m in 0..data.nrows() {
        let row: Vec<f64> = data.row(m).iter().copied().collect();
        for (p, bin) in power.iter_mut().zip(stft(&row, nfft, 0)?) {
            *p += bin.iter().map(|c| c.norm_sqr()).sum::<f64>() / bin.len() as f64;
        }
    }
    let strongest = power[1..].iter().copied().fold(0.0, f64::max);
    if strongest <= 0.0 {
        return Err(MusicError::SilentData);
    }
    let floor = strongest * 10f64.powf(-threshold_db / 10.0);
    Ok((1..power.len()).filter(|&k| power[k] >= floor).collect())
}

/// A power of two close to `max_abs`, the largest absolute value in some data, to divide
/// the data by. MUSIC does not depend on the data's scale, and dividing by a power of two
/// is exact, so this changes no result: it only keeps the sums of squares in the
/// covariance and energy from overflowing for very loud data (or underflowing for very
/// quiet data).
fn power_of_two_scale(max_abs: f64) -> f64 {
    if max_abs > 0.0 {
        2f64.powi((max_abs.log2().floor() as i32).clamp(-1000, 1000))
    } else {
        1.0
    }
}

/// `data` divided by [`power_of_two_scale`] of its largest absolute value.
fn normalised_real(data: &DMatrix<f64>) -> DMatrix<f64> {
    data / power_of_two_scale(data.amax())
}

/// Checks that real data has samples and that all of them are finite.
fn check_real_samples(data: &DMatrix<f64>) -> Result<()> {
    if data.is_empty() {
        return Err(MusicError::EmptyData);
    }
    if !data.iter().all(|x| x.is_finite()) {
        return Err(MusicError::NonFiniteData);
    }
    Ok(())
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
/// Functions that take data return [`MusicError`] for data they cannot use (wrong row
/// count, no samples, NaN, silence) and for invalid parameters (too many sources, an
/// unusable STFT size or bin list).
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
    /// Angular search range `(start_rad, end_rad)`; the grid includes `start_rad` and
    /// excludes `end_rad`. A range spanning exactly 2π is treated as the full circle (its
    /// ends are adjacent when picking peaks); any other range is open at both ends.
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
    ///
    /// The angle grid is not checked here, because the fields are public and can change
    /// later: the methods that search the grid return [`MusicError::InvalidAngleGrid`]
    /// unless `angle_steps_num > 0` and `search_interval` is finite with `start < end`.
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

    /// Whether [`search_interval`](Self::search_interval) spans exactly the full circle,
    /// so that the first and last grid angles are neighbours.
    fn covers_full_circle(&self) -> bool {
        let (start, end) = self.search_interval;
        ((end - start).abs() - 2.0 * PI).abs() <= 1e-9
    }

    /// The search grid: `angle_steps_num` angles from `start` (included) to `end`
    /// (excluded). pyMUSIC asserts `start < end` in `MUSICEst.__init__`.
    fn angle_grid(&self) -> Result<Vec<f64>> {
        let (start, end) = self.search_interval;
        let steps = self.angle_steps_num;
        if steps == 0 || !(start.is_finite() && end.is_finite() && start < end) {
            return Err(MusicError::InvalidAngleGrid {
                angle_steps_num: steps,
                search_interval: self.search_interval,
            });
        }
        Ok((0..steps)
            .map(|i| start + (end - start) * i as f64 / steps as f64)
            .collect())
    }

    fn check_rows(&self, got: usize) -> Result<()> {
        if got == self.n_antennas() {
            Ok(())
        } else {
            Err(MusicError::RowCountMismatch {
                expected: self.n_antennas(),
                got,
            })
        }
    }

    fn check_sources(&self, n_sources: usize) -> Result<()> {
        if n_sources < self.n_antennas() {
            Ok(())
        } else {
            Err(MusicError::TooManySources {
                n_sources,
                n_antennas: self.n_antennas(),
            })
        }
    }

    // ─── Core algorithm ───────────────────────────────────────────────────────

    /// Computes the steering vector for a plane wave arriving from `angle` (radians)
    /// at `frequency` (Hz).
    ///
    /// Element *j* is `exp(+i·2π·f·τⱼ)` where `τⱼ = (pⱼ · d̂) / c` is the projection of
    /// antenna *j*'s position onto the arrival direction, divided by the speed of sound. This
    /// matches the non-conjugated [`analytic_signal`] and STFT bins
    /// (pyMUSIC conjugates its analytic signal and uses `exp(−i·2π·f·τⱼ)`; the two agree below
    /// Nyquist). Relative to [`sample_rate`](Self::sample_rate):
    /// * above Nyquist the exponent is negated, `exp(−i·2π·f·τⱼ)`, because the recorded tone
    ///   is aliased and its phase is conjugated;
    /// * within 0.5 Hz of Nyquist the vector is real, `cos(2π·f·τⱼ)`, because the analytic
    ///   signal there is real too.
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
    /// * `cov`       – Hermitian covariance matrix (n_antennas × n_antennas).
    /// * `n_sources` – Number of signal sources (must be < n_antennas).
    ///
    /// # Errors
    /// [`MusicError::CovarianceShape`] if `cov` is not n_antennas × n_antennas,
    /// [`MusicError::NonFiniteData`] if it contains NaN or infinity, and
    /// [`MusicError::TooManySources`] if `n_sources >= n_antennas`.
    pub fn noise_subspace(&self, cov: &CMatrix, n_sources: usize) -> Result<CMatrix> {
        let n = self.n_antennas();
        if cov.shape() != (n, n) {
            return Err(MusicError::CovarianceShape {
                n_antennas: n,
                rows: cov.nrows(),
                cols: cov.ncols(),
            });
        }
        self.check_sources(n_sources)?;
        if !cov.iter().all(|c| c.re.is_finite() && c.im.is_finite()) {
            return Err(MusicError::NonFiniteData);
        }

        // Eigenvectors in ascending eigenvalue order; the first n_noise columns are the
        // noise subspace (smallest eigenvalues).
        let (_eigenvalues, eigenvectors) = hermitian_eigen(cov);
        let n_noise = cov.nrows() - n_sources;
        Ok(eigenvectors.columns(0, n_noise).into())
    }

    /// Generates the MUSIC pseudo-spectrum.
    ///
    /// For each angle θ in the search grid, the score is `1 / ‖Eₙᴴ · a(θ)‖²`.
    /// High values indicate signal directions of arrival.
    ///
    /// # Arguments
    /// * `noise_mat` – Noise subspace matrix (n_antennas × n_noise) from
    ///   [`noise_subspace`](Self::noise_subspace).
    /// * `frequency` – Signal frequency in Hz.
    ///
    /// # Returns
    /// `(spectrum, angles)` — corresponding vectors of the same length.
    ///
    /// # Errors
    /// [`MusicError::RowCountMismatch`] if `noise_mat` does not have one row per antenna,
    /// [`MusicError::InvalidAngleGrid`] if the angle grid is unusable (see [`new`](Self::new)).
    pub fn spectrum(&self, noise_mat: &CMatrix, frequency: f64) -> Result<(Vec<f64>, Vec<f64>)> {
        self.check_rows(noise_mat.nrows())?;
        let angles = self.angle_grid()?;
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

        Ok((scores, angles))
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
    ///
    /// # Errors
    /// - [`MusicError::RowCountMismatch`] if `data` does not have one row per antenna.
    /// - [`MusicError::EmptyData`] if it has no samples, [`MusicError::NonFiniteData`]
    ///   if it contains NaN or infinity, [`MusicError::SilentData`] if it is all zeros.
    /// - [`MusicError::TooManySources`] if `n_sources >= n_antennas`.
    /// - [`MusicError::InvalidAngleGrid`] if the angle grid is unusable (see [`new`](Self::new)).
    pub fn estimate_doa(
        &self,
        data: &CMatrix,
        n_sources: usize,
        frequency: f64,
    ) -> Result<Vec<f64>> {
        self.check_rows(data.nrows())?;
        self.check_sources(n_sources)?;
        if data.ncols() == 0 {
            return Err(MusicError::EmptyData);
        }
        if !data.iter().all(|c| c.re.is_finite() && c.im.is_finite()) {
            return Err(MusicError::NonFiniteData);
        }
        let max_abs = data
            .iter()
            .fold(0.0, |m: f64, c| m.max(c.re.abs()).max(c.im.abs()));
        let scale = power_of_two_scale(max_abs);
        let data = data.map(|c| c / scale);
        let n_samples = data.ncols() as f64;
        let cov = (&data * data.adjoint()).scale(1.0 / n_samples);
        if cov.trace().re == 0.0 {
            return Err(MusicError::SilentData);
        }
        let noise_mat = self.noise_subspace(&cov, n_sources)?;
        let (scores, angles) = self.spectrum(&noise_mat, frequency)?;
        Ok(find_peaks(
            &scores,
            &angles,
            n_sources,
            self.covers_full_circle(),
        ))
    }

    /// Estimates DOAs from real-valued microphone data, like pyMUSIC's `getSourceAngles`.
    ///
    /// Removes each channel's mean, converts each channel to its analytic signal with
    /// [`analytic_signal`] and then calls [`estimate_doa`](Self::estimate_doa). Without
    /// the mean removal a DC offset acts as a second source at broadside.
    ///
    /// # Arguments
    /// * `data`      – Real array data (n_antennas × n_samples).
    /// * `n_sources` – Number of sources to detect.
    /// * `frequency` – Signal frequency in Hz.
    ///
    /// # Errors
    /// As [`estimate_doa`](Self::estimate_doa); data that is constant on every channel (its
    /// energy after removing the mean is 200 dB below the input's) is
    /// [`MusicError::SilentData`].
    ///
    /// # Examples
    /// ```
    /// use musicrs::MusicEstimator;
    /// use nalgebra::DMatrix;
    /// use std::f64::consts::PI;
    ///
    /// let est = MusicEstimator::default();
    /// let (freq, doa, sr) = (1000.0, 60.0_f64.to_radians(), est.sample_rate);
    /// // Plane wave: microphone j hears the source delayed by -(p_j · d) / c
    /// let data = DMatrix::from_fn(4, 4000, |j, t| {
    ///     let p = est.antenna_positions[j];
    ///     let advance = (p[0] * doa.cos() + p[1] * doa.sin()) / 343.2;
    ///     (2.0 * PI * freq * (t as f64 / sr + advance)).cos()
    /// });
    /// let angles = est.estimate_doa_from_real(&data, 1, freq)?;
    /// // A linear array cannot tell θ from 360° − θ
    /// let deg = angles[0].to_degrees();
    /// assert!((deg - 60.0).abs() < 2.0 || (deg - 300.0).abs() < 2.0, "got {deg}");
    /// # Ok::<(), musicrs::MusicError>(())
    /// ```
    pub fn estimate_doa_from_real(
        &self,
        data: &DMatrix<f64>,
        n_sources: usize,
        frequency: f64,
    ) -> Result<Vec<f64>> {
        self.check_rows(data.nrows())?;
        check_real_samples(data)?;
        let data = &normalised_real(data);
        let mut complex = CMatrix::zeros(data.nrows(), data.ncols());
        let (mut raw_energy, mut centred_energy) = (0.0, 0.0);
        for m in 0..data.nrows() {
            let mean = data.row(m).mean();
            let row: Vec<f64> = data.row(m).iter().map(|x| x - mean).collect();
            raw_energy += data.row(m).norm_squared();
            centred_energy += row.iter().map(|x| x * x).sum::<f64>();
            for (t, c) in analytic_signal(&row).into_iter().enumerate() {
                complex[(m, t)] = c;
            }
        }
        if centred_energy <= SILENCE_RATIO * raw_energy {
            return Err(MusicError::SilentData);
        }
        self.estimate_doa(&complex, n_sources, frequency)
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
    /// * `bins`          – Frequency bin indices to use; `None` uses all bins except DC.
    ///   Bins at or above `nfft / 2 + 1` are ignored, and so are bins with no power.
    ///
    /// Bin frequencies are derived from [`MusicEstimator::sample_rate`], the same rate
    /// [`steering_vector`](Self::steering_vector) uses for its Nyquist handling.
    /// Bins only cover `[0, sample_rate / 2]`, so a tone recorded above Nyquist is
    /// steered at its aliased frequency and its DOA cannot be recovered here.
    ///
    /// # Returns
    /// `(spectrum, angles)` — the IFB-MUSIC spectrum and corresponding angles.
    ///
    /// # Errors
    /// - [`MusicError::RowCountMismatch`], [`MusicError::EmptyData`],
    ///   [`MusicError::NonFiniteData`] for unusable data, and
    ///   [`MusicError::DataShorterThanWindow`] if it has no complete STFT window.
    /// - [`MusicError::InvalidNfft`] if `nfft` is 0.
    /// - [`MusicError::TooManySources`] if `n_sources >= n_antennas`.
    /// - [`MusicError::NoUsableBins`] if none of the bins exists (for `None`: `nfft` is 1).
    /// - [`MusicError::SilentData`] if no usable bin has power.
    /// - [`MusicError::InvalidAngleGrid`] if the angle grid is unusable (see [`new`](Self::new)).
    pub fn ifb_music_spectrum(
        &self,
        data: &DMatrix<f64>,
        nfft: usize,
        n_sources: usize,
        bins: Option<&[usize]>,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        self.check_rows(data.nrows())?;
        check_real_samples(data)?;
        self.check_sources(n_sources)?;
        if nfft == 0 {
            return Err(MusicError::InvalidNfft);
        }
        if data.ncols() < nfft {
            return Err(MusicError::DataShorterThanWindow {
                nfft,
                samples: data.ncols(),
            });
        }
        let data = &normalised_real(data);
        let n_mics = data.nrows();
        let freq_bins = nfft / 2 + 1;

        let default_bins: Vec<usize> = (1..freq_bins).collect();
        let active_bins: Vec<usize> = bins
            .unwrap_or(&default_bins)
            .iter()
            .copied()
            .filter(|&bin| bin < freq_bins)
            .collect();
        if active_bins.is_empty() {
            return Err(MusicError::NoUsableBins);
        }

        // STFT per microphone channel
        let stfts: Vec<Vec<Vec<Cplx>>> = (0..n_mics)
            .map(|m| {
                let row: Vec<f64> = data.row(m).iter().copied().collect();
                stft(&row, nfft, 0)
            })
            .collect::<Result<_>>()?;
        let n_windows = stfts[0][0].len();

        // Geometric mean, accumulated as a sum of logarithms: the product of hundreds of
        // per-bin spectra overflows f64.
        let mut log_sum: Option<Vec<f64>> = None;
        let mut angles_out: Vec<f64> = Vec::new();
        let mut n_processed = 0usize;

        // Cross-spectral matrix of each bin, averaged over time windows
        let covs: Vec<(usize, CMatrix)> = active_bins
            .into_iter()
            .map(|bin| {
                let bin_slices: Vec<&[Cplx]> = stfts.iter().map(|s| s[bin].as_slice()).collect();
                let mut cov = CMatrix::zeros(n_mics, n_mics);
                for col in (0..n_windows)
                    .map(|w| CVec::from_iterator(n_mics, bin_slices.iter().map(|s| s[w])))
                {
                    cov += &col * col.adjoint();
                }
                (bin, cov.scale(1.0 / n_windows as f64))
            })
            .collect();
        // Silence is judged against the strongest bin of the whole spectrum, DC included,
        // so that a requested bin holding only rounding residue is still recognised.
        let strongest = stfts
            .iter()
            .fold(vec![0.0; freq_bins], |mut power, mic| {
                for (p, bin) in power.iter_mut().zip(mic) {
                    *p += bin.iter().map(|c| c.norm_sqr()).sum::<f64>() / n_windows as f64;
                }
                power
            })
            .into_iter()
            .fold(0.0, f64::max);

        for (bin, cov) in covs {
            if cov.trace().re <= SILENCE_RATIO * strongest {
                // No power in this bin: its noise subspace would be arbitrary.
                continue;
            }
            let frequency = bin as f64 * self.sample_rate / nfft as f64;
            let noise_mat = self.noise_subspace(&cov, n_sources)?;
            let (spec, angles) = self.spectrum(&noise_mat, frequency)?;

            match &mut log_sum {
                None => {
                    log_sum = Some(spec.iter().map(|s| s.ln()).collect());
                    angles_out = angles;
                }
                Some(sum) => {
                    for (acc, s) in sum.iter_mut().zip(spec.iter()) {
                        *acc += s.ln();
                    }
                }
            }
            n_processed += 1;
        }

        let sum = log_sum.ok_or(MusicError::SilentData)?;
        let merged: Vec<f64> = sum
            .iter()
            .map(|&v| (v / n_processed as f64).exp())
            .collect();
        Ok((merged, angles_out))
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
        let (ev, _) = hermitian_eigen(&cov);
        // Ascending order, as returned (not re-sorted here): three near-zero eigenvalues, one equal to ‖sv‖² ≈ 4
        assert_abs_diff_eq!(ev[0], 0.0, epsilon = 1e-8);
        assert_abs_diff_eq!(ev[1], 0.0, epsilon = 1e-8);
        assert_abs_diff_eq!(ev[2], 0.0, epsilon = 1e-8);
        assert_abs_diff_eq!(ev[3], 4.0, epsilon = 1e-8);
    }

    /// MUSIC from a rank-1 covariance built with NOISE: the noise subspace
    /// should be orthogonal to the steering vector, so `estimate_doa` returns the true angle.
    /// The signal follows musicrs's sign convention: z_i = a(θ)_i × exp(iωt).
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

        let noise = est.noise_subspace(&cov, 1).unwrap();
        let (scores, angles) = est.spectrum(&noise, freq).unwrap();
        let peaks = find_peaks(&scores, &angles, 1, true);
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

    /// Eigenvalues of a diagonal real matrix are its diagonal entries, returned in
    /// ascending order even when the diagonal is not.
    #[test]
    fn hermitian_eigen_diagonal_real() {
        let mat = CMatrix::from_diagonal(&DVec::from_vec(vec![
            Cplx::new(3.0, 0.0),
            Cplx::new(1.0, 0.0),
            Cplx::new(2.0, 0.0),
        ]));
        let (ev, _) = hermitian_eigen(&mat);
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

    /// A dense Hermitian 4×4 matrix with distinct eigenvalues.
    fn dense_hermitian() -> CMatrix {
        let est = default_est();
        let a = est.steering_vector(0.4, 700.0);
        let b = est.steering_vector(2.1, 700.0);
        let diag = CMatrix::from_diagonal(&DVec::from_vec(
            [0.1, 0.2, 0.3, 0.4].map(|d| Cplx::new(d, 0.0)).to_vec(),
        ));
        (&a * a.adjoint()).scale(3.0) + &b * b.adjoint() + diag
    }

    /// Eigenvalues come back ascending (`noise_subspace` relies on this) and every
    /// column satisfies A·v = λ·v.
    #[test]
    fn hermitian_eigen_ascending_and_satisfies_definition() {
        let mat = dense_hermitian();
        let (vals, vecs) = hermitian_eigen(&mat);
        assert!(
            vals.windows(2).all(|w| w[0] <= w[1]),
            "eigenvalues not ascending: {vals:?}"
        );
        for (k, &lambda) in vals.iter().enumerate() {
            let v = vecs.column(k);
            let residual = (&mat * v - v * Cplx::new(lambda, 0.0)).norm();
            assert!(residual < 1e-10, "column {k}: ‖A·v − λ·v‖ = {residual:e}");
        }
    }

    /// The noise subspace is spanned by the eigenvectors of the smallest eigenvalues, so it
    /// is orthogonal to every source steering vector.
    #[test]
    fn noise_subspace_orthogonal_to_sources() {
        let est = default_est();
        let n = est.n_antennas();
        let a = est.steering_vector(0.4, 700.0);
        let b = est.steering_vector(2.1, 700.0);
        let cov = (&a * a.adjoint()).scale(3.0) + &b * b.adjoint();
        let noise = est
            .noise_subspace(&(cov + CMatrix::identity(n, n).scale(1e-3)), 2)
            .unwrap();
        assert_eq!(noise.ncols(), n - 2);
        for sv in [&a, &b] {
            let leak = (noise.adjoint() * sv).norm();
            assert!(leak < 1e-8, "‖Eₙᴴ·a‖ = {leak:e}");
        }
    }

    #[test]
    fn noise_subspace_rejects_too_many_sources() {
        let est = default_est();
        let n = est.n_antennas();
        assert_eq!(
            est.noise_subspace(&CMatrix::identity(n, n), n),
            Err(MusicError::TooManySources {
                n_sources: n,
                n_antennas: n
            })
        );
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

    /// `exp(+j·2π·f·τⱼ)` with `τⱼ = (pⱼ·d̂)/c`, written out independently of
    /// `steering_vector`.
    fn expected_steering(est: &MusicEstimator, angle: f64, frequency: f64) -> Vec<Cplx> {
        est.antenna_positions
            .iter()
            .map(|p| {
                let tau = (p[0] * angle.cos() + p[1] * angle.sin()) / SPEED_OF_SOUND;
                Cplx::from_polar(1.0, 2.0 * PI * frequency * tau)
            })
            .collect()
    }

    fn assert_cplx_eq(got: Cplx, expected: Cplx) {
        assert_abs_diff_eq!(got.re, expected.re, epsilon = 1e-12);
        assert_abs_diff_eq!(got.im, expected.im, epsilon = 1e-12);
    }

    /// Below Nyquist, element j is `exp(+j·2π·f·τⱼ)` (musicrs's sign convention).
    #[test]
    fn steering_vector_phase_below_nyquist() {
        let est = MusicEstimator {
            antenna_positions: vec![[0.1, 0.0], [0.0, 0.07], [-0.05, -0.03]],
            ..default_est()
        };
        for (angle, f) in [(0.7, 1200.0), (2.5, 300.0), (4.0, 7999.0)] {
            let sv = est.steering_vector(angle, f);
            for (&got, want) in sv.iter().zip(expected_steering(&est, angle, f)) {
                assert_cplx_eq(got, want);
            }
        }
    }

    /// Above Nyquist the phase sign flips: the vector is the conjugate of the formula.
    #[test]
    fn steering_vector_conjugated_above_nyquist() {
        let est = default_est();
        let (angle, f) = (0.7, 9100.0);
        let sv = est.steering_vector(angle, f);
        for (&got, want) in sv.iter().zip(expected_steering(&est, angle, f)) {
            assert_cplx_eq(got, want.conj());
        }
    }

    /// Within 0.5 Hz of Nyquist the vector is real, `cos(2π·f·τⱼ)`; just outside it is not.
    #[test]
    fn steering_vector_real_at_nyquist() {
        let est = default_est();
        let angle = 0.7;
        for f in [8000.0, 7999.6, 8000.4] {
            let sv = est.steering_vector(angle, f);
            for (&got, want) in sv.iter().zip(expected_steering(&est, angle, f)) {
                assert_cplx_eq(got, Cplx::new(want.re, 0.0));
            }
        }
        let above = est.steering_vector(angle, 8000.6);
        assert!(
            above.iter().any(|c| c.im.abs() > 1e-3),
            "8000.6 Hz should be complex"
        );
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
        let noise = est.noise_subspace(&cov, 1).unwrap();
        assert_eq!(noise.nrows(), n);
        assert_eq!(noise.ncols(), n - 1);
    }

    /// Noise-subspace columns should be orthonormal: EₙᴴEₙ ≈ I.
    #[test]
    fn noise_subspace_columns_orthonormal() {
        let est = default_est();
        let n = est.n_antennas();
        let cov = CMatrix::identity(n, n);
        let noise = est.noise_subspace(&cov, 1).unwrap();
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
        let (scores, _) = est.spectrum(&noise, 300.0).unwrap();
        let first = scores[0];
        for &s in &scores[1..] {
            assert_abs_diff_eq!(s, first, epsilon = 1e-9);
        }
    }

    // ── select_bins_by_energy ──────────────────────────────────────────────────

    /// Two on-bin tones, the second 6 dB weaker, plus a DC offset stronger than both.
    fn two_tones_with_dc() -> DMatrix<f64> {
        let (nfft, sr) = (64.0, 16000.0);
        DMatrix::from_fn(3, 64 * 20, |m, t| {
            let t = t as f64 / sr;
            let tone = |bin: f64, amp: f64| amp * (2.0 * PI * bin * sr / nfft * t + m as f64).cos();
            5.0 + tone(5.0, 1.0) + tone(20.0, 0.5)
        })
    }

    #[test]
    fn select_bins_by_energy_applies_threshold_and_skips_dc() {
        let data = two_tones_with_dc();
        assert_eq!(select_bins_by_energy(&data, 64, 20.0).unwrap(), [5, 20]);
        assert_eq!(select_bins_by_energy(&data, 64, 7.0).unwrap(), [5, 20]);
        assert_eq!(select_bins_by_energy(&data, 64, 5.0).unwrap(), [5]);
        assert_eq!(select_bins_by_energy(&data, 64, 0.0).unwrap(), [5]);
    }

    #[test]
    fn select_bins_by_energy_rejects_silence_and_short_data() {
        assert_eq!(
            select_bins_by_energy(&DMatrix::zeros(4, 640), 64, 20.0),
            Err(MusicError::SilentData)
        );
        // A DC offset alone is silence too: DC is never selected
        assert_eq!(
            select_bins_by_energy(&DMatrix::from_element(4, 640, 0.5), 64, 20.0),
            Err(MusicError::SilentData)
        );
        assert_eq!(
            select_bins_by_energy(&two_tones_with_dc().columns(0, 63).into(), 64, 20.0),
            Err(MusicError::DataShorterThanWindow {
                nfft: 64,
                samples: 63
            })
        );
    }

    #[test]
    fn select_bins_by_energy_rejects_bad_parameters() {
        let data = two_tones_with_dc();
        for t in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                select_bins_by_energy(&data, 64, t),
                Err(MusicError::InvalidThreshold(_))
            ));
        }
        assert_eq!(
            select_bins_by_energy(&data, 0, 20.0),
            Err(MusicError::InvalidNfft)
        );
        assert_eq!(
            select_bins_by_energy(&data, 1, 20.0),
            Err(MusicError::NoUsableBins)
        );
        let mut nan = data.clone();
        nan[(1, 7)] = f64::NAN;
        assert_eq!(
            select_bins_by_energy(&nan, 64, 20.0),
            Err(MusicError::NonFiniteData)
        );
        assert_eq!(
            select_bins_by_energy(&DMatrix::zeros(0, 640), 64, 20.0),
            Err(MusicError::EmptyData)
        );
    }

    // ── find_peaks ────────────────────────────────────────────────────────────

    #[test]
    fn find_peaks_returns_top_n() {
        let scores = vec![0.1, 1.0, 0.1, 0.1, 2.0, 0.1, 0.1, 3.0, 0.1];
        let angles: Vec<f64> = (0..scores.len()).map(|i| i as f64).collect();
        let peaks = find_peaks(&scores, &angles, 2, true);
        assert_eq!(peaks.len(), 2);
        assert_abs_diff_eq!(peaks[0], 7.0, epsilon = 1e-12); // highest peak first
    }

    /// On the full circle the first point is a neighbour of the last one.
    #[test]
    fn find_peaks_circular_wraps_around() {
        let angles = [0.0, 1.0, 2.0, 3.0, 4.0];
        // Index 4 is a peak only through its wrapped neighbour at index 0
        assert_eq!(
            find_peaks(&[4.0, 1.0, 1.0, 1.0, 5.0], &angles, 5, true),
            [4.0]
        );
        // Index 0 is a peak only through its wrapped neighbour at index 4
        assert_eq!(
            find_peaks(&[5.0, 1.0, 1.0, 1.0, 4.0], &angles, 5, true),
            [0.0]
        );
    }

    /// On an open range the ends are not neighbours: each is compared to its one neighbour.
    #[test]
    fn find_peaks_open_range_keeps_both_ends() {
        let angles = [0.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            find_peaks(&[4.0, 1.0, 1.0, 1.0, 5.0], &angles, 5, false),
            [4.0, 0.0]
        );
        // An end lower than its neighbour is not a peak
        assert_eq!(
            find_peaks(&[1.0, 2.0, 3.0, 2.0, 1.0], &angles, 5, false),
            [2.0]
        );
    }

    /// A run of equal scores is one peak, reported at its midpoint.
    #[test]
    fn find_peaks_plateau_reports_midpoint() {
        let angles = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        for circular in [false, true] {
            assert_eq!(
                find_peaks(&[0.0, 1.0, 3.0, 3.0, 1.0, 0.0], &angles, 5, circular),
                [2.5]
            );
            assert_eq!(
                find_peaks(&[0.0, 3.0, 3.0, 3.0, 1.0, 0.0], &angles, 5, circular),
                [2.0]
            );
        }
        // A plateau at an open end peaks there
        assert_eq!(
            find_peaks(&[3.0, 3.0, 1.0, 0.0, 1.0, 2.0], &angles, 5, false),
            [0.5, 5.0]
        );
    }

    /// On the circle a plateau may wrap from the last grid point to the first.
    #[test]
    fn find_peaks_plateau_wraps_around() {
        let angles = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        // Indices 5, 0, 1: the midpoint is index 0
        assert_eq!(
            find_peaks(&[3.0, 3.0, 1.0, 0.0, 1.0, 3.0], &angles, 5, true),
            [0.0]
        );
        // Indices 5, 0: the midpoint lies between the last and the first grid point
        assert_eq!(
            find_peaks(&[3.0, 1.0, 0.0, 0.0, 1.0, 3.0], &angles, 5, true),
            [5.5]
        );
        // Open range: the same scores are two separate end peaks
        assert_eq!(
            find_peaks(&[3.0, 1.0, 0.0, 0.0, 1.0, 3.0], &angles, 5, false),
            [0.0, 5.0]
        );
    }

    #[test]
    fn find_peaks_single_point() {
        assert_eq!(find_peaks(&[2.0], &[0.5], 1, false), vec![0.5]);
        // On the circle the only point is its own neighbour, so it is not higher.
        assert!(find_peaks(&[2.0], &[0.5], 1, true).is_empty());
    }

    #[test]
    fn find_peaks_constant_spectrum_has_no_peak() {
        let angles = [0.0, 1.0, 2.0];
        for circular in [false, true] {
            assert!(find_peaks(&[2.0; 3], &angles, 1, circular).is_empty());
        }
    }

    #[test]
    fn covers_full_circle_only_for_2pi_span() {
        let with_interval = |interval| MusicEstimator {
            search_interval: interval,
            ..MusicEstimator::default()
        };
        assert!(with_interval((0.0, 2.0 * PI)).covers_full_circle());
        assert!(with_interval((-PI, PI)).covers_full_circle());
        assert!(!with_interval((0.0, PI)).covers_full_circle());
        assert!(!with_interval((0.0, 1.9 * PI)).covers_full_circle());
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

        let angles = est.estimate_doa(&data, 1, frequency).unwrap();
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

    // ── find_peaks on partial search ranges ───────────────────────────────────

    /// Deterministic pseudo-random phases in [0, 2π) (LCG), so the two test sources are
    /// uncorrelated without pulling in a `rand` dependency.
    fn pseudo_random_phases(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                2.0 * PI * (state >> 11) as f64 / (1u64 << 53) as f64
            })
            .collect()
    }

    /// Two uncorrelated sources sitting on the first and last grid angles of a half-circle
    /// search range. Both are true peaks; the first and last grid points are not neighbours,
    /// so neither may be compared against the other.
    #[test]
    fn estimate_doa_finds_sources_at_both_ends_of_half_circle() {
        let steps = 360;
        let est = MusicEstimator::new(
            DEFAULT_ANTENNA_POSITIONS.to_vec(),
            steps,
            (0.0, PI),
            16000.0,
        );
        let first = 0.0;
        let last = PI * (steps - 1) as f64 / steps as f64;
        let frequency = 1000.0;
        let n_samples = 4000;

        let a1 = est.steering_vector(first, frequency);
        let a2 = est.steering_vector(last, frequency);
        let phases = pseudo_random_phases(n_samples, 7);
        let noise = pseudo_random_phases(4 * n_samples, 11);
        let data = CMatrix::from_fn(est.n_antennas(), n_samples, |m, t| {
            let carrier = 2.0 * PI * frequency * t as f64 / est.sample_rate;
            a1[m] * Cplx::from_polar(1.0, carrier)
                + a2[m] * Cplx::from_polar(1.0, carrier + phases[t])
                + Cplx::from_polar(0.05, noise[m * n_samples + t])
        });

        let mut found: Vec<f64> = est.estimate_doa(&data, 2, frequency).unwrap();
        found.sort_by(f64::total_cmp);
        let step = PI / steps as f64;
        assert_eq!(found.len(), 2, "expected two peaks, got {found:?}");
        assert!(
            (found[0] - first).abs() < 0.5 * step && (found[1] - last).abs() < 0.5 * step,
            "expected peaks at {:.2}° and {:.2}°, got {:?}",
            first.to_degrees(),
            last.to_degrees(),
            found.iter().map(|a| a.to_degrees()).collect::<Vec<_>>()
        );
    }
}
