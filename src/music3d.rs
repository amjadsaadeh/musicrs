//! 3-D MUSIC: direction of arrival as azimuth and elevation.
//!
//! [`MusicEstimator3d`] is the counterpart of [`MusicEstimator`](crate::MusicEstimator)
//! for arrays whose microphones are not all on one plane (or for searching elevation on a
//! planar array). The covariance, eigendecomposition and noise subspace are the same as in
//! 2-D; only the steering vectors and the search grid differ.
//!
//! Directions are given as
//! * **azimuth**: the angle in the x-y plane from +x towards +y, as in 2-D;
//! * **elevation**: the angle from the x-y plane towards +z, in [−π/2, π/2].
//!
//! The unit vector towards a source is
//! `(cos el · cos az, cos el · sin az, sin el)`, and microphone *j* at `pⱼ` hears a far-field
//! source `τⱼ = pⱼ · u / c` seconds before the origin does.
//!
//! What an array can resolve depends on its shape: a planar array cannot tell a source
//! above its plane from its mirror image below (search one hemisphere), and a linear array
//! only finds the angle between the source and its axis.

use std::collections::VecDeque;
use std::f64::consts::{FRAC_PI_2, PI};

use nalgebra::{DMatrix, DVector};
use num_complex::Complex;

use crate::error::{MusicError, Result};
use crate::music::{
    SPEED_OF_SOUND, check_frequency, check_not_silent, check_rows, check_sample_rate,
    check_sources, complex_data_covariance, music_score, noise_subspace, real_data_covariance,
    steering_from_delays,
};

type Cplx = Complex<f64>;
type CMatrix = DMatrix<Cplx>;
type CVec = DVector<Cplx>;

/// Slack, in radians, for interval ends that are meant to be exactly 2π apart or exactly
/// on a pole but carry a rounding error (the same tolerance as the 2-D estimator).
const ANGLE_TOLERANCE: f64 = 1e-9;

/// A direction of arrival, in radians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Direction {
    /// Angle in the x-y plane, from +x towards +y.
    pub azimuth: f64,
    /// Angle from the x-y plane towards +z, in [−π/2, π/2].
    pub elevation: f64,
}

impl Direction {
    /// The unit vector pointing towards the source:
    /// `(cos el · cos az, cos el · sin az, sin el)`.
    pub fn unit_vector(&self) -> [f64; 3] {
        let (cos_el, sin_el) = (self.elevation.cos(), self.elevation.sin());
        [
            cos_el * self.azimuth.cos(),
            cos_el * self.azimuth.sin(),
            sin_el,
        ]
    }
}

/// A 3-D MUSIC pseudo-spectrum over the azimuth × elevation grid.
#[derive(Debug, Clone, PartialEq)]
pub struct Spectrum3d {
    /// Scores, one row per elevation and one column per azimuth:
    /// `values[(i, j)]` belongs to `elevations[i]` and `azimuths[j]`.
    pub values: DMatrix<f64>,
    /// The azimuth grid (radians).
    pub azimuths: Vec<f64>,
    /// The elevation grid (radians).
    pub elevations: Vec<f64>,
}

/// MUSIC estimator for directions in 3-D (azimuth and elevation).
///
/// As with [`MusicEstimator`](crate::MusicEstimator), the fields are public and are checked
/// where they are used, not in [`new`](Self::new).
///
/// # Examples
/// ```
/// use musicrs::MusicEstimator3d;
/// use nalgebra::DMatrix;
/// use std::f64::consts::PI;
///
/// // Tetrahedral array, 1° grid over the whole sphere
/// let a = 0.03;
/// let est = MusicEstimator3d::new(
///     vec![[a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]],
///     16000.0,
/// );
/// // A 1 kHz far-field source at azimuth 40°, elevation 25°
/// let (az, el) = (40.0_f64.to_radians(), 25.0_f64.to_radians());
/// let u = [el.cos() * az.cos(), el.cos() * az.sin(), el.sin()];
/// let data = DMatrix::from_fn(4, 4000, |j, t| {
///     let p = est.antenna_positions[j];
///     let advance = (p[0] * u[0] + p[1] * u[1] + p[2] * u[2]) / 343.2;
///     (2.0 * PI * 1000.0 * (t as f64 / 16000.0 + advance)).cos()
/// });
/// let found = est.estimate_doa_from_real(&data, 1, 1000.0)?;
/// assert!((found[0].azimuth.to_degrees() - 40.0).abs() < 1.0);
/// assert!((found[0].elevation.to_degrees() - 25.0).abs() < 1.0);
/// # Ok::<(), musicrs::MusicError>(())
/// ```
#[derive(Debug, Clone)]
pub struct MusicEstimator3d {
    /// Antenna (microphone) positions as `[x, y, z]` in metres.
    pub antenna_positions: Vec<[f64; 3]>,
    /// Number of azimuths in the search grid.
    pub azimuth_steps: usize,
    /// Azimuth range `(start, end)` in radians, as `MusicEstimator::search_interval`:
    /// the grid includes `start` and excludes `end`, the span is at most 2π, and a span of
    /// exactly 2π is the full circle (its ends are neighbours when picking peaks). The start
    /// may be any finite angle, e.g. `(-π, π)`; estimates are reported inside the range.
    pub azimuth_interval: (f64, f64),
    /// Number of elevations in the search grid.
    pub elevation_steps: usize,
    /// Elevation range `(start, end)` in radians, within [−π/2, π/2]. It is split into
    /// `elevation_steps` equal cells and the grid holds their centres, so neither end is on the
    /// grid: for the whole sphere the poles, where every azimuth is the same direction, are
    /// never evaluated. Its ends are open when picking peaks.
    pub elevation_interval: (f64, f64),
    /// Sample rate of the input signal in Hz, used for the steering vector's Nyquist
    /// handling (see [`MusicEstimator::steering_vector`](crate::MusicEstimator::steering_vector)).
    pub sample_rate: f64,
}

impl MusicEstimator3d {
    /// Creates an estimator for the array at `antenna_positions` (`[x, y, z]` in metres) with a
    /// 1° grid over the whole sphere: 360 azimuths over [0, 2π) and 180 elevations over
    /// [−π/2, π/2] (at −89.5°, −88.5°, …, 89.5°). Change the public fields for another grid.
    pub fn new(antenna_positions: Vec<[f64; 3]>, sample_rate: f64) -> Self {
        Self {
            antenna_positions,
            azimuth_steps: 360,
            azimuth_interval: (0.0, 2.0 * PI),
            elevation_steps: 180,
            elevation_interval: (-FRAC_PI_2, FRAC_PI_2),
            sample_rate,
        }
    }

    /// Number of antennas in the array.
    #[inline]
    pub fn n_antennas(&self) -> usize {
        self.antenna_positions.len()
    }

    /// Whether the azimuth range spans exactly the full circle.
    fn covers_full_circle(&self) -> bool {
        let (start, end) = self.azimuth_interval;
        ((end - start).abs() - 2.0 * PI).abs() <= ANGLE_TOLERANCE
    }

    /// The azimuth grid: `azimuth_steps` angles from `start` (included) to `end` (excluded),
    /// computed exactly as the 2-D grid.
    fn azimuth_grid(&self) -> Result<Vec<f64>> {
        let (start, end) = self.azimuth_interval;
        let steps = self.azimuth_steps;
        let usable = start.is_finite() && end.is_finite() && start < end;
        if steps == 0 || !usable || end - start > 2.0 * PI + ANGLE_TOLERANCE {
            return Err(MusicError::InvalidAngleGrid {
                angle_steps_num: steps,
                search_interval: self.azimuth_interval,
            });
        }
        Ok((0..steps)
            .map(|i| start + (end - start) * i as f64 / steps as f64)
            .collect())
    }

    /// The elevation grid: the centres of `elevation_steps` equal cells of the interval.
    fn elevation_grid(&self) -> Result<Vec<f64>> {
        let (start, end) = self.elevation_interval;
        let steps = self.elevation_steps;
        let usable = start.is_finite()
            && end.is_finite()
            && start < end
            && start >= -FRAC_PI_2 - ANGLE_TOLERANCE
            && end <= FRAC_PI_2 + ANGLE_TOLERANCE;
        if steps == 0 || !usable {
            return Err(MusicError::InvalidElevationGrid {
                elevation_steps: steps,
                elevation_interval: self.elevation_interval,
            });
        }
        Ok((0..steps)
            .map(|i| start + (end - start) * (i as f64 + 0.5) / steps as f64)
            .collect())
    }

    /// Checks a finite, positive sample rate and finite antenna positions.
    fn check_array(&self) -> Result<()> {
        check_sample_rate(self.sample_rate)?;
        match self
            .antenna_positions
            .iter()
            .position(|p| !p.iter().all(|c| c.is_finite()))
        {
            Some(index) => Err(MusicError::InvalidAntennaPosition { index }),
            None => Ok(()),
        }
    }

    /// The steering vector for a far-field source in `direction` at `frequency` (Hz).
    ///
    /// Element *j* is `exp(+i·2π·f·τⱼ)` with `τⱼ = pⱼ · u / c`, `u` the
    /// [unit vector](Direction::unit_vector) towards the source; the sign and Nyquist
    /// conventions are those of [`MusicEstimator::steering_vector`](crate::MusicEstimator::steering_vector).
    /// For an array in the x-y plane and elevation 0 the two are identical.
    pub fn steering_vector(&self, direction: Direction, frequency: f64) -> CVec {
        let mut out = vec![Cplx::new(0.0, 0.0); self.n_antennas()];
        self.steering_vector_into(direction.unit_vector(), frequency, &mut out);
        CVec::from_vec(out)
    }

    fn steering_vector_into(&self, u: [f64; 3], frequency: f64, out: &mut [Cplx]) {
        let delays = self
            .antenna_positions
            .iter()
            .map(|p| (p[0] * u[0] + p[1] * u[1] + p[2] * u[2]) / SPEED_OF_SOUND);
        steering_from_delays(delays, frequency, self.sample_rate, out);
    }

    /// The noise subspace of the covariance `cov`, as
    /// [`MusicEstimator::noise_subspace`](crate::MusicEstimator::noise_subspace) (same errors).
    pub fn noise_subspace(&self, cov: &CMatrix, n_sources: usize) -> Result<CMatrix> {
        noise_subspace(self.n_antennas(), cov, n_sources)
    }

    /// The MUSIC pseudo-spectrum `1 / ‖Eₙᴴ · a(az, el)‖²` over the azimuth × elevation grid.
    ///
    /// # Errors
    /// - [`MusicError::RowCountMismatch`] if `noise_mat` does not have one row per antenna.
    /// - [`MusicError::InvalidFrequency`] unless `frequency` is finite and positive.
    /// - [`MusicError::InvalidAngleGrid`] for an unusable azimuth grid,
    ///   [`MusicError::InvalidElevationGrid`] for an unusable elevation grid,
    ///   [`MusicError::InvalidSampleRate`] or [`MusicError::InvalidAntennaPosition`].
    pub fn spectrum(&self, noise_mat: &CMatrix, frequency: f64) -> Result<Spectrum3d> {
        check_rows(self.n_antennas(), noise_mat.nrows())?;
        self.check_array()?;
        check_frequency(frequency)?;
        let azimuths = self.azimuth_grid()?;
        let elevations = self.elevation_grid()?;

        let az_trig: Vec<(f64, f64)> = azimuths.iter().map(|a| (a.cos(), a.sin())).collect();
        let mut a = vec![Cplx::new(0.0, 0.0); self.n_antennas()];
        let mut values = DMatrix::zeros(elevations.len(), azimuths.len());
        for (i, el) in elevations.iter().enumerate() {
            let (cos_el, sin_el) = (el.cos(), el.sin());
            for (j, &(cos_az, sin_az)) in az_trig.iter().enumerate() {
                let u = [cos_el * cos_az, cos_el * sin_az, sin_el];
                self.steering_vector_into(u, frequency, &mut a);
                values[(i, j)] = music_score(noise_mat, &a);
            }
        }
        Ok(Spectrum3d {
            values,
            azimuths,
            elevations,
        })
    }

    /// Estimates the directions of `n_sources` uncorrelated narrowband sources from complex
    /// array data (n_antennas × n_samples), highest spectrum peak first.
    ///
    /// # Errors
    /// As [`MusicEstimator::estimate_doa`](crate::MusicEstimator::estimate_doa), plus
    /// [`MusicError::InvalidElevationGrid`].
    pub fn estimate_doa(
        &self,
        data: &CMatrix,
        n_sources: usize,
        frequency: f64,
    ) -> Result<Vec<Direction>> {
        check_rows(self.n_antennas(), data.nrows())?;
        check_sources(n_sources, self.n_antennas())?;
        self.check_array()?;
        check_frequency(frequency)?;
        let cov = complex_data_covariance(data)?;
        self.doa_from_covariance(&cov, n_sources, frequency)
    }

    /// Estimates directions from real-valued microphone data (n_antennas × n_samples),
    /// with the same preprocessing as
    /// [`MusicEstimator::estimate_doa_from_real`](crate::MusicEstimator::estimate_doa_from_real):
    /// each channel's mean is removed and the covariance of the analytic signals is used.
    ///
    /// # Errors
    /// As [`MusicEstimator::estimate_doa_from_real`](crate::MusicEstimator::estimate_doa_from_real),
    /// plus [`MusicError::InvalidElevationGrid`].
    pub fn estimate_doa_from_real(
        &self,
        data: &DMatrix<f64>,
        n_sources: usize,
        frequency: f64,
    ) -> Result<Vec<Direction>> {
        check_rows(self.n_antennas(), data.nrows())?;
        check_sources(n_sources, self.n_antennas())?;
        self.check_array()?;
        check_frequency(frequency)?;
        let cov = real_data_covariance(data)?;
        self.doa_from_covariance(&cov, n_sources, frequency)
    }

    fn doa_from_covariance(
        &self,
        cov: &CMatrix,
        n_sources: usize,
        frequency: f64,
    ) -> Result<Vec<Direction>> {
        check_not_silent(cov)?;
        let noise_mat = self.noise_subspace(cov, n_sources)?;
        let spectrum = self.spectrum(&noise_mat, frequency)?;
        Ok(find_peaks_2d(
            &spectrum,
            self.azimuth_interval,
            n_sources,
            self.covers_full_circle(),
        ))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Peak detection on the azimuth × elevation grid
// ─────────────────────────────────────────────────────────────────────────────

/// The value at fractional index `sum / count` of a regular `grid` whose index is known to
/// lie within it: a grid point when the index is whole, else interpolated between the two
/// neighbouring points.
fn grid_at(grid: &[f64], sum: i64, count: i64) -> f64 {
    let lo = sum.div_euclid(count) as usize;
    let rem = sum.rem_euclid(count);
    if rem == 0 {
        grid[lo]
    } else {
        let frac = rem as f64 / count as f64;
        grid[lo] + (grid[lo + 1] - grid[lo]) * frac
    }
}

/// The `n` highest peaks of `spectrum`, highest first (equal heights by elevation, then
/// azimuth).
///
/// As in 2-D, a peak is a connected run of equal scores that is strictly higher than every
/// point bordering it, reported at its centre. A grid point's neighbours are the points in
/// its own row and the rows above and below that are at most `k` azimuth steps away, `k`
/// the number of azimuth steps that span one elevation step at that elevation (1 near the
/// equator of a grid with equal steps, the 8-neighbourhood; more towards the poles). With `circular`, the first and last azimuths are neighbours and a run may
/// cross the seam; otherwise both azimuth ends are open. Elevation ends are always open.
/// Grid points next to a pole are not treated as neighbours across it. A run that goes all
/// the way round in azimuth (an array that cannot tell azimuths apart) has no azimuth of
/// its own and is reported at the first grid azimuth. A spectrum that is constant over a
/// grid of two or more points has no peak; a one-point grid with an open azimuth range
/// peaks at its only point.
fn find_peaks_2d(
    spectrum: &Spectrum3d,
    azimuth_interval: (f64, f64),
    n: usize,
    circular: bool,
) -> Vec<Direction> {
    let values = &spectrum.values;
    let (n_el, n_az) = values.shape();
    if n_el * n_az == 0 || n == 0 {
        return Vec::new();
    }
    let (az_start, az_end) = azimuth_interval;
    let span = az_end - az_start;

    // How many azimuth steps either side count as neighbours in each row. One azimuth step
    // spans less arc the further a row is from the equator (by cos el), so the grid is
    // stretched in azimuth there: reach as many steps as span one elevation step, so that
    // one source does not show up as several peaks a few columns apart. On one row, and
    // near the equator of a grid with equal steps, this is the usual 8-neighbourhood. The
    // reach is capped below half the circle, so that each step has one shortest offset.
    let az_step = span / n_az as f64;
    let max_reach = if circular { (n_az - 1) / 2 } else { n_az - 1 }.max(1);
    let reach: Vec<usize> = spectrum
        .elevations
        .iter()
        .map(|el| {
            if n_el < 2 {
                return 1;
            }
            let el_step = spectrum.elevations[1] - spectrum.elevations[0];
            let ratio = el_step / (az_step * el.cos());
            if ratio.is_finite() && ratio > 1.0 {
                (ratio.round() as usize).clamp(1, max_reach)
            } else {
                1
            }
        })
        .collect();

    // Neighbours of (i, j) as (row, column, column step); the step tracks the unwrapped
    // azimuth index of a run that crosses the seam.
    let neighbours = |i: usize, j: usize| {
        let mut out: Vec<(usize, usize, i64)> = Vec::new();
        for di in -1i64..=1 {
            let ni = i as i64 + di;
            if ni < 0 || ni >= n_el as i64 {
                continue;
            }
            let k = reach[i].max(reach[ni as usize]) as i64;
            for dj in -k..=k {
                if di == 0 && dj == 0 {
                    continue;
                }
                let nj = j as i64 + dj;
                let nj = if circular {
                    nj.rem_euclid(n_az as i64)
                } else if (0..n_az as i64).contains(&nj) {
                    nj
                } else {
                    continue;
                };
                out.push((ni as usize, nj as usize, dj));
            }
        }
        out
    };

    let mut visited = vec![false; n_el * n_az];
    let mut queue: VecDeque<(usize, usize, i64)> = VecDeque::new();
    let mut peaks: Vec<(f64, Direction)> = Vec::new();
    for i0 in 0..n_el {
        for j0 in 0..n_az {
            if visited[i0 * n_az + j0] {
                continue;
            }
            let v = values[(i0, j0)];
            visited[i0 * n_az + j0] = true;
            queue.push_back((i0, j0, j0 as i64));
            let (mut count, mut sum_i, mut sum_j) = (0i64, 0i64, 0i64);
            let mut columns = vec![false; n_az];
            let (mut is_peak, mut bordered) = (true, false);
            while let Some((i, j, unwrapped)) = queue.pop_front() {
                count += 1;
                sum_i += i as i64;
                sum_j += unwrapped;
                columns[j] = true;
                for (ni, nj, dj) in neighbours(i, j) {
                    let w = values[(ni, nj)];
                    if w == v {
                        if !visited[ni * n_az + nj] {
                            visited[ni * n_az + nj] = true;
                            queue.push_back((ni, nj, unwrapped + dj));
                        }
                    } else {
                        bordered = true;
                        // Not strictly higher (or a NaN): not a peak
                        if v.partial_cmp(&w) != Some(std::cmp::Ordering::Greater) {
                            is_peak = false;
                        }
                    }
                }
            }
            let only_point = n_el * n_az == 1 && !circular;
            if !(is_peak && (bordered || only_point)) {
                continue;
            }

            let elevation = grid_at(&spectrum.elevations, sum_i, count);
            let azimuth = if circular && columns.iter().all(|&c| c) {
                spectrum.azimuths[0]
            } else {
                // Bring the mean unwrapped column into 0..n_az.
                let n = n_az as i64;
                let shift = sum_j.div_euclid(count).div_euclid(n) * n;
                let sum = sum_j - shift * count;
                let lo = sum.div_euclid(count) as usize;
                if lo + 1 < n_az || sum.rem_euclid(count) == 0 {
                    grid_at(&spectrum.azimuths, sum, count)
                } else {
                    // Between the last grid azimuth and the first one, across the seam.
                    let frac = sum.rem_euclid(count) as f64 / count as f64;
                    let az = spectrum.azimuths[lo] + span / n_az as f64 * frac;
                    if az >= az_end { az - span } else { az }
                }
            };
            peaks.push((v, Direction { azimuth, elevation }));
        }
    }

    peaks.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then(a.1.elevation.total_cmp(&b.1.elevation))
            .then(a.1.azimuth.total_cmp(&b.1.azimuth))
    });
    peaks.into_iter().take(n).map(|(_, d)| d).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MusicEstimator;

    /// Grid step of the test spectra: small, so that the grid is close to the equator and
    /// every row's neighbourhood is the 8-neighbourhood.
    const STEP: f64 = 0.001;

    /// A spectrum over a grid with azimuths and elevations 0, STEP, 2·STEP, …
    fn spectrum(rows: &[&[f64]]) -> Spectrum3d {
        let (n_el, n_az) = (rows.len(), rows[0].len());
        Spectrum3d {
            values: DMatrix::from_fn(n_el, n_az, |i, j| rows[i][j]),
            azimuths: (0..n_az).map(|j| j as f64 * STEP).collect(),
            elevations: (0..n_el).map(|i| i as f64 * STEP).collect(),
        }
    }

    /// The peaks of `s` as (azimuth, elevation) in grid steps.
    fn peaks(s: &Spectrum3d, n: usize, circular: bool) -> Vec<(f64, f64)> {
        let end = s.azimuths.len() as f64 * STEP;
        let steps = |x: f64| (x / STEP * 1e6).round() / 1e6;
        find_peaks_2d(s, (0.0, end), n, circular)
            .iter()
            .map(|d| (steps(d.azimuth), steps(d.elevation)))
            .collect()
    }

    #[test]
    fn steering_vector_matches_2d_in_the_plane() {
        let positions = [
            [0.113, 0.02],
            [-0.036, -0.05],
            [-0.076, 0.0],
            [-0.113, 0.07],
        ];
        let est_2d = MusicEstimator::new(positions.to_vec(), 720, (0.0, 2.0 * PI), 16000.0);
        let est_3d = MusicEstimator3d::new(
            positions.iter().map(|p| [p[0], p[1], 0.0]).collect(),
            16000.0,
        );
        for freq in [300.0, 1000.0, 7999.8, 8000.0, 9100.0] {
            for az in [0.0, 0.7, 2.0, 4.5] {
                let d = Direction {
                    azimuth: az,
                    elevation: 0.0,
                };
                assert_eq!(
                    est_3d.steering_vector(d, freq),
                    est_2d.steering_vector(az, freq),
                    "{freq} Hz, azimuth {az}"
                );
            }
        }
    }

    #[test]
    fn steering_vector_uses_elevation() {
        // One microphone 10 cm up: its phase is 2π f · 0.1 sin(el) / c
        let est = MusicEstimator3d::new(vec![[0.0, 0.0, 0.1]], 16000.0);
        for el in [-1.2, -0.3, 0.0, 0.5, FRAC_PI_2] {
            let sv = est.steering_vector(
                Direction {
                    azimuth: 1.0,
                    elevation: el,
                },
                1000.0,
            );
            let expected =
                Cplx::from_polar(1.0, 2.0 * PI * 1000.0 * 0.1 * el.sin() / SPEED_OF_SOUND);
            assert!((sv[0] - expected).norm() < 1e-12, "elevation {el}");
        }
    }

    #[test]
    fn unit_vector_points_at_the_source() {
        let u = Direction {
            azimuth: PI / 2.0,
            elevation: 0.0,
        }
        .unit_vector();
        assert!(u[0].abs() < 1e-15 && (u[1] - 1.0).abs() < 1e-15 && u[2] == 0.0);
        let up = Direction {
            azimuth: 2.0,
            elevation: FRAC_PI_2,
        }
        .unit_vector();
        assert!((up[2] - 1.0).abs() < 1e-15 && up[0].abs() < 1e-15 && up[1].abs() < 1e-15);
    }

    #[test]
    fn elevation_grid_holds_cell_centres() {
        let est = MusicEstimator3d::new(vec![[0.0; 3]; 2], 16000.0);
        let grid = est.elevation_grid().unwrap();
        assert_eq!(grid.len(), 180);
        assert!((grid[0].to_degrees() + 89.5).abs() < 1e-9);
        assert!((grid[179].to_degrees() - 89.5).abs() < 1e-9);
        assert!(grid.iter().all(|e| e.abs() < FRAC_PI_2));
        // A one-cell interval symmetric around 0 holds exactly 0
        let flat = MusicEstimator3d {
            elevation_steps: 1,
            elevation_interval: (-0.01, 0.01),
            ..est
        };
        assert_eq!(flat.elevation_grid().unwrap(), vec![0.0]);
    }

    #[test]
    fn find_peaks_2d_returns_top_n() {
        let s = spectrum(&[
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[0.0, 5.0, 0.0, 0.0, 2.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 9.0, 0.0, 0.0],
        ]);
        assert_eq!(peaks(&s, 2, true), vec![(3.0, 3.0), (1.0, 1.0)]);
        assert_eq!(
            peaks(&s, 10, true),
            vec![(3.0, 3.0), (1.0, 1.0), (4.0, 1.0)]
        );
        assert!(peaks(&s, 0, true).is_empty());
    }

    #[test]
    fn find_peaks_2d_needs_diagonal_neighbours_lower() {
        // (1, 1) is higher than its row and column but not than its diagonal neighbour
        let s = spectrum(&[&[0.0, 0.0, 0.0], &[0.0, 5.0, 0.0], &[0.0, 0.0, 6.0]]);
        assert_eq!(peaks(&s, 5, false), vec![(2.0, 2.0)]);
    }

    #[test]
    fn find_peaks_2d_wraps_azimuth_only_on_the_circle() {
        let s = spectrum(&[&[3.0, 1.0, 0.0, 0.0, 1.0, 2.0]]);
        // On the circle, 2.0 at the last azimuth is next to 3.0 at the first: one peak
        assert_eq!(peaks(&s, 5, true), vec![(0.0, 0.0)]);
        // On an open range both ends rise: two peaks
        assert_eq!(peaks(&s, 5, false), vec![(0.0, 0.0), (5.0, 0.0)]);
    }

    #[test]
    fn find_peaks_2d_plateau_reports_centre() {
        let s = spectrum(&[
            &[0.0, 0.0, 0.0, 0.0, 0.0],
            &[0.0, 4.0, 4.0, 0.0, 0.0],
            &[0.0, 4.0, 4.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0, 0.0],
        ]);
        assert_eq!(peaks(&s, 5, false), vec![(1.5, 1.5)]);
    }

    #[test]
    fn find_peaks_2d_plateau_across_the_seam() {
        // Equal scores at the last and the first azimuth: the peak lies between them, half
        // a step past the last grid azimuth, still inside [0, 6)
        let s = spectrum(&[&[4.0, 1.0, 0.0, 0.0, 1.0, 4.0], &[0.0; 6]]);
        assert_eq!(peaks(&s, 5, true), vec![(5.5, 0.0)]);
        // Three equal points across the seam: the middle one, the first azimuth
        let s = spectrum(&[&[4.0, 4.0, 0.0, 0.0, 1.0, 4.0]]);
        assert_eq!(peaks(&s, 5, true), vec![(0.0, 0.0)]);
    }

    #[test]
    fn find_peaks_2d_elevation_ends_are_open() {
        // Rising towards the top row: the peak is on the edge of the grid
        let s = spectrum(&[&[0.0, 1.0, 0.0], &[0.0, 2.0, 0.0], &[0.0, 3.0, 0.0]]);
        assert_eq!(peaks(&s, 5, false), vec![(1.0, 2.0)]);
    }

    #[test]
    fn find_peaks_2d_ring_reports_first_azimuth() {
        // Constant in azimuth, highest in the middle row: one peak with no azimuth of its own
        let s = spectrum(&[&[1.0; 4], &[2.0; 4], &[1.0; 4]]);
        assert_eq!(peaks(&s, 5, true), vec![(0.0, 1.0)]);
    }

    #[test]
    fn find_peaks_2d_constant_and_single_point() {
        let flat = spectrum(&[&[1.0; 4], &[1.0; 4]]);
        assert!(peaks(&flat, 5, true).is_empty());
        assert!(peaks(&flat, 5, false).is_empty());
        let one = spectrum(&[&[1.0]]);
        assert_eq!(peaks(&one, 5, false), vec![(0.0, 0.0)]);
        assert!(peaks(&one, 5, true).is_empty());
    }

    #[test]
    fn find_peaks_2d_infinite_scores_and_ties() {
        let s = spectrum(&[&[0.0, f64::INFINITY, 0.0, 0.0], &[0.0, 0.0, 0.0, 7.0]]);
        assert_eq!(peaks(&s, 5, false), vec![(1.0, 0.0), (3.0, 1.0)]);
        // Equal heights: lower elevation first, then lower azimuth
        let s = spectrum(&[
            &[0.0, 0.0, 0.0, 0.0, 0.0],
            &[3.0, 0.0, 0.0, 3.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0, 0.0],
            &[0.0, 3.0, 0.0, 0.0, 0.0],
        ]);
        assert_eq!(
            peaks(&s, 5, false),
            vec![(0.0, 1.0), (3.0, 1.0), (1.0, 3.0)]
        );
    }

    #[test]
    fn find_peaks_2d_reaches_further_in_azimuth_towards_the_poles() {
        // 1° grid at elevations 59.5° and 60.5°, where one azimuth step is half an elevation
        // step: two maxima two columns apart on a diagonal ridge are one source.
        let az: Vec<f64> = (0..8).map(|j| (256.0 + j as f64).to_radians()).collect();
        let s = Spectrum3d {
            values: DMatrix::from_row_slice(
                2,
                8,
                &[
                    0.0, 1.0, 2.0, 9.0, 3.0, 2.0, 1.0, 0.0, //
                    0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 1.0, 0.0,
                ],
            ),
            azimuths: az.clone(),
            elevations: vec![59.5_f64.to_radians(), 60.5_f64.to_radians()],
        };
        let found = find_peaks_2d(&s, (az[0], az[0] + 8f64.to_radians()), 5, false);
        assert_eq!(found.len(), 1, "{found:?}");
        assert!((found[0].azimuth.to_degrees() - 259.0).abs() < 1e-9);
        // At the equator the same values are two separate peaks
        let equator = Spectrum3d {
            elevations: vec![-0.5_f64.to_radians(), 0.5_f64.to_radians()],
            ..s
        };
        assert_eq!(
            find_peaks_2d(&equator, (az[0], az[0] + 8f64.to_radians()), 5, false).len(),
            2
        );
    }

    #[test]
    fn find_peaks_2d_one_row_matches_1d() {
        // With one elevation the 2-D finder is the 1-D one
        let scores = [1.0, 3.0, 3.0, 0.0, 2.0, 5.0, 0.5, 5.0];
        let angles: Vec<f64> = (0..8).map(|i| i as f64 * 0.5).collect();
        let s = Spectrum3d {
            values: DMatrix::from_row_slice(1, 8, &scores),
            azimuths: angles.clone(),
            elevations: vec![0.0],
        };
        for circular in [false, true] {
            let one_d = crate::music::find_peaks(&scores, &angles, 8, circular);
            let two_d: Vec<f64> = find_peaks_2d(&s, (0.0, 4.0), 8, circular)
                .iter()
                .map(|d| d.azimuth)
                .collect();
            assert_eq!(two_d, one_d, "circular {circular}");
        }
    }
}
