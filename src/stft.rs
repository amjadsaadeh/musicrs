//! Short-Time Fourier Transform and analytic signal computation.

use num_complex::Complex;
use rustfft::FftPlanner;

use crate::error::{MusicError, Result};

/// Computes the analytic signal of a real-valued input using the Hilbert transform.
///
/// The analytic signal `z(t) = x(t) + j·H{x(t)}` has a one-sided spectrum; its magnitude is the
/// instantaneous envelope and its argument is the instantaneous phase.
///
/// # Algorithm
/// 1. Forward FFT of the real input.
/// 2. Apply the one-sided spectral window H: `H[0] = 1`, `H[k] = 2` for the positive
///    frequencies `1 ≤ k < (N+1)/2`, `H[N/2] = 1` for the Nyquist bin of even N, and `H = 0` for
///    the negative frequencies `k > N/2` (as in `scipy.signal.hilbert`).
/// 3. Inverse FFT → analytic signal.
///
/// # Examples
/// ```
/// use musicrs::analytic_signal;
/// use std::f64::consts::PI;
/// let n = 256;
/// let cos: Vec<f64> = (0..n).map(|i| (2.0 * PI * 10.0 * i as f64 / n as f64).cos()).collect();
/// let z = analytic_signal(&cos);
/// // Magnitude should be approximately 1 everywhere (after transients settle)
/// for &c in &z[10..n-10] {
///     assert!((c.norm() - 1.0).abs() < 0.01);
/// }
/// ```
pub fn analytic_signal(data: &[f64]) -> Vec<Complex<f64>> {
    analytic_signal_with(data, &mut FftPlanner::new())
}

/// [`analytic_signal`] with a caller-owned planner, so that several signals of the same
/// length (one per channel) share its cached FFT plans instead of rebuilding them.
pub(crate) fn analytic_signal_with(
    data: &[f64],
    planner: &mut FftPlanner<f64>,
) -> Vec<Complex<f64>> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }

    // Forward FFT
    let fft_forward = planner.plan_fft_forward(n);
    let mut spectrum: Vec<Complex<f64>> = data.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft_forward.process(&mut spectrum);

    // One-sided spectral window. DC is unchanged; so is the Nyquist bin N/2, which only
    // exists for even N. Odd N has no Nyquist bin: bin (N-1)/2 is a positive frequency.
    for sample in &mut spectrum[1..n.div_ceil(2)] {
        *sample *= 2.0;
    }
    for sample in &mut spectrum[n / 2 + 1..] {
        *sample = Complex::new(0.0, 0.0);
    }

    // Inverse FFT
    let fft_inverse = planner.plan_fft_inverse(n);
    fft_inverse.process(&mut spectrum);

    // Normalise
    let norm = n as f64;
    spectrum.iter().map(|c| c / norm).collect()
}

/// Computes the Short-Time Fourier Transform of a real-valued signal.
///
/// Returns a matrix of shape `[freq_bins][num_windows]` where
/// `freq_bins = nfft / 2 + 1` (single-sided spectrum via real FFT).
///
/// # Arguments
/// * `data`       – Real-valued input samples.
/// * `nfft`       – Window (FFT) size.
/// * `no_overlap` – Number of samples shared between consecutive windows.
///
/// Data shorter than `nfft` gives `nfft / 2 + 1` empty bins.
///
/// # Errors
/// [`MusicError::InvalidNfft`] if `nfft` is 0, [`MusicError::InvalidOverlap`] if
/// `no_overlap >= nfft`.
///
/// # Examples
/// ```
/// use musicrs::stft;
/// use std::f64::consts::PI;
/// let n = 512;
/// let freq_bin = 16usize;
/// let sr = 512.0_f64;
/// let sine: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * freq_bin as f64 * i as f64 / sr).sin())
///     .collect();
/// let result = stft(&sine, 64, 0)?;
/// assert_eq!(result.len(), 64 / 2 + 1); // bins 0..=32
/// assert_eq!(result[0].len(), 512 / 64); // non-overlapping windows
/// // 16 cycles per 512 samples is 2 cycles per 64-sample window: bin 2 in every window
/// for w in 0..result[0].len() {
///     let peak = (0..result.len())
///         .max_by(|&a, &b| result[a][w].norm().total_cmp(&result[b][w].norm()))
///         .unwrap();
///     assert_eq!(peak, 2);
/// }
/// # Ok::<(), musicrs::MusicError>(())
/// ```
pub fn stft(data: &[f64], nfft: usize, no_overlap: usize) -> Result<Vec<Vec<Complex<f64>>>> {
    stft_with(data, nfft, no_overlap, &mut FftPlanner::new())
}

/// [`stft`] with a caller-owned planner, so that several signals (one per channel) share
/// its cached FFT plan instead of rebuilding it.
pub(crate) fn stft_with(
    data: &[f64],
    nfft: usize,
    no_overlap: usize,
    planner: &mut FftPlanner<f64>,
) -> Result<Vec<Vec<Complex<f64>>>> {
    if nfft == 0 {
        return Err(MusicError::InvalidNfft);
    }
    if no_overlap >= nfft {
        return Err(MusicError::InvalidOverlap { no_overlap, nfft });
    }

    let step = nfft - no_overlap;
    let freq_bins = nfft / 2 + 1;
    // Use real-to-complex FFT plan (standard forward FFT on real-padded input)
    let fft = planner.plan_fft_forward(nfft);

    let num_windows = if data.len() >= nfft {
        (data.len() - nfft) / step + 1
    } else {
        0
    };

    // result[freq_bin][window_index]
    let mut result: Vec<Vec<Complex<f64>>> =
        vec![vec![Complex::new(0.0, 0.0); num_windows]; freq_bins];

    for (win_idx, start) in (0..).zip((0..).step_by(step).take(num_windows)) {
        let end = start + nfft;
        let mut buf: Vec<Complex<f64>> = data[start..end]
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        fft.process(&mut buf);
        for bin in 0..freq_bins {
            result[bin][win_idx] = buf[bin];
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    /// Pure cosine → analytic signal magnitude ≈ 1 (away from edges).
    #[test]
    fn analytic_signal_cosine_magnitude() {
        let n = 256usize;
        let f0 = 10.0_f64;
        let data: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64 / n as f64).cos())
            .collect();
        let z = analytic_signal(&data);
        // Skip 10 samples at each edge where Gibbs artefacts may occur
        for c in &z[10..n - 10] {
            assert_abs_diff_eq!(c.norm(), 1.0, epsilon = 0.02);
        }
    }

    /// Analytic signal of a cosine has imaginary part ≈ -sine (shifted by -π/2).
    #[test]
    fn analytic_signal_imag_is_negative_sine() {
        let n = 256usize;
        let f0 = 8.0_f64;
        let data: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64 / n as f64).cos())
            .collect();
        let z = analytic_signal(&data);
        for (i, zi) in z.iter().enumerate().take(n - 10).skip(10) {
            // H{cos(ωt)} = +sin(ωt), so z(t) = cos(ωt) + i·sin(ωt)
            let expected_imag = (2.0 * PI * f0 * i as f64 / n as f64).sin();
            assert_abs_diff_eq!(zi.im, expected_imag, epsilon = 0.02);
        }
    }

    /// Deterministic pseudo-random samples in [-1, 1) (LCG).
    fn noise(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                2.0 * (state >> 11) as f64 / (1u64 << 53) as f64 - 1.0
            })
            .collect()
    }

    /// The real part of the analytic signal is the input itself, for even and odd lengths.
    #[test]
    fn analytic_signal_real_part_is_input() {
        for n in [1, 2, 7, 64, 255] {
            let x = noise(n, n as u64);
            let z = analytic_signal(&x);
            assert_eq!(z.len(), n);
            for (zi, xi) in z.iter().zip(&x) {
                assert_abs_diff_eq!(zi.re, *xi, epsilon = 1e-12);
            }
        }
    }

    /// The analytic signal has no negative-frequency content, for even and odd lengths.
    #[test]
    fn analytic_signal_is_one_sided() {
        for n in [8, 9, 64, 255] {
            let mut z = analytic_signal(&noise(n, 1 + n as u64));
            FftPlanner::<f64>::new().plan_fft_forward(n).process(&mut z);
            for (k, c) in z.iter().enumerate().skip(n / 2 + 1) {
                assert!(
                    c.norm() < 1e-9,
                    "n = {n}: bin {k} has magnitude {}",
                    c.norm()
                );
            }
        }
    }

    /// A planner reused across signals gives bit-identical results to a fresh one per call.
    #[test]
    fn shared_planner_matches_fresh_planner() {
        let mut planner = FftPlanner::new();
        for seed in 0..3 {
            let x = noise(255, 100 + seed);
            assert_eq!(analytic_signal_with(&x, &mut planner), analytic_signal(&x));
            assert_eq!(
                stft_with(&x, 32, 8, &mut planner),
                stft(&x, 32, 8),
                "seed {seed}"
            );
        }
    }

    #[test]
    fn analytic_signal_empty_input() {
        assert!(analytic_signal(&[]).is_empty());
    }

    /// STFT column count matches expected number of windows.
    #[test]
    fn stft_window_count() {
        let data = vec![0.0f64; 100];
        let nfft = 16;
        let overlap = 8;
        let step = nfft - overlap; // 8
        let expected_windows = (data.len() - nfft) / step + 1; // (100 - 16) / 8 + 1 = 11
        let result = stft(&data, nfft, overlap).unwrap();
        assert_eq!(result[0].len(), expected_windows);
    }

    #[test]
    fn stft_rejects_zero_nfft() {
        assert_eq!(stft(&[0.0; 8], 0, 0), Err(MusicError::InvalidNfft));
    }

    #[test]
    fn stft_rejects_overlap_of_whole_window() {
        assert_eq!(
            stft(&[0.0; 8], 4, 4),
            Err(MusicError::InvalidOverlap {
                no_overlap: 4,
                nfft: 4
            })
        );
    }

    /// With overlap, window w starts at sample w·(nfft − no_overlap): its DC bin is the sum
    /// of exactly those samples.
    #[test]
    fn stft_overlapping_windows_start_at_step() {
        let data: Vec<f64> = (0..20).map(f64::from).collect();
        let (nfft, overlap) = (8, 5);
        let result = stft(&data, nfft, overlap).unwrap();
        assert_eq!(result[0].len(), 5); // (20 − 8) / 3 + 1
        for (w, dc) in result[0].iter().enumerate() {
            let start = w * (nfft - overlap);
            let expected: f64 = data[start..start + nfft].iter().sum();
            assert_abs_diff_eq!(dc.re, expected, epsilon = 1e-9);
            assert_abs_diff_eq!(dc.im, 0.0, epsilon = 1e-9);
        }
    }

    /// STFT freq_bins = nfft/2 + 1.
    #[test]
    fn stft_freq_bins() {
        let data = vec![0.0f64; 64];
        let nfft = 32;
        let result = stft(&data, nfft, 0).unwrap();
        assert_eq!(result.len(), nfft / 2 + 1);
    }

    /// STFT of a pure sine wave has a peak at the expected frequency bin.
    #[test]
    fn stft_peak_at_correct_bin() {
        let nfft = 64usize;
        let target_bin = 4usize; // frequency = target_bin * (sr / nfft)
        // Generate enough samples for several non-overlapping windows
        let n = nfft * 8;
        let data: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * target_bin as f64 * i as f64 / nfft as f64).sin())
            .collect();
        let result = stft(&data, nfft, 0).unwrap();
        // In each window, bin `target_bin` should dominate
        for win in 0..result[0].len() {
            let peak_bin = (0..result.len())
                .max_by(|&a, &b| {
                    result[a][win]
                        .norm()
                        .partial_cmp(&result[b][win].norm())
                        .unwrap()
                })
                .unwrap();
            assert_eq!(peak_bin, target_bin);
        }
    }
}
