//! Short-Time Fourier Transform and analytic signal computation.

use num_complex::Complex;
use rustfft::FftPlanner;

/// Computes the analytic signal of a real-valued input using the Hilbert transform.
///
/// The analytic signal `z(t) = x(t) + j·H{x(t)}` has a one-sided spectrum; its magnitude is the
/// instantaneous envelope and its argument is the instantaneous phase.
///
/// # Algorithm
/// 1. Forward FFT of the real input.
/// 2. Apply the one-sided spectral window H: `H[0] = 1`, `H[1..N/2] = 2`, `H[N/2] = 1` (for
///    even N), `H[N/2+1..] = 0`.
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
    let n = data.len();
    let mut planner = FftPlanner::<f64>::new();

    // Forward FFT
    let fft_forward = planner.plan_fft_forward(n);
    let mut spectrum: Vec<Complex<f64>> = data.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft_forward.process(&mut spectrum);

    // One-sided spectral window
    let half = n / 2;
    spectrum[0] *= 1.0; // DC – unchanged
    for k in 1..half {
        spectrum[k] *= 2.0;
    }
    // Nyquist bin (index half) is kept unchanged when N is even
    // Negative-frequency bins are zeroed
    for k in (half + 1)..n {
        spectrum[k] = Complex::new(0.0, 0.0);
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
/// * `nfft`       – Window (FFT) size. Must be ≥ 1.
/// * `no_overlap` – Number of samples shared between consecutive windows.
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
/// let result = stft(&sine, 64, 0);
/// // Each window should have a peak at `freq_bin` (relative to the window size)
/// assert!(!result.is_empty());
/// ```
pub fn stft(data: &[f64], nfft: usize, no_overlap: usize) -> Vec<Vec<Complex<f64>>> {
    assert!(nfft >= 1, "nfft must be at least 1");
    assert!(no_overlap < nfft, "no_overlap must be less than nfft");

    let step = nfft - no_overlap;
    let freq_bins = nfft / 2 + 1;
    let mut planner = FftPlanner::<f64>::new();
    // Use real-to-complex FFT plan (standard forward FFT on real-padded input)
    let fft = planner.plan_fft_forward(nfft);

    let num_windows = if data.len() >= nfft {
        (data.len() - nfft) / step + 1
    } else {
        0
    };

    // result[freq_bin][window_index]
    let mut result: Vec<Vec<Complex<f64>>> = vec![vec![Complex::new(0.0, 0.0); num_windows]; freq_bins];

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

    result
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
        for i in 10..n - 10 {
            // H{cos(ωt)} = +sin(ωt), so z(t) = cos(ωt) + i·sin(ωt)
            let expected_imag = (2.0 * PI * f0 * i as f64 / n as f64).sin();
            assert_abs_diff_eq!(z[i].im, expected_imag, epsilon = 0.02);
        }
    }

    /// STFT column count matches expected number of windows.
    #[test]
    fn stft_window_count() {
        let data = vec![0.0f64; 100];
        let nfft = 16;
        let overlap = 8;
        let step = nfft - overlap; // 8
        let expected_windows = (data.len() - nfft) / step + 1; // (100 - 16) / 8 + 1 = 11
        let result = stft(&data, nfft, overlap);
        assert_eq!(result[0].len(), expected_windows);
    }

    /// STFT freq_bins = nfft/2 + 1.
    #[test]
    fn stft_freq_bins() {
        let data = vec![0.0f64; 64];
        let nfft = 32;
        let result = stft(&data, nfft, 0);
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
        let result = stft(&data, nfft, 0);
        // In each window, bin `target_bin` should dominate
        for win in 0..result[0].len() {
            let peak_bin = (0..result.len())
                .max_by(|&a, &b| result[a][win].norm().partial_cmp(&result[b][win].norm()).unwrap())
                .unwrap();
            assert_eq!(peak_bin, target_bin);
        }
    }
}
