//! Short-Time Fourier Transform and analytic signal computation.

use std::cell::RefCell;
use std::sync::Arc;

use num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
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

/// How many real FFT plans [`real_fft_plan`] keeps per thread.
const REAL_PLAN_CACHE_SIZE: usize = 4;

thread_local! {
    /// The most recently used real forward FFT plans of this thread, most recent last.
    static REAL_PLANS: RefCell<Vec<Arc<dyn RealToComplex<f64>>>> = const { RefCell::new(Vec::new()) };
}

/// A real forward FFT plan of length `n`, reused across calls on the same thread.
///
/// Planning a long transform (and computing its twiddle factors) costs about as much as
/// running it a few times, and repeated calls on recordings of one length are the common
/// case. Only the last [`REAL_PLAN_CACHE_SIZE`] lengths are kept, so that a stream of
/// different lengths does not keep every plan alive.
///
/// Each thread that calls this keeps its plans (their twiddle factors take memory
/// proportional to the transform length) until the thread exits. With
/// many long-lived threads on long recordings, that is up to
/// [`REAL_PLAN_CACHE_SIZE`] plans per thread.
pub(crate) fn real_fft_plan(n: usize) -> Arc<dyn RealToComplex<f64>> {
    REAL_PLANS.with(|cache| {
        let mut cache = cache.borrow_mut();
        let plan = match cache.iter().position(|p| p.len() == n) {
            Some(i) => cache.remove(i),
            None => {
                if cache.len() == REAL_PLAN_CACHE_SIZE {
                    cache.remove(0);
                }
                RealFftPlanner::<f64>::new().plan_fft_forward(n)
            }
        };
        cache.push(Arc::clone(&plan));
        plan
    })
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
/// The FFT plan for `nfft` is kept in a per-thread cache of the last 4 transform lengths
/// (shared with [`MusicEstimator`](crate::MusicEstimator)), so repeated calls do not plan
/// again. A cached plan holds memory proportional to its length until it is evicted or the
/// thread exits.
///
/// # Errors
/// [`MusicError::InvalidNfft`] if `nfft` is 0, [`MusicError::InvalidOverlap`] if
/// `no_overlap >= nfft`, then [`MusicError::DataShorterThanWindow`] if `data` is shorter
/// than `nfft`. The data length is checked before the FFT is planned, so memory stays bounded
/// by the input size whatever `nfft` is.
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
    if nfft == 0 {
        return Err(MusicError::InvalidNfft);
    }
    if no_overlap >= nfft {
        return Err(MusicError::InvalidOverlap { no_overlap, nfft });
    }
    if data.len() < nfft {
        // Planning the FFT and allocating the bins are O(nfft): return before either.
        return Err(MusicError::DataShorterThanWindow {
            nfft,
            samples: data.len(),
        });
    }

    let step = nfft - no_overlap;
    let freq_bins = nfft / 2 + 1;
    // Real-to-complex FFT: bins 0..=nfft/2 of the complex FFT, at about half the cost.
    // The plan and the buffers are reused for every window.
    let fft = real_fft_plan(nfft);
    let mut input = fft.make_input_vec();
    let mut spectrum = fft.make_output_vec();
    let mut scratch = fft.make_scratch_vec();

    let num_windows = (data.len() - nfft) / step + 1;

    // result[freq_bin][window_index]
    let mut result: Vec<Vec<Complex<f64>>> =
        vec![vec![Complex::new(0.0, 0.0); num_windows]; freq_bins];

    for (win_idx, start) in (0..).zip((0..).step_by(step).take(num_windows)) {
        input.copy_from_slice(&data[start..start + nfft]);
        fft.process_with_scratch(&mut input, &mut spectrum, &mut scratch)
            .expect("buffer lengths come from the plan");
        for (bin, &value) in result.iter_mut().zip(&spectrum) {
            bin[win_idx] = value;
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
        }
    }

    /// Each STFT column is bins 0..=nfft/2 of the complex FFT of its window, for even and
    /// odd `nfft` (with and without a Nyquist bin) and with overlap.
    #[test]
    fn stft_matches_complex_fft_of_each_window() {
        let x = noise(500, 7);
        for (nfft, overlap) in [
            (1, 0),
            (2, 0),
            (7, 3),
            (32, 0),
            (33, 8),
            (160, 0),
            (160, 80),
        ] {
            let result = stft(&x, nfft, overlap).unwrap();
            assert_eq!(result.len(), nfft / 2 + 1);
            let fft = FftPlanner::<f64>::new().plan_fft_forward(nfft);
            let step = nfft - overlap;
            for w in 0..result[0].len() {
                let mut buf: Vec<Complex<f64>> = x[w * step..w * step + nfft]
                    .iter()
                    .map(|&v| Complex::new(v, 0.0))
                    .collect();
                fft.process(&mut buf);
                for (k, bin) in result.iter().enumerate() {
                    let diff = (bin[w] - buf[k]).norm();
                    assert!(
                        diff < 1e-12 * nfft as f64,
                        "nfft {nfft}, overlap {overlap}, window {w}, bin {k}: diff {diff}"
                    );
                }
            }
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

    /// Data shorter than one window is an error, returned before the O(nfft) FFT plan and
    /// bins are allocated: a huge `nfft` must not abort the process.
    #[test]
    fn stft_rejects_data_shorter_than_window() {
        assert_eq!(
            stft(&[0.0; 4], 1 << 40, 0),
            Err(MusicError::DataShorterThanWindow {
                nfft: 1 << 40,
                samples: 4
            })
        );
        assert_eq!(
            stft(&[], 2, 1),
            Err(MusicError::DataShorterThanWindow {
                nfft: 2,
                samples: 0
            })
        );
        // Parameter errors come first.
        assert_eq!(
            stft(&[0.0; 4], 1 << 40, 1 << 40),
            Err(MusicError::InvalidOverlap {
                no_overlap: 1 << 40,
                nfft: 1 << 40
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

    /// The plan cache returns the cached plan for a repeated length, keeps at most
    /// `REAL_PLAN_CACHE_SIZE` lengths and evicts the least recently used one.
    #[test]
    fn real_fft_plan_cache_reuses_and_evicts() {
        // Runs on its own test thread, so the thread-local cache starts empty
        let lengths: Vec<usize> = (1..=REAL_PLAN_CACHE_SIZE).map(|i| 100 + i).collect();
        let first: Vec<_> = lengths.iter().map(|&n| real_fft_plan(n)).collect();
        // Every length is cached: the same plan comes back
        for (n, plan) in lengths.iter().zip(&first) {
            assert!(
                Arc::ptr_eq(&real_fft_plan(*n), plan),
                "length {n} not reused"
            );
        }
        // Touch the first length so the second becomes the least recently used, then add one
        assert!(Arc::ptr_eq(&real_fft_plan(lengths[0]), &first[0]));
        real_fft_plan(999);
        let cached: Vec<usize> = REAL_PLANS.with(|c| c.borrow().iter().map(|p| p.len()).collect());
        assert_eq!(cached.len(), REAL_PLAN_CACHE_SIZE);
        assert!(
            !cached.contains(&lengths[1]),
            "least recently used length kept: {cached:?}"
        );
        assert!(cached.contains(&lengths[0]) && cached.contains(&999));
        assert_eq!(real_fft_plan(37).len(), 37);
    }
}
