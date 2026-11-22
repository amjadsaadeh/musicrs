//! Errors returned by the musicrs API.

use std::fmt;

/// Why a musicrs function could not produce a result.
#[derive(Debug, Clone, PartialEq)]
pub enum MusicError {
    /// The data (or noise subspace) has the wrong number of rows: one per antenna is
    /// expected.
    RowCountMismatch { expected: usize, got: usize },
    /// The covariance matrix is not `n_antennas × n_antennas`.
    CovarianceShape {
        n_antennas: usize,
        rows: usize,
        cols: usize,
    },
    /// The data has no rows or no samples.
    EmptyData,
    /// The data is shorter than one STFT window, so it has no complete window.
    DataShorterThanWindow { nfft: usize, samples: usize },
    /// The data contains a NaN or infinite value.
    NonFiniteData,
    /// The data (after removing each channel's mean, for real data) is all zeros, or
    /// every selected frequency bin is: there is no signal to locate.
    SilentData,
    /// `n_sources` must be less than the number of antennas, so that a noise subspace
    /// is left.
    TooManySources { n_sources: usize, n_antennas: usize },
    /// The STFT window size `nfft` is 0.
    InvalidNfft,
    /// The STFT overlap must be less than the window size.
    InvalidOverlap { no_overlap: usize, nfft: usize },
    /// The bin-selection threshold must be finite and non-negative.
    InvalidThreshold(f64),
    /// None of the requested frequency bins exists (each must be in `1..=nfft / 2`
    /// for the default selection, or below `nfft / 2 + 1`).
    NoUsableBins,
    /// The angle grid is unusable: `angle_steps_num` must be positive and the search
    /// interval finite with `start < end`.
    InvalidAngleGrid {
        angle_steps_num: usize,
        search_interval: (f64, f64),
    },
}

impl fmt::Display for MusicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RowCountMismatch { expected, got } => {
                write!(f, "expected {expected} rows (one per antenna), got {got}")
            }
            Self::CovarianceShape {
                n_antennas,
                rows,
                cols,
            } => write!(
                f,
                "covariance matrix is {rows} × {cols}, expected {n_antennas} × {n_antennas} \
                 (one row and column per antenna)"
            ),
            Self::EmptyData => write!(f, "data has no samples"),
            Self::DataShorterThanWindow { nfft, samples } => write!(
                f,
                "data has {samples} samples, fewer than one STFT window of {nfft}"
            ),
            Self::NonFiniteData => write!(f, "data contains NaN or infinite values"),
            Self::SilentData => write!(f, "data is silent: there is no signal to locate"),
            Self::TooManySources {
                n_sources,
                n_antennas,
            } => write!(
                f,
                "{n_sources} sources requested, but at most {} fit {n_antennas} antennas",
                n_antennas.saturating_sub(1)
            ),
            Self::InvalidNfft => write!(f, "nfft must be at least 1"),
            Self::InvalidOverlap { no_overlap, nfft } => write!(
                f,
                "overlap of {no_overlap} samples must be less than nfft = {nfft}"
            ),
            Self::InvalidThreshold(t) => {
                write!(f, "threshold must be finite and non-negative, got {t} dB")
            }
            Self::NoUsableBins => write!(f, "none of the requested frequency bins exists"),
            Self::InvalidAngleGrid {
                angle_steps_num,
                search_interval: (start, end),
            } => write!(
                f,
                "invalid angle grid: {angle_steps_num} steps over ({start}, {end}); \
                 need at least one step and a finite interval with start < end"
            ),
        }
    }
}

impl std::error::Error for MusicError {}

/// `Result` with [`MusicError`] as the error type.
pub type Result<T> = std::result::Result<T, MusicError>;
