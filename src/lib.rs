//! # musicrs
//!
//! Rust port of [pyMUSIC](https://github.com/amjadsaadeh/pyMUSIC) — an implementation of the
//! **MUSIC** (MUltiple SIgnal Classification) algorithm for Direction-of-Arrival (DOA) estimation
//! with microphone/antenna arrays.
//!
//! ## Overview
//!
//! The MUSIC algorithm decomposes the covariance matrix of array observations into signal and
//! noise subspaces. Steering vectors that are orthogonal to the noise subspace indicate the
//! directions from which signals arrive. Peaks in the resulting pseudo-spectrum correspond to
//! estimated arrival angles.
//!
//! ## Quick Start
//!
//! ```rust
//! use musicrs::MusicEstimator;
//! use nalgebra::DMatrix;
//!
//! // Default Kinect 4-mic linear array, 720 angle steps over [0, 2π), 16 kHz
//! let estimator = MusicEstimator::default();
//!
//! // `real_data` is 4 × N real-valued microphone data (e.g. read from a WAV file).
//! // Here: silence plus a tiny 300 Hz tone, just to show the call.
//! let real_data = DMatrix::from_fn(4, 1600, |_, t| {
//!     (2.0 * std::f64::consts::PI * 300.0 * t as f64 / 16000.0).cos()
//! });
//!
//! // Analytic signal per channel, covariance, noise subspace, pseudo-spectrum, peaks:
//! let angles = estimator.estimate_doa_from_real(&real_data, 1, 300.0)?;
//! println!("DOA estimate: {:.1}°", angles[0].to_degrees());
//! # Ok::<(), musicrs::MusicError>(())
//! ```
//!
//! Use [`MusicEstimator::estimate_doa`] directly if you already have complex data, and
//! [`MusicEstimator::ifb_music_spectrum`] for the wideband (IFB) variant.

pub mod error;
pub mod music;
pub mod music3d;
pub mod stft;

pub use error::MusicError;
pub use music::{MusicEstimator, select_bins_by_energy};
pub use music3d::{Direction, MusicEstimator3d, Spectrum3d};
pub use stft::{analytic_signal, stft};
