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
//! use musicrs::{MusicEstimator, analytic_signal};
//! use nalgebra::DMatrix;
//! use num_complex::Complex;
//!
//! // Build estimator with the default Kinect 4-mic array geometry
//! let estimator = MusicEstimator::default();
//!
//! // Suppose `real_data` is a 4×N real-valued DMatrix<f64> read from a WAV file.
//! // Convert each row to an analytic (complex) signal:
//! // let complex_data = ...;
//!
//! // Estimate DOA for a 300 Hz narrowband source:
//! // let angles = estimator.estimate_doa(&complex_data, 1, 300.0);
//! // println!("DOA estimate: {}°", angles[0].to_degrees());
//! ```

pub mod music;
pub mod stft;

pub use music::MusicEstimator;
pub use stft::{analytic_signal, stft};
