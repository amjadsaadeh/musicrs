//! Times musicrs on the workloads of `bench/bench_pymusic.py` and writes every sample to a CSV.
//!
//! ```sh
//! cargo run --release --example bench_pymusic -- bench/results/musicrs.csv
//! ```
//!
//! Each workload runs a few warm-up calls, then times every call separately, so the analysis
//! (`bench/analyze.py`) can look at the whole distribution rather than one aggregate. The
//! workloads, recordings and CSV format mirror the Python script; see `bench/README.md`.
//! Set `BENCH_SCALE` (e.g. `0.1`) to scale every sample count for a quick run.

use std::fs::File;
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::time::Instant;

use musicrs::MusicEstimator;
use nalgebra::DMatrix;

const DATA_DIR: &str = "tests/data/single-source";
const NFFT: usize = 160;
const FREQ: u32 = 1400;
const WARMUP: usize = 3;

/// A 4-channel, 24-bit PCM WAV file as a `(4 × n)` matrix normalised by 2²³ (as the tests).
fn read_wav_4ch(path: &str) -> DMatrix<f64> {
    let mut reader = hound::WavReader::open(path).expect("failed to open WAV file");
    let spec = reader.spec();
    assert_eq!((spec.channels, spec.bits_per_sample), (4, 24));
    let raw: Vec<i32> = reader.samples::<i32>().map(Result::unwrap).collect();
    let scale = 2.0_f64.powi(23);
    DMatrix::from_fn(4, raw.len() / 4, |ch, t| raw[t * 4 + ch] as f64 / scale)
}

type Workload = Box<dyn Fn(&MusicEstimator, &DMatrix<f64>)>;

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "bench/results/musicrs.csv".into());
    let scale: f64 = std::env::var("BENCH_SCALE").map_or(1.0, |s| s.parse().expect("BENCH_SCALE"));

    let one_bin = [FREQ as usize * NFFT / 16000];
    let all_bins: Vec<usize> = (1..=NFFT / 2).collect();
    // (workload, call, recordings, samples per recording). Same calls and recordings as
    // bench_pymusic.py; musicrs is fast enough for many more samples.
    let workloads: Vec<(&str, Workload, &[&str], usize)> = vec![
        (
            "narrowband",
            Box::new(|est, d| {
                black_box(est.estimate_doa_from_real(d, 1, FREQ as f64).unwrap());
            }),
            &["1.0", "2.0"],
            1000,
        ),
        (
            "ifb_one_bin",
            Box::new(move |est, d| {
                black_box(est.ifb_music_spectrum(d, NFFT, 1, Some(&one_bin)).unwrap());
            }),
            &["1.0", "2.0"],
            1000,
        ),
        (
            "ifb_all_bins",
            Box::new(move |est, d| {
                black_box(est.ifb_music_spectrum(d, NFFT, 1, Some(&all_bins)).unwrap());
            }),
            &["1.0"],
            1000,
        ),
    ];

    if let Some(dir) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(dir).expect("create output directory");
    }
    let mut out = BufWriter::new(File::create(&out_path).expect("create output CSV"));
    writeln!(out, "implementation,workload,recording_s,iteration,seconds").unwrap();

    let est = MusicEstimator::default();
    for (name, run, durations, samples) in &workloads {
        for duration in *durations {
            let data = read_wav_4ch(&format!(
                "{DATA_DIR}/single-source_{FREQ}Hz_40deg_{duration}s.wav"
            ));
            let samples = ((*samples as f64 * scale).round() as usize).max(5);
            for _ in 0..WARMUP {
                run(&est, black_box(&data));
            }
            for i in 0..samples {
                let t0 = Instant::now();
                run(&est, black_box(&data));
                let dt = t0.elapsed().as_secs_f64();
                writeln!(out, "musicrs,{name},{duration},{i},{dt:e}").unwrap();
            }
            eprintln!("musicrs {name} {duration}s: {samples} samples");
        }
    }
    out.flush().unwrap();
}
