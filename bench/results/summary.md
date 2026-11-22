| Workload | Implementation | n | Mean ± stddev | Median | p5 – p95 | CV | musicrs speedup (mean) |
|---|---|---:|---:|---:|---:|---:|---:|
| Narrowband DOA · 1 s recording | pyMUSIC | 200 | 160 ms ± 6.59 ms | 159 ms | 151 ms – 174 ms | 4.1% | 25× |
|  | numpy (vectorized) | 1000 | 3.03 ms ± 263 µs | 2.95 ms | 2.78 ms – 3.55 ms | 8.7% | 0.48× |
|  | musicrs | 1000 | 6.29 ms ± 699 µs | 6.03 ms | 5.58 ms – 7.46 ms | 11.1% |  |
| Narrowband DOA · 2 s recording | pyMUSIC | 200 | 158 ms ± 6.69 ms | 156 ms | 151 ms – 173 ms | 4.2% | 12× |
|  | numpy (vectorized) | 1000 | 5.02 ms ± 536 µs | 4.84 ms | 4.45 ms – 6.14 ms | 10.7% | 0.38× |
|  | musicrs | 1000 | 13.4 ms ± 938 µs | 13 ms | 12.5 ms – 15.2 ms | 7.0% |  |
| IFB-MUSIC, 1 bin · 1 s recording | pyMUSIC | 200 | 144 ms ± 5.94 ms | 142 ms | 139 ms – 157 ms | 4.1% | 129× |
|  | numpy (vectorized) | 1000 | 512 µs ± 44 µs | 502 µs | 466 µs – 594 µs | 8.6% | 0.46× |
|  | musicrs | 1000 | 1.12 ms ± 79 µs | 1.1 ms | 1.05 ms – 1.26 ms | 7.1% |  |
| IFB-MUSIC, 1 bin · 2 s recording | pyMUSIC | 200 | 149 ms ± 6.33 ms | 147 ms | 145 ms – 163 ms | 4.2% | 60× |
|  | numpy (vectorized) | 1000 | 855 µs ± 105 µs | 819 µs | 753 µs – 1.07 ms | 12.3% | 0.35× |
|  | musicrs | 1000 | 2.48 ms ± 181 µs | 2.5 ms | 2.04 ms – 2.66 ms | 7.3% |  |
| IFB-MUSIC, 80 bins · 1 s recording | pyMUSIC | 30 | 11.56 s ± 256 ms | 11.58 s | 11.27 s – 11.91 s | 2.2% | 2,129× |
|  | numpy (vectorized) | 1000 | 18.8 ms ± 1.82 ms | 18.2 ms | 17 ms – 22.4 ms | 9.7% | 3.46× |
|  | musicrs | 1000 | 5.43 ms ± 553 µs | 5.46 ms | 4.73 ms – 6.21 ms | 10.2% |  |

| Workload | Implementation | Shape | Skew | Excess kurtosis | Shapiro p (t / log t) | (Mean − median) / median | Outliers | Tail excess | Lag-1 autocorr. | Drift |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Narrowband DOA · 1 s recording | pyMUSIC | right-skewed, not normal (closer to log-normal) | 0.93 | 1.0 | 6.9e-08 / 6.8e-07 | +0.2% | 5.5% | +0.6% | 0.52 | +4.2% |
|  | numpy (vectorized) | right-skewed, heavy tail, not normal (closer to log-normal) | 2.14 | 5.7 | 6.2e-34 / 5.5e-31 | +2.6% | 6.1% | +1.7% | 0.52 | -0.4% |
|  | musicrs | right-skewed, heavy tail, not normal (closer to log-normal) | 1.81 | 6.8 | 5.9e-30 / 8.9e-26 | +4.2% | 2.3% | +0.9% | 0.66 | +7.1% |
| Narrowband DOA · 2 s recording | pyMUSIC | right-skewed, not normal (closer to log-normal) | 1.31 | 1.9 | 3.6e-11 / 2.4e-10 | +1.2% | 5.5% | +0.6% | 0.51 | -4.5% |
|  | numpy (vectorized) | right-skewed, not normal (closer to log-normal) | 1.22 | 1.0 | 2.6e-27 / 1.1e-24 | +3.8% | 5.5% | +1.6% | 0.75 | +6.0% |
|  | musicrs | right-skewed, heavy tail, not normal (closer to log-normal) | 2.56 | 16.8 | 5.5e-34 / 6.4e-31 | +2.5% | 12.2% | +2.1% | 0.67 | -0.4% |
| IFB-MUSIC, 1 bin · 1 s recording | pyMUSIC | right-skewed, heavy tail, not normal (closer to log-normal) | 2.31 | 5.7 | 1e-18 / 3.3e-18 | +1.5% | 18.0% | +1.7% | 0.39 | -2.7% |
|  | numpy (vectorized) | right-skewed, heavy tail, not normal (closer to log-normal) | 1.74 | 4.2 | 6.6e-30 / 9.5e-27 | +2.0% | 3.5% | +1.0% | 0.71 | -6.4% |
|  | musicrs | right-skewed, heavy tail, not normal (closer to log-normal) | 3.63 | 28.5 | 5.9e-37 / 7.1e-33 | +2.0% | 4.8% | +1.0% | 0.45 | -0.1% |
| IFB-MUSIC, 1 bin · 2 s recording | pyMUSIC | right-skewed, heavy tail, not normal (closer to log-normal) | 2.31 | 5.8 | 2.9e-18 / 1.1e-17 | +1.7% | 15.0% | +1.6% | 0.52 | -2.8% |
|  | numpy (vectorized) | right-skewed, not normal (closer to log-normal) | 1.53 | 2.6 | 5.9e-30 / 9.9e-27 | +4.3% | 5.8% | +2.1% | 0.76 | -3.9% |
|  | musicrs | near-symmetric, heavy tail, not normal (closer to normal) | 0.25 | 10.0 | 8.6e-34 / 9.8e-34 | -0.7% | 1.9% | +0.5% | 0.65 | +24.7% |
| IFB-MUSIC, 80 bins · 1 s recording | pyMUSIC | right-skewed, not normal (closer to log-normal) | 0.97 | 1.1 | 0.0043 / 0.0057 | -0.2% | 0.0% | +0.0% | 0.18 | -1.2% |
|  | numpy (vectorized) | right-skewed, heavy tail, not normal (closer to log-normal) | 2.57 | 11.7 | 8.2e-35 / 7.3e-31 | +3.0% | 12.1% | +2.9% | 0.70 | -1.6% |
|  | musicrs | near-symmetric, not normal (closer to log-normal) | 0.42 | -0.2 | 2.4e-24 / 6.1e-24 | -0.7% | 0.2% | +0.1% | 0.87 | -13.6% |
