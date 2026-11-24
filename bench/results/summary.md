| Workload | Implementation | n | Mean ± stddev | Median | p5 – p95 | CV | musicrs speedup (mean) |
|---|---|---:|---:|---:|---:|---:|---:|
| Narrowband DOA · 1 s recording | pyMUSIC | 200 | 164 ms ± 12.1 ms | 160 ms | 153 ms – 191 ms | 7.4% | 152× |
|  | numpy (vectorized) | 1000 | 3.28 ms ± 370 µs | 3.2 ms | 2.88 ms – 4.06 ms | 11.3% | 3.03× |
|  | musicrs | 1000 | 1.08 ms ± 90 µs | 1.07 ms | 982 µs – 1.22 ms | 8.3% |  |
| Narrowband DOA · 2 s recording | pyMUSIC | 200 | 161 ms ± 8.79 ms | 158 ms | 154 ms – 175 ms | 5.5% | 61× |
|  | numpy (vectorized) | 1000 | 5.47 ms ± 639 µs | 5.33 ms | 4.88 ms – 6.45 ms | 11.7% | 2.05× |
|  | musicrs | 1000 | 2.66 ms ± 998 µs | 2.48 ms | 2.24 ms – 3.44 ms | 37.5% |  |
| IFB-MUSIC, 1 bin · 1 s recording | pyMUSIC | 200 | 146 ms ± 5.88 ms | 144 ms | 141 ms – 161 ms | 4.0% | 124× |
|  | numpy (vectorized) | 1000 | 552 µs ± 53 µs | 538 µs | 497 µs – 657 µs | 9.6% | 0.47× |
|  | musicrs | 1000 | 1.17 ms ± 111 µs | 1.16 ms | 1.05 ms – 1.32 ms | 9.5% |  |
| IFB-MUSIC, 1 bin · 2 s recording | pyMUSIC | 200 | 150 ms ± 5.8 ms | 148 ms | 145 ms – 163 ms | 3.9% | 70× |
|  | numpy (vectorized) | 1000 | 922 µs ± 166 µs | 869 µs | 792 µs – 1.23 ms | 18.0% | 0.43× |
|  | musicrs | 1000 | 2.15 ms ± 178 µs | 2.11 ms | 1.95 ms – 2.44 ms | 8.3% |  |
| IFB-MUSIC, 80 bins · 1 s recording | pyMUSIC | 30 | 11.63 s ± 216 ms | 11.55 s | 11.37 s – 11.92 s | 1.9% | 2,447× |
|  | numpy (vectorized) | 1000 | 18.7 ms ± 1.35 ms | 18.4 ms | 17.4 ms – 21.5 ms | 7.2% | 3.94× |
|  | musicrs | 1000 | 4.75 ms ± 1.29 ms | 4.45 ms | 4.15 ms – 5.64 ms | 27.2% |  |

| Workload | Implementation | Shape | Skew | Excess kurtosis | Shapiro p (t / log t) | (Mean − median) / median | Outliers | Tail excess | Lag-1 autocorr. | Drift |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Narrowband DOA · 1 s recording | pyMUSIC | right-skewed, heavy tail, not normal (closer to log-normal) | 2.15 | 5.8 | 2.7e-16 / 5.7e-15 | +2.6% | 17.0% | +2.8% | 0.68 | -0.6% |
|  | numpy (vectorized) | right-skewed, heavy tail, not normal (closer to log-normal) | 2.25 | 8.5 | 7e-32 / 1.9e-26 | +2.5% | 6.7% | +2.3% | 0.66 | -13.5% |
|  | musicrs | right-skewed, heavy tail, not normal (closer to log-normal) | 3.85 | 38.6 | 7.2e-34 / 4.9e-26 | +1.4% | 1.4% | +0.5% | 0.51 | -2.8% |
| Narrowband DOA · 2 s recording | pyMUSIC | right-skewed, heavy tail, not normal (closer to log-normal) | 2.65 | 9.2 | 1.7e-17 / 3.1e-16 | +1.8% | 7.5% | +1.2% | 0.35 | -1.6% |
|  | numpy (vectorized) | right-skewed, heavy tail, not normal (closer to log-normal) | 4.31 | 37.4 | 4.1e-38 / 1.2e-30 | +2.7% | 4.8% | +1.8% | 0.73 | +5.6% |
|  | musicrs | right-skewed, heavy tail, not normal (closer to log-normal) | 15.97 | 352.9 | 2.2e-52 / 1.5e-41 | +7.3% | 7.5% | +6.0% | 0.24 | -5.3% |
| IFB-MUSIC, 1 bin · 1 s recording | pyMUSIC | right-skewed, not normal (closer to log-normal) | 1.61 | 1.8 | 8.8e-16 / 2.7e-15 | +1.6% | 13.5% | +1.4% | 0.38 | +0.7% |
|  | numpy (vectorized) | right-skewed, not normal (closer to log-normal) | 1.46 | 2.8 | 1.6e-26 / 1e-22 | +2.7% | 3.6% | +1.1% | 0.66 | +3.7% |
|  | musicrs | right-skewed, heavy tail, not normal (closer to log-normal) | 3.82 | 29.8 | 3.3e-36 / 1.1e-28 | +0.9% | 0.9% | +0.6% | 0.40 | +6.4% |
| IFB-MUSIC, 1 bin · 2 s recording | pyMUSIC | right-skewed, not normal (closer to log-normal) | 1.82 | 2.9 | 4.8e-16 / 1.7e-15 | +1.3% | 13.0% | +1.3% | 0.47 | +2.4% |
|  | numpy (vectorized) | right-skewed, heavy tail, not normal (closer to log-normal) | 4.78 | 51.2 | 7.2e-40 / 1.1e-31 | +6.1% | 10.8% | +4.9% | 0.69 | +1.8% |
|  | musicrs | right-skewed, heavy tail, not normal (closer to log-normal) | 2.63 | 14.9 | 7.6e-32 / 1.7e-26 | +1.8% | 2.5% | +0.8% | 0.38 | +1.1% |
| IFB-MUSIC, 80 bins · 1 s recording | pyMUSIC | right-skewed, not normal (closer to log-normal) | 0.53 | -0.7 | 0.026 / 0.029 | +0.7% | 0.0% | +0.0% | 0.26 | -0.6% |
|  | numpy (vectorized) | right-skewed, heavy tail, not normal (closer to log-normal) | 2.16 | 7.2 | 5e-32 / 8.4e-29 | +1.8% | 7.2% | +1.5% | 0.62 | +3.7% |
|  | musicrs | right-skewed, heavy tail, not normal (closer to log-normal) | 10.40 | 167.8 | 2.4e-50 / 6.9e-43 | +6.8% | 10.1% | +5.6% | 0.49 | -0.1% |
