| Workload | Implementation | n | Mean ± stddev | Median | p5 – p95 | CV | musicrs speedup (mean) |
|---|---|---:|---:|---:|---:|---:|---:|
| Narrowband DOA · 1 s recording | pyMUSIC | 200 | 162 ms ± 7.49 ms | 161 ms | 153 ms – 175 ms | 4.6% | 168× |
|  | numpy (vectorized) | 1000 | 3.38 ms ± 338 µs | 3.32 ms | 2.94 ms – 4.07 ms | 10.0% | 3.50× |
|  | musicrs | 1000 | 966 µs ± 91 µs | 938 µs | 869 µs – 1.15 ms | 9.4% |  |
| Narrowband DOA · 2 s recording | pyMUSIC | 200 | 165 ms ± 7.96 ms | 163 ms | 155 ms – 178 ms | 4.8% | 79× |
|  | numpy (vectorized) | 1000 | 5.29 ms ± 529 µs | 5.18 ms | 4.57 ms – 6.26 ms | 10.0% | 2.55× |
|  | musicrs | 1000 | 2.08 ms ± 178 µs | 2.04 ms | 1.86 ms – 2.42 ms | 8.6% |  |
| IFB-MUSIC, 1 bin · 1 s recording | pyMUSIC | 200 | 153 ms ± 10.4 ms | 150 ms | 142 ms – 176 ms | 6.8% | 316× |
|  | numpy (vectorized) | 1000 | 563 µs ± 50 µs | 561 µs | 495 µs – 649 µs | 8.9% | 1.16× |
|  | musicrs | 1000 | 485 µs ± 34 µs | 475 µs | 455 µs – 554 µs | 7.0% |  |
| IFB-MUSIC, 1 bin · 2 s recording | pyMUSIC | 200 | 152 ms ± 10.8 ms | 147 ms | 145 ms – 170 ms | 7.1% | 171× |
|  | numpy (vectorized) | 1000 | 891 µs ± 113 µs | 859 µs | 767 µs – 1.15 ms | 12.7% | 1.00× |
|  | musicrs | 1000 | 889 µs ± 74 µs | 868 µs | 812 µs – 1.04 ms | 8.3% |  |
| IFB-MUSIC, 80 bins · 1 s recording | pyMUSIC | 30 | 11.75 s ± 281 ms | 11.70 s | 11.39 s – 12.19 s | 2.4% | 3,904× |
|  | numpy (vectorized) | 1000 | 19 ms ± 1.28 ms | 18.8 ms | 17.3 ms – 21.6 ms | 6.8% | 6.30× |
|  | musicrs | 1000 | 3.01 ms ± 186 µs | 2.96 ms | 2.81 ms – 3.41 ms | 6.2% |  |

| Workload | Implementation | Shape | Skew | Excess kurtosis | Shapiro p (t / log t) | (Mean − median) / median | Outliers | Tail excess | Lag-1 autocorr. | Drift |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Narrowband DOA · 1 s recording | pyMUSIC | right-skewed, heavy tail, not normal (closer to log-normal) | 1.94 | 7.5 | 4.3e-12 / 2.1e-10 | +1.0% | 5.0% | +0.7% | 0.54 | -2.1% |
|  | numpy (vectorized) | right-skewed, not normal (closer to log-normal) | 0.91 | 0.6 | 5.9e-19 / 3.1e-14 | +1.9% | 3.3% | +0.9% | 0.61 | +5.7% |
|  | musicrs | right-skewed, not normal (closer to log-normal) | 0.99 | 0.2 | 8.9e-26 / 6.6e-24 | +3.0% | 2.0% | +0.5% | 0.73 | -4.2% |
| Narrowband DOA · 2 s recording | pyMUSIC | right-skewed, heavy tail, not normal (closer to log-normal) | 1.94 | 6.4 | 5.4e-13 / 1.8e-11 | +1.0% | 8.0% | +1.1% | 0.40 | +0.3% |
|  | numpy (vectorized) | right-skewed, not normal (closer to log-normal) | 0.66 | -0.2 | 1.2e-16 / 3.2e-13 | +2.0% | 0.5% | +0.2% | 0.70 | +3.0% |
|  | musicrs | right-skewed, not normal (closer to log-normal) | 0.73 | -0.3 | 1.3e-20 / 1.5e-18 | +2.0% | 0.2% | +0.1% | 0.72 | +0.1% |
| IFB-MUSIC, 1 bin · 1 s recording | pyMUSIC | right-skewed, not normal (closer to log-normal) | 1.24 | 0.7 | 1.6e-12 / 1.5e-11 | +2.5% | 14.5% | +2.4% | 0.81 | +0.7% |
|  | numpy (vectorized) | right-skewed, not normal (closer to log-normal) | 0.56 | 0.1 | 2.8e-13 / 4.5e-10 | +0.3% | 0.2% | +0.1% | 0.66 | -3.3% |
|  | musicrs | right-skewed, heavy tail, not normal (closer to log-normal) | 2.02 | 9.2 | 1.8e-32 / 4.2e-30 | +2.3% | 5.6% | +1.1% | 0.60 | -5.1% |
| IFB-MUSIC, 1 bin · 2 s recording | pyMUSIC | right-skewed, heavy tail, not normal (closer to log-normal) | 3.05 | 11.2 | 1.9e-20 / 2.1e-19 | +2.9% | 19.5% | +2.8% | 0.74 | +12.0% |
|  | numpy (vectorized) | right-skewed, not normal (closer to log-normal) | 1.41 | 1.9 | 1.1e-27 / 2.7e-23 | +3.8% | 6.3% | +2.4% | 0.80 | +5.0% |
|  | musicrs | right-skewed, not normal (closer to log-normal) | 1.07 | 0.4 | 1.2e-26 / 6.2e-25 | +2.5% | 2.4% | +0.6% | 0.76 | -6.3% |
| IFB-MUSIC, 80 bins · 1 s recording | pyMUSIC | right-skewed, not normal (closer to log-normal) | 1.26 | 2.2 | 0.014 / 0.025 | +0.4% | 3.3% | +0.3% | 0.38 | +1.8% |
|  | numpy (vectorized) | right-skewed, not normal (closer to log-normal) | 1.05 | 1.4 | 5.9e-20 / 2e-16 | +1.1% | 3.1% | +0.6% | 0.62 | +2.2% |
|  | musicrs | right-skewed, not normal (closer to log-normal) | 1.20 | 1.0 | 2.1e-26 / 1.1e-24 | +1.6% | 3.7% | +0.7% | 0.62 | -2.5% |
