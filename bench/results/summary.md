| Workload | Implementation | n | Mean ± stddev | Median | p5 – p95 | CV | Speedup (mean) |
|---|---|---:|---:|---:|---:|---:|---:|
| Narrowband DOA · 1 s recording | pyMUSIC | 200 | 159 ms ± 8.03 ms | 158 ms | 150 ms – 173 ms | 5.0% |  |
|  | musicrs | 1000 | 5.63 ms ± 570 µs | 5.41 ms | 5.1 ms – 6.73 ms | 10.1% | 28× |
| Narrowband DOA · 2 s recording | pyMUSIC | 200 | 163 ms ± 18.9 ms | 157 ms | 151 ms – 205 ms | 11.6% |  |
|  | musicrs | 1000 | 11.7 ms ± 832 µs | 11.5 ms | 11 ms – 13.7 ms | 7.1% | 14× |
| IFB-MUSIC, 1 bin · 1 s recording | pyMUSIC | 200 | 144 ms ± 7.19 ms | 142 ms | 139 ms – 157 ms | 5.0% |  |
|  | musicrs | 1000 | 757 µs ± 121 µs | 770 µs | 633 µs – 845 µs | 16.0% | 191× |
| IFB-MUSIC, 1 bin · 2 s recording | pyMUSIC | 200 | 152 ms ± 6.85 ms | 151 ms | 143 ms – 164 ms | 4.5% |  |
|  | musicrs | 1000 | 1.24 ms ± 94 µs | 1.22 ms | 1.15 ms – 1.44 ms | 7.5% | 122× |
| IFB-MUSIC, 80 bins · 1 s recording | pyMUSIC | 30 | 11.53 s ± 244 ms | 11.43 s | 11.30 s – 11.83 s | 2.1% |  |
|  | musicrs | 1000 | 14.7 ms ± 748 µs | 14.5 ms | 13.9 ms – 16.4 ms | 5.1% | 784× |

| Workload | Implementation | Shape | Skew | Excess kurtosis | Shapiro p (t / log t) | (Mean − median) / median | Outliers | Tail excess | Lag-1 autocorr. | Drift |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Narrowband DOA · 1 s recording | pyMUSIC | right-skewed, heavy tail, not normal (closer to log-normal) | 2.15 | 7.9 | 1.1e-13 / 5.1e-12 | +0.8% | 2.5% | +0.5% | 0.46 | +6.5% |
|  | musicrs | right-skewed, heavy tail, not normal (closer to log-normal) | 2.13 | 11.4 | 7e-33 / 1.5e-29 | +4.0% | 10.3% | +2.5% | 0.72 | -0.3% |
| Narrowband DOA · 2 s recording | pyMUSIC | right-skewed, heavy tail, not normal (closer to log-normal) | 2.93 | 9.6 | 1.2e-20 / 3.1e-19 | +4.2% | 12.5% | +3.8% | 0.61 | +18.1% |
|  | musicrs | right-skewed, heavy tail, not normal (closer to log-normal) | 1.94 | 4.4 | 1.4e-33 / 1.5e-31 | +2.4% | 11.0% | +2.1% | 0.68 | +3.2% |
| IFB-MUSIC, 1 bin · 1 s recording | pyMUSIC | right-skewed, heavy tail, not normal (closer to log-normal) | 3.23 | 15.3 | 1.4e-19 / 1.4e-18 | +1.8% | 15.5% | +1.7% | 0.41 | +0.3% |
|  | musicrs | right-skewed, heavy tail, not normal (closer to log-normal) | 16.60 | 419.5 | 1.6e-48 / 3.2e-33 | -1.8% | 0.6% | +0.7% | 0.30 | +18.2% |
| IFB-MUSIC, 1 bin · 2 s recording | pyMUSIC | right-skewed, not normal (closer to log-normal) | 1.13 | 1.4 | 3e-09 / 2.9e-08 | +0.5% | 2.5% | +0.4% | 0.50 | +4.5% |
|  | musicrs | right-skewed, not normal (closer to log-normal) | 1.51 | 2.1 | 2.5e-30 / 2.3e-28 | +2.4% | 7.7% | +1.6% | 0.72 | -3.3% |
| IFB-MUSIC, 80 bins · 1 s recording | pyMUSIC | right-skewed, heavy tail, not normal (closer to log-normal) | 1.66 | 3.9 | 0.00034 / 0.00055 | +0.9% | 3.3% | +0.3% | 0.35 | +1.1% |
|  | musicrs | right-skewed, not normal (closer to log-normal) | 1.44 | 2.1 | 1.5e-27 / 1.1e-25 | +1.2% | 6.1% | +0.9% | 0.69 | -1.8% |
