"""Summarise the benchmark samples, classify their distributions and draw the charts.

Usage (from the repo root, after the three benchmark runs; see bench/README.md):

    uv run --python 3.12 --with numpy --with scipy --with matplotlib \
        python -I bench/analyze.py

Reads   bench/results/{pymusic,numpy,musicrs}.csv  (one row per timed call)
Writes  bench/results/summary.csv            (every statistic, machine-readable)
        bench/results/summary.md             (the tables in the README)
        docs/benchmark{,-dark}.svg           (mean and p5 – p95 per case, log time axis)
        docs/benchmark-distribution{,-dark}.svg  (histograms of t / median)

Scheduling noise (preemption, interrupts, frequency changes) only ever adds time, so it shows
up as a right tail: mean > median, positive skew and a few slow outliers. Besides mean,
stddev and median, the summary therefore reports
  * the spread around the median (MAD, p5/p95, min/max),
  * the shape (skewness, excess kurtosis, Shapiro-Wilk tests on t and on log t, and whether
    a normal or a log-normal fit has the lower AIC),
  * outliers: calls slower than median + 3 * 1.4826 * MAD (3 robust sigmas), their share,
    and how much they raise the mean ("tail excess": mean of all calls over the mean of the
    calls that are not outliers),
  * run order: lag-1 autocorrelation and drift (mean of the last 10 % of calls over the
    first 10 %). Scheduling noise hits calls independently; a regime change (CPU frequency,
    thermal state, migration to another core) makes neighbouring calls alike instead.
"""

import csv
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

RESULTS = "bench/results"
DOCS = "docs"
# Display order. "numpy" is pyMUSIC's algorithm vectorized (bench/bench_numpy.py): the baseline
# that separates what Rust buys from what pyMUSIC's per-angle Python loop costs.
IMPLS = ["pymusic", "numpy", "musicrs"]
LABEL = {"pymusic": "pyMUSIC", "numpy": "numpy (vectorized)", "musicrs": "musicrs"}
BASELINES = ["pymusic", "numpy"]
WORKLOADS = [
    ("narrowband", "Narrowband DOA"),
    ("ifb_one_bin", "IFB-MUSIC, 1 bin"),
    ("ifb_all_bins", "IFB-MUSIC, 80 bins"),
]

# Reference palette of the dataviz method: categorical slots 1-3, chart chrome and ink. The light
# aqua is below 3:1 against the surface, so every mark carries a direct value label.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", musicrs="#2a78d6", pymusic="#eb6834",
                  numpy="#1baf7a"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", musicrs="#3987e5", pymusic="#d95926",
                  numpy="#199e70"),
}


def load():
    samples = defaultdict(list)
    for impl in IMPLS:
        with open(os.path.join(RESULTS, f"{impl}.csv")) as f:
            for row in csv.DictReader(f):
                samples[(impl, row["workload"], row["recording_s"])].append(float(row["seconds"]))
    return {k: np.array(v) for k, v in samples.items()}


def describe(t):
    med = float(np.median(t))
    mad = float(np.median(np.abs(t - med))) * 1.4826  # scaled to sigma for normal data
    outliers = t > med + 3 * mad if mad > 0 else np.zeros(len(t), bool)
    mean = float(t.mean())
    # Shapiro-Wilk is only defined for 3..5000 samples
    sw = t[:5000]
    p_norm = float(stats.shapiro(sw).pvalue)
    p_lognorm = float(stats.shapiro(np.log(sw)).pvalue)
    # AIC of maximum-likelihood fits (2 parameters each; log-normal with loc fixed at 0)
    mu, sd = stats.norm.fit(t)
    aic_norm = 4 - 2 * stats.norm.logpdf(t, mu, sd).sum()
    shape, _, scale = stats.lognorm.fit(t, floc=0)
    aic_lognorm = 4 - 2 * stats.lognorm.logpdf(t, shape, 0, scale).sum()
    return dict(
        n=len(t),
        mean=mean,
        std=float(t.std(ddof=1)),
        cv=float(t.std(ddof=1)) / mean,
        median=med,
        mad=mad,
        min=float(t.min()),
        p5=float(np.percentile(t, 5)),
        p95=float(np.percentile(t, 95)),
        max=float(t.max()),
        skew=float(stats.skew(t)),
        kurtosis=float(stats.kurtosis(t)),
        shapiro_p_normal=p_norm,
        shapiro_p_lognormal=p_lognorm,
        better_fit="log-normal" if aic_lognorm < aic_norm else "normal",
        outlier_share=float(outliers.mean()),
        tail_excess=mean / float(t[~outliers].mean()) - 1,
        # In run order: independent noise has ~0 autocorrelation and no drift; a step or trend
        # (CPU frequency, thermal state, a move to another core) shows up in both
        autocorr_lag1=float(np.corrcoef(t[:-1], t[1:])[0, 1]),
        drift=float(t[-max(1, len(t) // 10):].mean() / t[: max(1, len(t) // 10)].mean()) - 1,
    )


def shape_verdict(s):
    """One phrase for the README table."""
    if s["shapiro_p_normal"] > 0.05:
        return "consistent with normal"
    if s["shapiro_p_lognormal"] > 0.05:
        return "consistent with log-normal"
    tail = "right-skewed" if s["skew"] > 0.5 else "skewed left" if s["skew"] < -0.5 else "near-symmetric"
    heavy = ", heavy tail" if s["kurtosis"] > 3 else ""
    return f"{tail}{heavy}, not normal (closer to {s['better_fit']})"


def fmt_time(x):
    if x >= 1:
        return f"{x:.2f} s"
    if x >= 1e-3:
        return f"{x * 1e3:.3g} ms" if x * 1e3 < 100 else f"{x * 1e3:.0f} ms"
    return f"{x * 1e6:.0f} µs"


def fmt_speedup(ratio):
    """musicrs's speedup over a baseline; below 1 means musicrs is slower."""
    return f"{ratio:,.0f}×" if ratio >= 10 else f"{ratio:.2f}×"


def relation(ratio, baseline):
    """'28× faster than pyMUSIC' / '1.4× slower than numpy', for musicrs."""
    if ratio >= 1:
        return f"{fmt_speedup(ratio)} faster than {baseline}"
    return f"{fmt_speedup(1 / ratio)} slower than {baseline}"


def cases(samples):
    """(workload key, label, recording) in display order, for the cases every implementation ran."""
    out = []
    for key, label in WORKLOADS:
        for rec in sorted({r for (_, w, r) in samples if w == key}):
            if all((impl, key, rec) in samples for impl in IMPLS):
                out.append((key, f"{label} · {float(rec):g} s recording", rec))
    return out


def style(ax, th):
    ax.set_facecolor(th["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(th["axis"])
    ax.tick_params(colors=th["muted"], length=0, labelsize=9)


def time_chart(samples, summary, theme, path):
    th = THEMES[theme]
    rows = cases(samples)
    fig, ax = plt.subplots(figsize=(9.6, 1.15 * len(rows) + 1.3), dpi=100)
    fig.patch.set_facecolor(th["surface"])
    style(ax, th)
    ax.grid(axis="x", which="major", color=th["grid"], linewidth=1)
    ax.set_axisbelow(True)

    # Log time axis: the three implementations span four decades (pyMUSIC's 11.5 s next to
    # sub-millisecond calls), and on a log axis equal ratios are equal distances. Dots, not
    # bars, because a bar's length means nothing on a log axis.
    offsets = {"pymusic": 0.28, "numpy": 0.0, "musicrs": -0.28}
    for i, (key, label, rec) in enumerate(rows):
        y0 = len(rows) - 1 - i
        for impl in IMPLS:
            s = summary[(impl, key, rec)]
            y = y0 + offsets[impl]
            ax.plot([s["p5"], s["p95"]], [y, y], color=th[impl], linewidth=2,
                    solid_capstyle="round", zorder=2)
            ax.plot(s["mean"], y, "o", color=th[impl], markersize=7,
                    markeredgecolor=th["surface"], markeredgewidth=1.5, zorder=3)
            ax.text(s["p95"] * 1.12, y, fmt_time(s["mean"]), va="center", ha="left",
                    fontsize=8.5, color=th["ink2"])
        musicrs = summary[("musicrs", key, rec)]["mean"]
        notes = [relation(summary[(b, key, rec)]["mean"] / musicrs, LABEL[b].split(" ")[0])
                 for b in BASELINES]
        ax.text(1.02, y0 + 0.14, notes[1], transform=ax.get_yaxis_transform(), va="center",
                ha="left", fontsize=9.5, fontweight="bold", color=th["ink"])
        ax.text(1.02, y0 - 0.16, notes[0], transform=ax.get_yaxis_transform(), va="center",
                ha="left", fontsize=8.5, color=th["ink2"])

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in reversed(rows)], fontsize=9.5, color=th["ink"])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xscale("log")
    ax.set_xlim(2e-4, 30)
    ticks = [1e-3, 1e-2, 1e-1, 1, 10]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1 ms", "10 ms", "100 ms", "1 s", "10 s"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Time per call, log scale (dot: mean, line: p5 – p95)",
                  color=th["muted"], fontsize=9)
    handles = [matplotlib.lines.Line2D([], [], marker="o", linestyle="", markersize=7,
                                       color=th[impl], label=LABEL[impl]) for impl in IMPLS]
    leg = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
                    frameon=False, fontsize=9.5, handletextpad=0.3)
    for t in leg.get_texts():
        t.set_color(th["ink"])
    fig.suptitle("Runtime per call: musicrs vs pyMUSIC and vectorized numpy", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=th["ink"])
    fig.tight_layout()
    fig.savefig(path, facecolor=th["surface"], metadata={"Date": None})
    plt.close(fig)


def distribution_chart(samples, summary, theme, path):
    th = THEMES[theme]
    rows = cases(samples)
    fig, axes = plt.subplots(len(rows), len(IMPLS), figsize=(10.2, 1.45 * len(rows) + 0.9), dpi=100,
                             sharex=True)
    fig.patch.set_facecolor(th["surface"])
    edges = np.linspace(0.8, 2.0, 61)  # t / median, overflow folded into the last bin
    for i, (key, label, rec) in enumerate(rows):
        for j, impl in enumerate(IMPLS):
            ax = axes[i, j]
            style(ax, th)
            ax.set_yticks([])
            t = samples[(impl, key, rec)]
            s = summary[(impl, key, rec)]
            rel = np.clip(t / s["median"], edges[0], edges[-1] - 1e-9)
            ax.hist(rel, bins=edges, color=th[impl], linewidth=0, rwidth=0.85)
            ax.axvline(1.0, color=th["ink2"], linewidth=1)
            ax.axvline(s["mean"] / s["median"], color=th["ink"], linewidth=1, linestyle=(0, (1, 1.5)))
            ax.text(0.99, 0.92,
                    f"n = {s['n']}  ·  CV {s['cv']:.1%}\nskew {s['skew']:.1f}  ·  outliers {s['outlier_share']:.1%}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=th["ink2"])
            if j == 0:
                ax.set_title(label, loc="left", fontsize=9, color=th["ink"], pad=3)
            if i == 0:
                ax.text(0, 1.32, LABEL[impl], transform=ax.transAxes, fontsize=10.5,
                        fontweight="bold", color=th["ink"])
    for ax in axes[-1]:
        ax.set_xticks([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
        ax.set_xticklabels(["0.8", "1", "1.2", "1.4", "1.6", "1.8", "≥ 2"])
        ax.set_xlabel("time / median  (solid line: median, dotted: mean)", color=th["muted"], fontsize=8.5)
    fig.tight_layout(h_pad=1.2)
    fig.savefig(path, facecolor=th["surface"], metadata={"Date": None})
    plt.close(fig)


def write_tables(samples, summary):
    keys = list(next(iter(summary.values())).keys())
    with open(os.path.join(RESULTS, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["implementation", "workload", "recording_s", *keys])
        for (impl, key, rec), s in sorted(summary.items()):
            w.writerow([impl, key, rec, *[s[k] for k in keys]])

    # Speedup of musicrs over the row's implementation (mean time); below 1× musicrs is slower
    lines = ["| Workload | Implementation | n | Mean ± stddev | Median | p5 – p95 | CV | musicrs speedup (mean) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key, label, rec in cases(samples):
        musicrs = summary[("musicrs", key, rec)]["mean"]
        for impl in IMPLS:
            s = summary[(impl, key, rec)]
            speedup = fmt_speedup(s["mean"] / musicrs) if impl in BASELINES else ""
            lines.append(
                f"| {label if impl == IMPLS[0] else ''} | {LABEL[impl]} | {s['n']} | "
                f"{fmt_time(s['mean'])} ± {fmt_time(s['std'])} | {fmt_time(s['median'])} | "
                f"{fmt_time(s['p5'])} – {fmt_time(s['p95'])} | {s['cv']:.1%} | {speedup} |")
    lines += ["", "| Workload | Implementation | Shape | Skew | Excess kurtosis | Shapiro p (t / log t) | (Mean − median) / median | Outliers | Tail excess | Lag-1 autocorr. | Drift |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, label, rec in cases(samples):
        for impl in IMPLS:
            s = summary[(impl, key, rec)]
            lines.append(
                f"| {label if impl == IMPLS[0] else ''} | {LABEL[impl]} | {shape_verdict(s)} | "
                f"{s['skew']:.2f} | {s['kurtosis']:.1f} | "
                f"{s['shapiro_p_normal']:.2g} / {s['shapiro_p_lognormal']:.2g} | "
                f"{(s['mean'] - s['median']) / s['median']:+.1%} | {s['outlier_share']:.1%} | "
                f"{s['tail_excess']:+.1%} | {s['autocorr_lag1']:.2f} | {s['drift']:+.1%} |")
    with open(os.path.join(RESULTS, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "musicrs-bench",
                         "font.family": "sans-serif"})
    samples = load()
    summary = {k: describe(v) for k, v in samples.items()}
    write_tables(samples, summary)
    os.makedirs(DOCS, exist_ok=True)
    for theme in THEMES:
        suffix = "" if theme == "light" else "-dark"
        time_chart(samples, summary, theme, os.path.join(DOCS, f"benchmark{suffix}.svg"))
        distribution_chart(samples, summary, theme,
                           os.path.join(DOCS, f"benchmark-distribution{suffix}.svg"))


if __name__ == "__main__":
    main()
