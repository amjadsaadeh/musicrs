"""Summarise the benchmark samples, classify their distributions and draw the charts.

Usage (from the repo root, after both benchmark runs; see bench/README.md):

    uv run --python 3.12 --with numpy --with scipy --with matplotlib \
        python -I bench/analyze.py

Reads   bench/results/{musicrs,pymusic}.csv  (one row per timed call)
Writes  bench/results/summary.csv            (every statistic, machine-readable)
        bench/results/summary.md             (the tables in the README)
        docs/benchmark{,-dark}.svg           (mean ± stddev bar chart)
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
IMPLS = ["pymusic", "musicrs"]
LABEL = {"pymusic": "pyMUSIC", "musicrs": "musicrs"}
WORKLOADS = [
    ("narrowband", "Narrowband DOA"),
    ("ifb_one_bin", "IFB-MUSIC, 1 bin"),
    ("ifb_all_bins", "IFB-MUSIC, 80 bins"),
]

# Reference palette of the dataviz method: categorical slots 1 and 2, chart chrome and ink
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", musicrs="#2a78d6", pymusic="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", musicrs="#3987e5", pymusic="#d95926"),
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


def cases(samples):
    """(workload key, label, recording) in display order, for the cases both sides ran."""
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


def bar_chart(samples, summary, theme, path):
    th = THEMES[theme]
    rows = cases(samples)
    fig, ax = plt.subplots(figsize=(8.4, 0.95 * len(rows) + 1.3), dpi=100)
    fig.patch.set_facecolor(th["surface"])
    style(ax, th)
    ax.set_xscale("log")
    ax.grid(axis="x", which="major", color=th["grid"], linewidth=1)
    ax.set_axisbelow(True)

    bar_h = 0.3  # ~22 px at this size: under the 24 px cap
    lo = 1e-4
    for i, (key, label, rec) in enumerate(rows):
        y0 = len(rows) - 1 - i
        for j, impl in enumerate(IMPLS):
            s = summary[(impl, key, rec)]
            y = y0 + (0.17 if j == 0 else -0.17)
            ax.barh(y, s["mean"] - lo, left=lo, height=bar_h, color=th[impl], linewidth=0)
            # ±1 stddev whisker; the lower end is clipped to the fastest call
            left = max(s["mean"] - s["std"], s["min"])
            ax.plot([left, s["mean"] + s["std"]], [y, y], color=th["ink2"], linewidth=1.2,
                    solid_capstyle="butt")
            ax.text(s["mean"] + s["std"], y, f"  {fmt_time(s['mean'])} ± {fmt_time(s['std'])}",
                    va="center", ha="left", fontsize=8.5, color=th["ink2"])
        speedup = summary[("pymusic", key, rec)]["mean"] / summary[("musicrs", key, rec)]["mean"]
        ax.text(1.02, y0, f"{speedup:,.0f}× faster", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=10, fontweight="bold", color=th["ink"])

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in reversed(rows)], fontsize=9.5, color=th["ink"])
    ax.set_xlim(lo, 300)
    ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0.1 ms", "1 ms", "10 ms", "100 ms", "1 s", "10 s"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Time per call, log scale (bar: mean, whisker: ± 1 stddev)",
                  color=th["muted"], fontsize=9)
    handles = [matplotlib.patches.Patch(color=th[impl], label=LABEL[impl]) for impl in IMPLS]
    leg = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
                    frameon=False, fontsize=9.5, handlelength=1, handleheight=1)
    for t in leg.get_texts():
        t.set_color(th["ink"])
    fig.suptitle("Runtime per call: musicrs vs pyMUSIC", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=th["ink"])
    fig.tight_layout()
    fig.savefig(path, facecolor=th["surface"], metadata={"Date": None})
    plt.close(fig)


def distribution_chart(samples, summary, theme, path):
    th = THEMES[theme]
    rows = cases(samples)
    fig, axes = plt.subplots(len(rows), 2, figsize=(8.4, 1.45 * len(rows) + 0.9), dpi=100,
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
        ax.set_xticklabels(["0.8", "1 = median", "1.2", "1.4", "1.6", "1.8", "≥ 2"])
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

    lines = ["| Workload | Implementation | n | Mean ± stddev | Median | p5 – p95 | CV | Speedup (mean) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key, label, rec in cases(samples):
        speedup = summary[("pymusic", key, rec)]["mean"] / summary[("musicrs", key, rec)]["mean"]
        for impl in IMPLS:
            s = summary[(impl, key, rec)]
            lines.append(
                f"| {label if impl == IMPLS[0] else ''} | {LABEL[impl]} | {s['n']} | "
                f"{fmt_time(s['mean'])} ± {fmt_time(s['std'])} | {fmt_time(s['median'])} | "
                f"{fmt_time(s['p5'])} – {fmt_time(s['p95'])} | {s['cv']:.1%} | "
                f"{f'{speedup:,.0f}×' if impl == 'musicrs' else ''} |")
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
        bar_chart(samples, summary, theme, os.path.join(DOCS, f"benchmark{suffix}.svg"))
        distribution_chart(samples, summary, theme,
                           os.path.join(DOCS, f"benchmark-distribution{suffix}.svg"))


if __name__ == "__main__":
    main()
