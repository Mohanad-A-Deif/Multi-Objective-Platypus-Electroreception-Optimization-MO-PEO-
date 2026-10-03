#!/usr/bin/env python3
"""Summarise results/tau_diag_*.pkl (written by tau_diag.py): tables in results/tau_diag.md and a greyscale figure.
Statistics: medians with 95% percentile bootstrap intervals (10,000 resamples of the seeds); paired differences (common random
numbers); Wilcoxon signed-rank as a secondary check. Uncorrected p-values are shown only as a secondary indication."""
import glob, pickle, sys
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAUS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ALLP = ["DTLZ2", "DTLZ4", "DTLZ5", "DTLZ6", "ZDT4", "DTLZ1", "DTLZ3"]
PROBS = ALLP
FR = [0.2, 0.4, 0.6, 0.8, 1.0]
rng = np.random.default_rng(12345)
NB = 10000

R = {}
for fn in sorted(glob.glob("../results/tau_diag_*.pkl")):
    R.update(pickle.load(open(fn, "rb")))
seeds = {n: sorted({k[2] for k in R if k[0] == n}) for n in ALLP}
PROBS = [n for n in ALLP if seeds[n]]
FIGP = [n for n in PROBS if n in ("DTLZ2", "DTLZ4", "DTLZ5", "DTLZ6")]


def boot_med(x, nb=NB):
    x = np.asarray(x); idx = rng.integers(0, len(x), (nb, len(x)))
    m = np.median(x[idx], axis=1); return np.percentile(m, [2.5, 97.5])


def boot_mean(x, nb=NB):
    x = np.asarray(x); idx = rng.integers(0, len(x), (nb, len(x)))
    m = x[idx].mean(1); return np.percentile(m, [2.5, 97.5])


def col(name, tau, key="final_igd", ss=None):
    return np.array([R[(name, tau, s)][key] for s in (ss or seeds[name])])


def at(name, tau, frac, key="igd", ss=None):
    """value of a traced quantity at the recorded iteration closest to frac * T"""
    out = []
    for s in (ss or seeds[name]):
        tr = R[(name, tau, s)]; t = np.array(tr["t"]); i = int(np.argmin(np.abs(t - frac * tr["T"]))); out.append(tr[key][i])
    return np.array(out)


def fmt(x, d=4):
    return ("%." + str(d) + "f") % x


L = []
P = L.append
P("# Does the deferred-refinement schedule hurt well-conditioned unimodal fronts?  (tau diagnostic)\n")
P("MO-PEO final configuration with only tau varied (tau = 1.0 is the schedule disabled: mode=fixed, prel=0). Non-protocol seeds. "
  "Runs with different tau share the random stream and are identical up to t = tau*T, so paired per-seed differences isolate the relative-noise branch.\n")
for n in PROBS:
    P("- %s: %d seeds (%d..%d)" % (n, len(seeds[n]), seeds[n][0], seeds[n][-1]))
P("")

for n in PROBS:
    P("\n## %s  (%d seeds)\n" % (n, len(seeds[n])))
    P("**Final IGD** (median with 95%% bootstrap interval; mean with 95%% bootstrap interval; worst run)\n")
    P("| tau | median [95% CI] | mean [95% CI] | worst |\n|---|---|---|---|")
    for tau in TAUS:
        v = col(n, tau); lo, hi = boot_med(v); mlo, mhi = boot_mean(v)
        P("| %.1f%s | %s [%s, %s] | %s [%s, %s] | %s |" % (tau, " (schedule off)" if tau == 1.0 else (" (reported)" if tau == 0.4 else ""),
          fmt(np.median(v)), fmt(lo), fmt(hi), fmt(v.mean()), fmt(mlo), fmt(mhi), fmt(v.max())))
    P("\n**Median archive IGD along the run** (fraction of the budget)\n")
    P("| tau | " + " | ".join("%d%%" % (f * 100) for f in FR) + " |\n|---|" + "---|" * len(FR))
    for tau in TAUS:
        P("| %.1f | " % tau + " | ".join(fmt(np.median(at(n, tau, f))) for f in FR) + " |")
    P("\n**Median convergence gd_ref along the run** (mean distance of the archive to the nearest reference-front point)\n")
    P("| tau | " + " | ".join("%d%%" % (f * 100) for f in FR) + " |\n|---|" + "---|" * len(FR))
    for tau in TAUS:
        P("| %.1f | " % tau + " | ".join(fmt(np.median(at(n, tau, f, "gd")), 5) for f in FR) + " |")
    P("\n**Late-stage change of the archive IGD, paired per seed** (IGD at the end minus IGD at the stated fraction; positive = worsening)\n")
    P("| tau | 60% -> 100% median [95% CI] | 80% -> 100% median [95% CI] | seeds worse (80% -> 100%) |\n|---|---|---|---|")
    for tau in TAUS:
        d6 = col(n, tau) - at(n, tau, 0.6); d8 = col(n, tau) - at(n, tau, 0.8)
        l6 = boot_med(d6); l8 = boot_med(d8)
        P("| %.1f | %s [%s, %s] | %s [%s, %s] | %d/%d |" % (tau, fmt(np.median(d6), 5), fmt(l6[0], 5), fmt(l6[1], 5),
          fmt(np.median(d8), 5), fmt(l8[0], 5), fmt(l8[1], 5), int((d8 > 0).sum()), len(d8)))
    P("\n**Paired contrasts on the final IGD** (tau = 0.4 minus the other; negative = the reported schedule is better)\n")
    P("| against | median difference [95% CI] | median IGD ratio other / tau=0.4 [95% CI] | seeds where tau=0.4 is better | Wilcoxon signed-rank p (uncorrected) |\n|---|---|---|---|---|")
    base = col(n, 0.4)
    for tau in TAUS:
        if tau == 0.4:
            continue
        d = base - col(n, tau); lo, hi = boot_med(d)
        try:
            p = stats.wilcoxon(base, col(n, tau)).pvalue
        except ValueError:
            p = float("nan")
        o = col(n, tau); idx = rng.integers(0, len(o), (NB, len(o)))
        rat = np.median(o[idx], 1) / np.median(base[idx], 1); rlo, rhi = np.percentile(rat, [2.5, 97.5])
        P("| tau = %.1f%s | %s [%s, %s] | %.2f [%.2f, %.2f] | %d/%d | %.3g |" % (tau, " (off)" if tau == 1.0 else "", fmt(np.median(d), 5), fmt(lo, 5), fmt(hi, 5),
          np.median(o) / np.median(base), rlo, rhi, int((d < 0).sum()), len(d), p))
    # extra-seed rule
    C = {}
    for tau in (1.0, 0.8):
        d = base - col(n, tau); lo, hi = boot_med(d); C[tau] = (lo, hi)
    d8 = col(n, 0.4) - at(n, 0.4, 0.8); C["late"] = boot_med(d8)
    wide = [k for k, (lo, hi) in C.items() if lo <= 0 <= hi]
    P("\nExtra-seed rule (primary contrasts: 0.4 vs off, 0.4 vs 0.8, late-stage change at 0.4): intervals containing 0 -> %s" % (wide if wide else "none"))

open("../results/tau_diag.md", "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))

# ---------------- figure (greyscale, 600 dpi)
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9, "axes.edgecolor": "black", "axes.linewidth": 0.8,
                     "axes.labelsize": 9, "axes.titlesize": 9, "xtick.direction": "in", "ytick.direction": "in", "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 7.5, "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
                     "mathtext.fontset": "dejavuserif", "figure.dpi": 600, "savefig.dpi": 600, "lines.linewidth": 1.0})
STY = {0.0: dict(color="0.65", ls=(0, (1, 1))), 0.2: dict(color="0.5", ls=(0, (3, 1, 1, 1))), 0.4: dict(color="black", ls="-", lw=1.6),
       0.6: dict(color="0.3", ls="--"), 0.8: dict(color="0.45", ls="-."), 1.0: dict(color="0.15", ls=(0, (5, 1)), marker="o", ms=2.0, mfc="none", markevery=8)}
fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6))
for ax, n, tag in zip(axs.ravel(), FIGP, "abcd"):
    for tau in TAUS:
        T = np.array(R[(n, tau, seeds[n][0])]["t"]); T = T / R[(n, tau, seeds[n][0])]["T"]
        M = np.array([R[(n, tau, s)]["igd"] for s in seeds[n]])
        ax.plot(T, np.median(M, 0), label=r"$\tau=%.1f$%s" % (tau, " (off)" if tau == 1.0 else ""), **STY[tau])
    ax.set_yscale("log"); ax.set_title("%s (%d seeds)" % (n, len(seeds[n])), loc="left")
    ax.text(-0.16, 1.05, tag, transform=ax.transAxes, fontsize=10, fontweight="bold")
    ax.tick_params(which="both", top=True, right=True)
for ax in axs.ravel():
    ax.set_xlabel("fraction of the evaluation budget")
for ax in axs[:, 0]:
    ax.set_ylabel("median archive IGD")
# DTLZ2: differences are ~10%, so zoom on the second half of the run (linear axis)
for _ax in (axs[0, 1], axs[1, 0], axs[1, 1]):
    _ax.set_xlim(0, 1.02)
axs[1, 0].set_ylim(0.0068, 0.0105)
axs[0, 0].set_yscale("linear"); axs[0, 0].set_xlim(0.3, 1.02); axs[0, 0].set_ylim(0.084, 0.104)
axs[0, 0].set_title("DTLZ2 (%d seeds), zoom on 30-100%% of the budget" % len(seeds["DTLZ2"]), loc="left", fontsize=8)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=2.6)
fig.subplots_adjust(left=0.10, right=0.98, bottom=0.09, top=0.88, hspace=0.42, wspace=0.28)
fig.savefig("../results/tau_diag.png", dpi=600, bbox_inches="tight")

# ---------------- LaTeX table for the supplement (fresh seeds)
def sci(x, d=2):
    if x == 0:
        return "0"
    m, e = ("%.*e" % (d, x)).split("e"); return "%se%d" % (m, int(e))

rows = []
for n in PROBS:
    med = [np.median(col(n, t)) for t in TAUS]; b = int(np.argmin(med)); cs = []
    for i, t in enumerate(TAUS):
        v = col(n, t); top = sci(np.median(v), 2); top = "\\textbf{%s}" % top if i == b else top
        cs.append("\\makecell{%s\\\\(%s)}" % (top, sci(v.mean(), 2)))
    rows.append("%s & %d & %s \\\\" % (n, len(seeds[n]), " & ".join(cs)))
tex = ("\\begin{table}[htbp]\n\\centering\n\\caption{Sensitivity of MO-PEO to the deferred-refinement threshold $\\tau$ measured on seeds outside the protocol "
       "($\\geq 5000$; the number of runs is given per problem). Median IGD (upper line, best in bold) and mean IGD (lower line). $\\tau=1$ disables the schedule "
       "(absolute noise only). Runs with different $\\tau$ use the same random stream and coincide up to iteration $\\tau T$. "
       "On DTLZ2 the median IGD falls as $\\tau\\to1$; ZDT4 and DTLZ6 degrade several-fold as $\\tau\\to1$ and DTLZ1 mildly; on DTLZ4 about half of the runs fail at every $\\tau$, and DTLZ5 and DTLZ3 show no separable effect (Section~\\ref{sec:limits}).}\n"
       "\\label{tab:tau_fresh}\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{llcccccc}\n\\toprule\nProblem & Runs & $\\tau=0$ & $0.2$ & $0.4$ & $0.6$ & $0.8$ & $1.0$ \\\\\n\\midrule\n%s\n\\bottomrule\n\\end{tabular}}\n\\end{table}"
       % "\n".join(rows))
open("../paper/table_tau_fresh.tex", "w", encoding="utf-8").write(tex)
