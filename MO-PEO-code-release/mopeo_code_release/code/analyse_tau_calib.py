#!/usr/bin/env python3
"""Analysis of the tau calibration (calibrate_tau.py). The rule and the split are those of results/tau_calib_protocol.json, fixed before the runs.
Writes results/tau_calib.md, results/tau_calib.json, results/tau_calib.png and paper/table_tau_calib.tex."""
import json, pickle, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

proto = json.load(open("../results/tau_calib_protocol.json"))
D = pickle.load(open("../results/tau_calib.pkl", "rb"))
TAUS = proto["taus"]; TUNE = proto["tune"]; HELD = proto["heldout"]
PROBS = ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6", "DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]
SEEDS = list(range(proto["seeds"][0], proto["seeds"][1] + 1))
NB = 5000
rng = np.random.default_rng(777)
A = {p: np.array([[D[(p, t, s)] for s in SEEDS] for t in TAUS]) for p in PROBS}      # (tau, seed)
n = len(SEEDS)


def med(a):  # medians over seeds, per tau
    return np.median(a, axis=-1)


def regret(m):  # m: (..., tau) medians -> regret ratios
    return m / m.min(axis=-1, keepdims=True)


def gm(x):
    return float(np.exp(np.mean(np.log(x))))


def select(probs, medians):
    r = np.array([regret(medians[p]) for p in probs]); g = np.exp(np.log(r).mean(0)); return int(np.argmin(g)), g


M = {p: med(A[p]) for p in PROBS}
L = []; P = L.append
P("# Calibration of tau on fresh seeds (%d..%d, %d runs per problem and tau)\n" % (SEEDS[0], SEEDS[-1], n))
P("Split (declared before the runs, results/tau_calib_protocol.json): tuning = %s; held-out = %s.\n" % (", ".join(TUNE), ", ".join(HELD)))

# ---- per-problem sensitivity, all 12 problems, medians with bootstrap CI
idx = rng.integers(0, n, (NB, n))
CI = {p: np.percentile(np.median(A[p][:, idx], axis=-1), [2.5, 97.5], axis=1) for p in PROBS}     # (2, tau)
P("## Sensitivity to tau: median IGD [95% bootstrap interval of the median], all twelve problems\n")
P("| problem | half | " + " | ".join("%.1f" % t for t in TAUS) + " |\n|---|---|" + "---|" * len(TAUS))
for p in PROBS:
    b = int(np.argmin(M[p]))
    cells = ["%s%.4g [%.3g, %.3g]%s" % ("**" if i == b else "", M[p][i], CI[p][0][i], CI[p][1][i], "**" if i == b else "") for i in range(len(TAUS))]
    P("| %s | %s | " % (p, "tune" if p in TUNE else "held-out") + " | ".join(cells) + " |")

# ---- regret
P("\n## Regret ratio r_p(tau) = median IGD_p(tau) / best median IGD_p over the grid\n")
P("| problem | half | " + " | ".join("%.1f" % t for t in TAUS) + " |\n|---|---|" + "---|" * len(TAUS))
for p in PROBS:
    P("| %s | %s | " % (p, "tune" if p in TUNE else "held-out") + " | ".join("%.2f" % x for x in regret(M[p])) + " |")

# ---- declared selection on the tuning half
k, g = select(TUNE, M)
P("\n## Declared selection (tuning half)\n")
P("Geometric-mean regret over the tuning problems: " + ", ".join("tau=%.1f: %.3f" % (t, x) for t, x in zip(TAUS, g)))
P("\n**Selected tau = %.1f**\n" % TAUS[k])
# selection stability: bootstrap over seeds (the same resampled seeds for all tau within a problem)
cnt = np.zeros(len(TAUS)); heldgm = []
for b in range(NB):
    ii = rng.integers(0, n, n)
    Mb = {p: np.median(A[p][:, ii], axis=-1) for p in PROBS}
    kb, _ = select(TUNE, Mb); cnt[kb] += 1
P("Selection frequency over %d bootstrap resamples of the seeds (tuning half only): " % NB + ", ".join("tau=%.1f: %.1f%%" % (t, 100 * c / NB) for t, c in zip(TAUS, cnt)))
P("")
# ---- held-out evaluation
rh = np.array([regret(M[p])[k] for p in HELD]); rt = np.array([regret(M[p])[k] for p in TUNE])
P("## Held-out evaluation of the selected tau = %.1f\n" % TAUS[k])
P("| problem | regret at the selected tau | best tau on this problem | IGD at selected tau [95% CI] | IGD at best tau |\n|---|---|---|---|---|")
for p in HELD:
    b = int(np.argmin(M[p])); P("| %s | %.2f | %.1f | %.4g [%.3g, %.3g] | %.4g |" % (p, regret(M[p])[k], TAUS[b], M[p][k], CI[p][0][k], CI[p][1][k], M[p][b]))
P("\nGeometric-mean regret: held-out %.3f, tuning %.3f. Held-out problems with regret > 2: %d of %d." % (gm(rh), gm(rt), int((rh > 2).sum()), len(rh)))
ok = gm(rh) <= 1.5 and (rh > 2).sum() <= 2
P("**Pre-registered reading (GM regret <= 1.5 and at most 2 of 6 held-out regrets > 2): %s**" % ("DEFENSIBLE" if ok else "NOT DEFENSIBLE"))
# ---- swapped split
ks, gs = select(HELD, M); rs = np.array([regret(M[p])[ks] for p in TUNE])
P("\n## Swapped split (select on the held-out half, evaluate on the tuning half)\n")
P("Selected tau = %.1f; GM regret on the other half %.3f; problems with regret > 2: %d of %d.\n" % (TAUS[ks], gm(rs), int((rs > 2).sum()), len(rs)))
# ---- leave one problem out
lo = []; P("## Leave-one-problem-out (tau chosen on the other 11 problems by the same rule)\n")
P("| left-out problem | chosen tau | regret on it |\n|---|---|---|")
for p in PROBS:
    kk, _ = select([q for q in PROBS if q != p], M); r = regret(M[p])[kk]; lo.append((p, TAUS[kk], r)); P("| %s | %.1f | %.2f |" % (p, TAUS[kk], r))
rl = np.array([x[2] for x in lo]); P("\nLOPO geometric-mean regret %.3f; problems with regret > 2: %d of 12; worst %.2f (%s)." % (gm(rl), int((rl > 2).sum()), rl.max(), lo[int(np.argmax(rl))][0]))
# ---- fixed values, all 12 problems
P("\n## Regret of each fixed tau over all twelve problems (in-sample; the minimum over the grid is optimistic)\n")
P("| tau | GM regret | worst regret (problem) | problems with regret > 2 |\n|---|---|---|---|")
allr = np.array([regret(M[p]) for p in PROBS])          # (12, tau)
for i, t in enumerate(TAUS):
    w = int(np.argmax(allr[:, i])); P("| %.1f | %.3f | %.2f (%s) | %d |" % (t, gm(allr[:, i]), allr[w, i], PROBS[w], int((allr[:, i] > 2).sum())))
i04 = TAUS.index(0.4)
P("\nThe previously reported tau = 0.4: GM regret %.3f, regret > 2 on %d of 12 problems." % (gm(allr[:, i04]), int((allr[:, i04] > 2).sum())))
open("../results/tau_calib.md", "w", encoding="utf-8").write("\n".join(L)); print("\n".join(L))
json.dump(dict(selected=TAUS[k], selected_swap=TAUS[ks], gm_heldout=gm(rh), n_heldout_gt2=int((rh > 2).sum()), defensible=bool(ok), lopo_gm=gm(rl), lopo_n_gt2=int((rl > 2).sum()),
               selection_freq={str(t): float(c / NB) for t, c in zip(TAUS, cnt)}), open("../results/tau_calib.json", "w"), indent=1)

# ---- LaTeX table (all 12 problems)
def sci(x, d=2):
    m, e = ("%.*e" % (d, x)).split("e"); return "%se%d" % (m, int(e))
rows = []
for p in PROBS:
    b = int(np.argmin(M[p])); cs = []
    for i in range(len(TAUS)):
        top = sci(M[p][i]); top = "\\textbf{%s}" % top if i == b else top
        cs.append("\\makecell{%s\\\\{[}%s, %s{]}}" % (top, sci(CI[p][0][i], 1), sci(CI[p][1][i], 1)))
    rows.append("%s & %s & %s \\\\" % (p, "T" if p in TUNE else "H", " & ".join(cs)))
tex = ("\\begin{table}[htbp]\n\\centering\n\\caption{Sensitivity of the median IGD of MO-PEO to the deferred-refinement threshold $\\tau$ on 40 calibration seeds (6000--6039, not the protocol seeds). "
       "Median with 95\\% bootstrap interval of the median (10{,}000 resamples of the seeds); best median per problem in bold. $\\tau=1$ disables the schedule. "
       "T: problem in the tuning half of the split fixed before the runs; H: held-out half. The value of $\\tau$ was selected on T only (Section~\\ref{sec:tau}).}\n\\label{tab:tau_calib}\n"
       "\\resizebox{\\textwidth}{!}{\\begin{tabular}{llcccccccc}\n\\toprule\nProblem & & " + " & ".join("$%s$" % ("%g" % t) for t in TAUS) + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}")
open("../paper/table_tau_calib.tex", "w", encoding="utf-8").write(tex)

# ---- figure: greyscale heatmap of log2 regret
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8, "axes.linewidth": 0.8, "figure.dpi": 600, "savefig.dpi": 600, "mathtext.fontset": "dejavuserif"})
order = TUNE + HELD
Z = np.log2(np.array([regret(M[p]) for p in order]))
fig, ax = plt.subplots(figsize=(4.6, 4.2))
im = ax.imshow(Z, cmap="Greys", aspect="auto", vmin=0, vmax=max(3, Z.max()))
for i, p in enumerate(order):
    for j in range(len(TAUS)):
        r = 2 ** Z[i, j]; ax.text(j, i, "%.1f" % r if r < 10 else "%.0f" % r, ha="center", va="center", fontsize=6, color="white" if Z[i, j] > 0.55 * max(3, Z.max()) else "black")
    ax.plot([TAUS.index(TAUS[np.argmin(M[p])])], [i], marker="s", ms=8, mfc="none", mec="black", mew=0.9)
ax.set_xticks(range(len(TAUS))); ax.set_xticklabels(["%g" % t for t in TAUS]); ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
ax.axhline(len(TUNE) - 0.5, color="black", lw=1.2); ax.set_xlabel(r"$\tau$"); ax.tick_params(length=2)
ax.set_xticks(np.arange(-.5, len(TAUS), 1), minor=True); ax.set_yticks(np.arange(-.5, len(order), 1), minor=True); ax.grid(which="minor", color="white", lw=0.6); ax.tick_params(which="minor", length=0)
ax.text(1.02, (len(TUNE) - 1) / 2 / (len(order) - 1) * 0 + 0.75, "tuning", transform=ax.transAxes, rotation=90, va="center", fontsize=7)
ax.text(1.02, 0.25, "held-out", transform=ax.transAxes, rotation=90, va="center", fontsize=7)
ax.plot([TAUS.index(TAUS[k])] * 2, [-0.5, len(order) - 0.5], color="black", lw=0.0)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.10); cb.set_label(r"$\log_2$ regret ratio", fontsize=7); cb.ax.tick_params(labelsize=6)
ax.set_title("regret of each $\\tau$ (boxed: best $\\tau$ of the problem)", fontsize=8, loc="left")
fig.savefig("../results/tau_calib.png", dpi=600, bbox_inches="tight")
