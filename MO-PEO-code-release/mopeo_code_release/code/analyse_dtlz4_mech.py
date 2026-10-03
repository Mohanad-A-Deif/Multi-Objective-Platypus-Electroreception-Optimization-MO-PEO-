#!/usr/bin/env python3
"""Summarise results/diag_dtlz4_mech.pkl (diag_dtlz4_mech.py): report results/diag_dtlz4_mech.md and greyscale figure results/diag_dtlz4_mech.png."""
import pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R_ = pickle.load(open("../results/diag_dtlz4_mech.pkl", "rb"))
seeds = sorted(R_); n = len(seeds)
VAR = ["full", "no_att", "no_E", "no_noise", "dense", "rnd_lead"]
NAME = {"full": "reported move", "no_att": "no leader attraction (gamma = lambda = 0)", "no_E": "no directional-sensing term", "no_noise": "no noise term",
        "dense": "noise on every coordinate", "rnd_lead": "random archive member as leader"}
UNI = (0.99949 - 0.97049) ** 2          # uniform sampling: probability that a candidate is interior
NS, NR = 91, None
rng = np.random.default_rng(1)
fin = np.array([R_[s]["final_igd"] for s in seeds]); failed = fin > 0.2
T = len(R_[seeds[0]]["tr"])
first = np.array([next((r["t"] for r in R_[s]["tr"] if r["e_raw_int"] + r["x_raw_int"] > 0), np.nan) for s in seeds])   # first realised interior offspring
disc = ~np.isnan(first)
L = ["# DTLZ4: where the interior of the front is lost in the update rule of MO-PEO (observation only)\n",
     "MO-PEO as reported (sparse, delayed refinement, tau = 0.1), %d runs on seeds %d..%d, %d iterations of 91 evaluations. Interior = x1, x2 in (0.9705, 0.99949) (both coordinates). "
     "Uniform random sampling puts a candidate in the interior with probability %.3e (%.1f interior candidates per run of %d evaluations).\n" % (n, seeds[0], seeds[-1], T, UNI, UNI * T * NS, T * NS),
     "Failure (final IGD > 0.2): %d of %d runs. Runs with at least one interior offspring during the run: %d (failed among them: %d).\n" % (failed.sum(), n, disc.sum(), (failed & disc).sum())]

# ---------------------------------------------------------------- 1 projection
tot = {k: sum(sum(r[b + k] for b in "ex") for s in seeds for r in R_[s]["tr"]) for k in ("_raw_int", "_clip_int", "_proj_gain", "_proj_loss", "_over1", "_over2")}
ncand = T * NS * n
L.append("## 1. Are interior candidates generated and then projected away?\n")
L.append("Candidates over all runs: %d. Interior before the projection to the box: %d; interior after it: %d; candidates made interior by the projection: %d; candidates made non-interior by the projection: %d. "
         "The clipping acts coordinate by coordinate and is the identity on the window (which lies inside the box), so it can neither remove nor create an interior candidate: %d of them exist before and after. "
         "The projection does act elsewhere: x1 leaves the box in %.1f%% of candidates and x2 in %.1f%%.\n" % (ncand, tot["_raw_int"], tot["_clip_int"], tot["_proj_gain"], tot["_proj_loss"], tot["_raw_int"], 100 * tot["_over1"] / ncand, 100 * tot["_over2"] / ncand))

# ---------------------------------------------------------------- 2 realised interior candidates by branch
L.append("## 2. Realised interior candidates, by branch\n")
L.append("| runs | offspring, exploration (s = 0) | interior | offspring, exploitation (s = 1) | interior |\n|---|---|---|---|---|")
for nm, m in (("failed", failed), ("not failed", ~failed)):
    ne = sum(r["n_exp"] for s, k in zip(seeds, m) if k for r in R_[s]["tr"]); nx = sum(r["n_expl"] for s, k in zip(seeds, m) if k for r in R_[s]["tr"])
    ie = sum(r["e_raw_int"] for s, k in zip(seeds, m) if k for r in R_[s]["tr"]); ix = sum(r["x_raw_int"] for s, k in zip(seeds, m) if k for r in R_[s]["tr"])
    L.append("| %s (n=%d) | %d | %d | %d | %d |" % (nm, m.sum(), ne, ie, nx, ix))
L.append("")

# ---------------------------------------------------------------- 3 counterfactual rates
def cfsum(seed_mask, upto_first):
    """sum of counterfactual counts over the iterations of the selected runs; if upto_first, only the iterations before the first realised interior offspring."""
    acc = np.zeros((len(VAR), 2, 4)); 
    for s, k, f in zip(seeds, seed_mask, first):
        if not k:
            continue
        for r in R_[s]["tr"]:
            if upto_first and (not np.isnan(f)) and r["t"] >= f:
                break
            acc += r["cf"]
    return acc

def boot_rate(seed_mask, vi, upto_first, col=1, br=None, nb=2000):
    idx = [i for i, k in enumerate(seed_mask) if k]
    per = []
    for i in idx:
        s = seeds[i]; f = first[i]; a = np.zeros((2, 4))
        for r in R_[s]["tr"]:
            if upto_first and (not np.isnan(f)) and r["t"] >= f:
                break
            a += r["cf"][vi]
        per.append(a)
    per = np.array(per)                                       # runs x branch x 4
    sel = per.sum(1) if br is None else per[:, br]
    est = sel[:, col].sum() / sel[:, 0].sum()
    bs = []
    for _ in range(nb):
        j = rng.integers(0, len(idx), len(idx)); bs.append(sel[j, col].sum() / max(1, sel[j, 0].sum()))
    return est, np.percentile(bs, [2.5, 97.5])

L.append("## 3. Probability that a candidate is interior, from the states visited (counterfactual draws, same state, separate random stream)\n")
L.append("Rates per candidate, pooled over the iterations before the first realised interior offspring (failed runs: the whole run), with 95%% bootstrap intervals over runs. "
         "Uniform sampling: %.2e. Each row changes ONE term of the move and keeps the others; nothing here is a proposal.\n" % UNI)
L.append("| move | all branches: failed runs | ... rate / uniform | all branches: not-failed runs, before discovery | exploration branch, failed | exploitation branch, failed |\n|---|---|---|---|---|---|")
rates = {}
for vi, v in enumerate(VAR):
    rf, cf_ = boot_rate(failed, vi, True); rn, cn = boot_rate(~failed, vi, True); re_, ce = boot_rate(failed, vi, True, br=0); rx, cx = boot_rate(failed, vi, True, br=1)
    rates[v] = dict(failed=rf, notfailed=rn, e=re_, x=rx)
    L.append("| %s | %.2e [%.1e, %.1e] | %.2f | %.2e [%.1e, %.1e] | %.2e | %.2e |" % (NAME[v], rf, cf_[0], cf_[1], rf / UNI, rn, cn[0], cn[1], re_, rx))
L.append("")

L.append("**Marginal against joint (reported move, failed runs).** Probability that a candidate has x1 in the window, x2 in the window, both, and the product of the marginals:\n")
L.append("| branch | P(x1 in window) | P(x2 in window) | product | P(both) | joint / product |\n|---|---|---|---|---|---|")
for br, nm in ((None, "all"), (0, "exploration"), (1, "exploitation")):
    p1, _ = boot_rate(failed, 0, True, col=2, br=br, nb=2); p2, _ = boot_rate(failed, 0, True, col=3, br=br, nb=2); pj, _ = boot_rate(failed, 0, True, col=1, br=br, nb=2)
    L.append("| %s | %.3f | %.3f | %.2e | %.2e | %.3f |" % (nm, p1, p2, p1 * p2, pj, pj / (p1 * p2) if p1 * p2 > 0 else float("nan")))
L.append("(For uniform sampling each marginal is %.4f and the joint equals the product.)\n" % (0.99949 - 0.97049))

# ---------------------------------------------------------------- 4 cumulative expected hits
L.append("## 4. Expected number of interior candidates over a run (integrated hazard)\n")
Rrep = None
H = {}
for vi, v in enumerate(VAR):
    hs = []
    for s, f in zip(seeds, first):
        a = 0.0
        for r in R_[s]["tr"]:
            if (not np.isnan(f)) and r["t"] >= f:
                break
            a += r["cf"][vi][:, 1].sum() / (r["cf"][vi][:, 0].sum() / NS)      # expected interior candidates in this iteration (per NS agents)
        hs.append(a)
    H[v] = np.array(hs)
L.append("H = expected number of interior candidates before the first realised one (whole run if none), summed over iterations from the counterfactual draws; a well-calibrated hazard makes H at the first event about 1 (exponential with mean 1).\n")
L.append("| move | median H, discovered runs (n=%d) | median H, runs without discovery (n=%d) | mean of exp(-H) over all runs (predicted fraction without discovery) |\n|---|---|---|---|" % (disc.sum(), (~disc).sum()))
for v in VAR:
    L.append("| %s | %.2f | %.2f | %.2f |" % (NAME[v], np.median(H[v][disc]) if disc.any() else np.nan, np.median(H[v][~disc]) if (~disc).any() else np.nan, np.mean(np.exp(-H[v]))))
L.append("\nObserved fraction of runs without any interior offspring: %.2f (%d of %d). Uniform sampling would give H = %.1f over a whole run.\n" % ((~disc).mean(), (~disc).sum(), n, UNI * T * NS))

# ---------------------------------------------------------------- 5 state
L.append("## 5. The state the moves start from (mean over runs of the fraction of agents and leaders whose coordinate is in the window)\n")
L.append("| run group | budget | agents x1 | agents x2 | leaders x1 | leaders x2 | agents both |\n|---|---|---|---|---|---|---|")
for nm, m in (("failed", failed), ("not failed", ~failed)):
    for c in (0.02, 0.20, 1.0):
        i = min(T - 1, int(round(c * T)) - 1)
        g = lambda k: np.mean([R_[s]["tr"][i][k] for s, kk in zip(seeds, m) if kk])
        L.append("| %s | %d%% | %.3f | %.3f | %.3f | %.3f | %.3f |" % (nm, c * 100, g("ag_w1"), g("ag_w2"), g("ld_w1"), g("ld_w2"), g("ag_int")))
open("../results/diag_dtlz4_mech.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
                     "legend.fontsize": 7, "figure.dpi": 600, "savefig.dpi": 600, "mathtext.fontset": "dejavuserif", "axes.labelsize": 8, "axes.titlesize": 8})
fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(7.2, 2.9))
x = np.arange(1, T + 1)
def cum(s, vi=0):
    return np.cumsum([r["cf"][vi][:, 1].sum() / (r["cf"][vi][:, 0].sum() / NS) for r in R_[s]["tr"]])
for grp_, nm, st in ((failed & ~disc, "no interior offspring (n=%d)" % (failed & ~disc).sum(), dict(color="black", ls="-")), (disc, "interior offspring found (n=%d)" % disc.sum(), dict(color="0.5", ls="--"))):
    C = np.array([cum(s) for s, k in zip(seeds, grp_) if k])
    if len(C):
        axs[0].plot(x, np.median(C, 0), lw=1.3, label=nm, **st); axs[0].fill_between(x, np.percentile(C, 25, 0), np.percentile(C, 75, 0), color=st["color"], alpha=0.15, lw=0)
axs[0].plot(x, UNI * NS * x, color="0.75", lw=1.0, ls=":", label="uniform sampling")
axs[0].set_yscale("log"); axs[0].set_ylim(1e-3, 3e3); axs[0].set_xlabel("iteration"); axs[0].set_ylabel("expected interior candidates so far"); axs[0].legend(loc="upper left", fontsize=6)
vals = [rates[v]["failed"] for v in VAR]; xs = np.arange(len(VAR))
axs[1].bar(xs, vals, color=["black"] + ["0.55"] * (len(VAR) - 1), width=0.65); axs[1].axhline(UNI, color="0.4", lw=1.0, ls=":")
axs[1].set_yscale("log"); axs[1].set_xticks(xs); axs[1].set_xticklabels(["reported", "no attr.", "no E", "no noise", "dense noise", "rand. leader"], fontsize=6.5, rotation=40, ha="right")
axs[1].set_ylabel("P(candidate is interior)"); axs[1].text(len(VAR) - 0.5, UNI * 1.25, "uniform", ha="right", fontsize=6.5, color="0.3")
axs[2].step(np.r_[0, np.sort(first[disc]), T], np.r_[0, np.arange(1, disc.sum() + 1), disc.sum()] / n, where="post", color="black", lw=1.3)
axs[2].set_xlabel("iteration"); axs[2].set_ylabel("runs with an interior offspring"); axs[2].set_ylim(0, 1.0)
for ax, tg in zip(axs, "abc"):
    ax.text(-0.22, 1.04, tg, transform=ax.transAxes, fontsize=10, fontweight="bold"); ax.tick_params(top=True, right=True)
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.2, top=0.95, wspace=0.42)
fig.savefig("../results/diag_dtlz4_mech.png", dpi=600, bbox_inches="tight")
