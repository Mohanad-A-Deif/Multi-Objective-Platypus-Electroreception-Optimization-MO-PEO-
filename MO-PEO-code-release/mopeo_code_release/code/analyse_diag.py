#!/usr/bin/env python3
"""Summarise results/diag_dtlz2.pkl (written by diag_dtlz2.py): report + greyscale figure."""
import pickle, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from problems import dtlz_pf
from diag_dtlz2 import set_metrics, dir_dist
from algorithms import weights

import glob
d = dict(mopeo=[], moead=[], final=[])
for fn in sorted(glob.glob("../results/diag_dtlz2_*.pkl")):
    part = pickle.load(open(fn, "rb"))
    for k in d: d[k] += part[k]
R = len(d["mopeo"]); T = min(len(t) for t in d["mopeo"])
P = dtlz_pf("DTLZ2")
out = []
def say(s=""):
    out.append(s); print(s)


def series(runs, key):
    return np.array([[rec[key] for rec in tr[:T]] for tr in runs if key in tr[0]])


def med(runs, key):
    a = series(runs, key); return np.median(a, 0) if len(a) else None


fe = np.array([rec["fe"] for rec in d["mopeo"][0][:T]])
cps = [0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
idx = [min(T - 1, int(c * T) - 1) for c in cps]
say("# DTLZ2 diagnosis: instrumented runs, %d runs (seeds 5000..%d), %d iterations of N = 91 evaluations\n" % (R, 5000 + R - 1, T))
say("Medians over runs at checkpoints (fraction of the budget). MO-PEO = final configuration, MOEA/D = reference.\n")

def table(title, cols, fmt="%.4g"):
    say("**%s**\n" % title)
    say("| budget | " + " | ".join(str(int(c * 100)) + "%" for c in cps) + " |")
    say("|---|" + "---|" * len(cps))
    for name, runs, key in cols:
        m = med(runs, key)
        if m is None: continue
        say("| %s | " % name + " | ".join(fmt % m[i] for i in idx) + " |")
    say()

M_, D_ = d["mopeo"], d["moead"]
table("Approximation quality (IGD, lower is better) and its split",
      [("MO-PEO archive IGD", M_, "igd"), ("MOEA/D non-dominated set IGD", D_, "igd"),
       ("MO-PEO IGD of the union before truncation", M_, "igd_pre"), ("MO-PEO IGD if the union were cut by iterative crowding", M_, "igd_cf"),
       ("MO-PEO convergence gd", M_, "gd"), ("MOEA/D convergence gd", D_, "gd"),
       ("MO-PEO largest hole (gap)", M_, "gap"), ("MOEA/D largest hole (gap)", D_, "gap"),
       ("MO-PEO gap of the union before truncation", M_, "gap_pre"), ("MO-PEO gap if cut by iterative crowding", M_, "gap_cf")])
table("Size and truncation (MO-PEO archive)",
      [("archive size", M_, "size"), ("size of the non-dominated union before truncation", M_, "pre_size"),
       ("iterations with truncation (fraction of runs)", M_, "trunc"), ("members removed", M_, "cut"), ("members of the old archive lost (churn)", M_, "churn")], "%.4g")
say("Fraction of ALL iterations in which truncation was triggered: %.3f; mean members removed per triggering iteration: %.1f; mean churn: %.1f\n"
    % (series(M_, "trunc").mean(), series(M_, "cut")[series(M_, "trunc") > 0].mean(), series(M_, "churn").mean()))
table("Leaders (MO-PEO, 91 agents)",
      [("distinct archive members used as leaders", M_, "leaders"), ("largest number of agents sharing one leader", M_, "lead_max"),
       ("archive members that lead nobody", M_, "idle"), ("archive size (before update)", M_, "archive_before"),
       ("MOEA/D: distinct solutions in the population", D_, "distinct")])
table("Tchebycheff values of the agents, phi(x_i|w_i), normalised",
      [("MO-PEO mean", M_, "phi_mean"), ("MO-PEO std", M_, "phi_std"), ("MO-PEO max", M_, "phi_max"),
       ("MOEA/D mean", D_, "phi_mean"), ("MOEA/D std", D_, "phi_std"), ("MOEA/D max", D_, "phi_max")])
table("Excess over the best attainable value, e_i = phi(x_i|w_i) - min_p phi(p|w_i)",
      [("MO-PEO agents: median", M_, "exc_med"), ("MO-PEO agents: 90th pct", M_, "exc_p90"), ("MO-PEO agents: agents with e > 0.05", M_, "unserved"),
       ("MO-PEO leaders: median", M_, "excL_med"), ("MO-PEO leaders: 90th pct", M_, "excL_p90"), ("MO-PEO leaders: weights with e > 0.05", M_, "unservedL"),
       ("MOEA/D subproblems: median", D_, "exc_med"), ("MOEA/D subproblems: 90th pct", D_, "exc_p90"), ("MOEA/D subproblems: with e > 0.05", D_, "unserved")])
table("Distance between the leader / subproblem solution of weight i and the Tchebycheff-optimal point of that weight",
      [("MO-PEO leaders: mean", M_, "tgt_mean"), ("MO-PEO leaders: max", M_, "tgt_max"),
       ("MOEA/D: mean", D_, "tgt_mean"), ("MOEA/D: max", D_, "tgt_max")])
table("Distance from the direction of each weight vector to the nearest set member",
      [("MO-PEO mean", M_, "dir_mean"), ("MO-PEO max", M_, "dir_max"), ("MOEA/D mean", D_, "dir_mean"), ("MOEA/D max", D_, "dir_max")])
table("Spacing of the set (nearest-neighbour distance)",
      [("MO-PEO mean", M_, "nn_mean"), ("MO-PEO std", M_, "nn_std"), ("MO-PEO max", M_, "nn_max"),
       ("MOEA/D mean", D_, "nn_mean"), ("MOEA/D std", D_, "nn_std"), ("MOEA/D max", D_, "nn_max")])
table("Exploitation fraction s (MO-PEO)", [("fraction of agents with s_i = 1", M_, "s")], "%.3f")

table("Does crowding truncation discard the better-converged members? (MO-PEO, iterations with truncation)",
      [("gd of the members kept (whole archive)", M_, "gd_kept"), ("gd of the members removed", M_, "gd_removed"),
       ("offspring that enter the archive, per iteration", M_, "entered")])
table("Which agents produce the archive entries? (MO-PEO)  n = agents per group, imp = fraction whose own phi improved, ent = fraction whose offspring is in the archive",
      [("exploitation, relative noise: n", M_, "n_rel"), ("   imp", M_, "imp_rel"), ("   ent", M_, "ent_rel"),
       ("exploitation, absolute noise: n", M_, "n_abs"), ("   imp", M_, "imp_abs"), ("   ent", M_, "ent_abs"),
       ("exploration (E + M + R): n", M_, "n_exp"), ("   imp", M_, "imp_exp"), ("   ent", M_, "ent_exp")], "%.3g")
table("Distance variables x_j, j >= 3 (DTLZ2 optimum 0.5): mean |x_j - 0.5|",
      [("archive members", M_, "xdev_arch"), ("agents", M_, "xdev_agents"),
       ("median relative-branch step of an agent |x - a| on the perturbed coordinate", M_, "relstep_med"), ("probability p_rel of the relative branch", M_, "pr")])
say("(the absolute-branch step is zeta * 0.1 * (U - L) = 0.09 on the perturbed coordinate, constant over the run; the perturbation M has scale beta * 0.1 = 0.1)\n")

# ---------------- what is intrinsic to weight-coherent leader selection on this front?
from algorithms import trunc_archive
Wc = np.maximum(weights(3, 91), 1e-6)
def distinct_leaders(A, W):
    zmin, zmax = A.min(0), A.max(0); r = np.maximum(zmax - zmin, 1e-12)
    nrm = (A - zmin) / (r + 1e-6)
    li = ((W[:, None, :] * nrm[None]).max(2)).argmin(1)
    return len(np.unique(li))
from problems import das_dennis
Ideal = das_dennis(3, 12); Ideal = Ideal / np.linalg.norm(Ideal, axis=1, keepdims=True)
say("**Distinct leaders that the Tchebycheff assignment gives when the archive is ideal**\n")
say("- archive = 91 points of the true front spaced as a Das-Dennis grid projected to the sphere: %d distinct leaders for 91 weights" % distinct_leaders(Ideal, Wc))
dl = [distinct_leaders(f["MOEAD"], Wc) for f in d["final"]]
say("- archive = the final MOEA/D non-dominated set (median size %d): %d distinct leaders (median over runs)" % (np.median([len(f["MOEAD"]) for f in d["final"]]), np.median(dl)))
dl = [distinct_leaders(f["MOPEO"], Wc) for f in d["final"]]
say("- archive = the final MO-PEO archive (91): %d distinct leaders (median over runs)" % np.median(dl))
Pf = P[::7]; say("- archive = 720 points of the true front: %d distinct leaders\n" % distinct_leaders(Pf, Wc))

# ---------------- final sets, all three algorithms
say("**Final sets (end of run), medians over runs**\n")
say("| algorithm | size | IGD | gd | gap | nn mean | nn std | nn max | dir mean | dir max |")
say("|---|---|---|---|---|---|---|---|---|---|")
W = weights(3, 91)
for name, key in (("MO-PEO", "MOPEO"), ("MOEA/D", "MOEAD"), ("NSGA-II", "NSGA2")):
    rows = []
    for f in d["final"]:
        A = f[key]; m = set_metrics(A, P); dm = dir_dist(A, np.maximum(W, 1e-6))
        rows.append([m["size"], m["igd"], m["gd"], m["gap"], m["nn_mean"], m["nn_std"], m["nn_max"], dm[0], dm[1]])
    r = np.median(np.array(rows), 0)
    say("| %s | %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |" % ((name, r[0]) + tuple(r[1:])))
say()

# ---------------- IGD after projecting every member radially onto the true front (removes the convergence part)
say("**Convergence versus coverage in the final IGD (medians over runs)**\n")
say("| algorithm | IGD | IGD after radial projection of the members onto the sphere | gd |")
say("|---|---|---|---|")
for name, key in (("MO-PEO", "MOPEO"), ("MOEA/D", "MOEAD"), ("NSGA-II", "NSGA2")):
    v1, v2, v3 = [], [], []
    for f in d["final"]:
        A = f[key]; Ap = A / np.linalg.norm(A, axis=1, keepdims=True)
        v1.append(set_metrics(A, P)["igd"]); v2.append(set_metrics(Ap, P)["igd"]); v3.append(set_metrics(A, P)["gd"])
    say("| %s | %.4f | %.4f | %.4f |" % (name, np.median(v1), np.median(v2), np.median(v3)))
say()

# ---------------- where are the holes: intrinsic to the weights, or specific to MO-PEO?
from diag_dtlz2 import Target
tgt = Target(Wc, P); Tset = np.unique(np.round(tgt.pt, 9), axis=0)
def depth(A): return np.sqrt(((P[:, None] - A[None]) ** 2).sum(2)).min(1)
hT = depth(Tset)
say("**Holes of the final sets: fraction of true-front points farther than a threshold from the nearest member (medians over runs)**\n")
say("| set | > 0.10 | > 0.15 | > 0.20 |")
say("|---|---|---|---|")
for name, key in (("MO-PEO", "MOPEO"), ("MOEA/D", "MOEAD"), ("NSGA-II", "NSGA2")):
    hs = np.array([[np.mean(depth(f[key] / np.linalg.norm(f[key], axis=1, keepdims=True)) > t) for t in (0.10, 0.15, 0.20)] for f in d["final"]])
    say("| %s (radially projected) | %s |" % (name, " | ".join("%.3f" % v for v in np.median(hs, 0))))
say("| Tchebycheff targets of the 91 weights (%d distinct) | %s |" % (len(Tset), " | ".join("%.3f" % np.mean(hT > t) for t in (0.10, 0.15, 0.20))))
say("| perfect uniform 91-point set | %s |" % " | ".join("%.3f" % np.mean(depth(Ideal) > t) for t in (0.10, 0.15, 0.20)))
fr = []
for f in d["final"]:
    A = f["MOPEO"] / np.linalg.norm(f["MOPEO"], axis=1, keepdims=True); hA = depth(A)
    hole = hA > 0.12
    fr.append([hole.mean(), (hT[hole] > 0.10).mean() if hole.any() else np.nan])
fr = np.array(fr)
say("\nMO-PEO (projected): %.1f%% of the front lies farther than 0.12 from every member; of those points, %.0f%% also lie farther than 0.10 from every Tchebycheff target (median over runs), i.e. in a region no weight vector points at.\n"
    % (100 * np.median(fr[:, 0]), 100 * np.nanmedian(fr[:, 1])))

# ---------------- across runs: does more leader concentration go with worse coverage?
from scipy.stats import spearmanr
half = T // 2
lead_late = np.array([np.mean([rec["leaders"] for rec in tr[half:]]) for tr in d["mopeo"]])
churn_late = np.array([np.mean([rec["churn"] for rec in tr[half:]]) for tr in d["mopeo"]])
igd_end = np.array([set_metrics(f["MOPEO"], P)["igd"] for f in d["final"]])
igd_proj = np.array([set_metrics(f["MOPEO"] / np.linalg.norm(f["MOPEO"], axis=1, keepdims=True), P)["igd"] for f in d["final"]])
gap_proj = np.array([set_metrics(f["MOPEO"] / np.linalg.norm(f["MOPEO"], axis=1, keepdims=True), P)["gap"] for f in d["final"]])
say("**Across the %d runs (Spearman rho, p):**\n" % len(igd_end))
for nm, xv in (("mean distinct leaders, second half of the run", lead_late), ("mean churn, second half", churn_late)):
    for yn, yv in (("final IGD", igd_end), ("projected IGD (coverage)", igd_proj), ("largest hole after projection", gap_proj)):
        rho, p = spearmanr(xv, yv); say("- %s vs %s: rho = %+.2f (p = %.2f)" % (nm, yn, rho, p))
say()

# ---------------- the last-fifth trajectory of IGD: is the gap still closing?
a, b = med(M_, "igd"), med(D_, "igd")
say("IGD gap MO-PEO / MOEA/D (medians): at 40%% of the budget %.2f, 60%% %.2f, 80%% %.2f, 100%% %.2f" % tuple(a[idx[i]] / b[idx[i]] for i in (4, 5, 6, 7)))
say("MO-PEO IGD decrease over the last 20%% of the budget: %.4f -> %.4f\n" % (a[idx[6]], a[idx[7]]))

open("../results/diag_dtlz2.md", "w", encoding="utf-8").write("\n".join(out) + "\n")

# ---------------- figure: greyscale, one column, shared x
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11, "axes.edgecolor": "black", "axes.linewidth": 0.8,
                     "axes.labelsize": 11, "xtick.direction": "in", "ytick.direction": "in", "xtick.major.size": 4, "ytick.major.size": 4,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5, "legend.frameon": False, "grid.alpha": 0.25,
                     "grid.linestyle": ":", "grid.color": "black", "figure.facecolor": "white", "axes.facecolor": "white",
                     "mathtext.fontset": "dejavuserif", "figure.dpi": 600, "savefig.dpi": 600})
ST = {"mopeo": dict(color="black", ls="-", lw=1.0), "moead": dict(color="0.65", ls=":", lw=1.2),
      "pre": dict(color="0.35", ls="--", lw=1.0), "cf": dict(color="0.5", ls="-.", lw=1.0)}
fig, ax = plt.subplots(5, 1, figsize=(3.4, 12.5), sharex=True)
x = fe
def band(a, runs, key, st, label):
    s = series(runs, key); ax_.plot(x, np.median(s, 0), label=label, **st)
ax_ = ax[0]; band(ax_, M_, "igd", ST["mopeo"], "MO-PEO archive"); band(ax_, D_, "igd", ST["moead"], "MOEA/D")
band(ax_, M_, "igd_pre", ST["pre"], "MO-PEO union before truncation"); band(ax_, M_, "igd_cf", ST["cf"], "same union, iterative crowding cut")
ax_.set_yscale("log"); ax_.set_ylabel("IGD"); ax_.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, fontsize=8, columnspacing=1.0, handlelength=2.2)
ax_ = ax[1]; band(ax_, M_, "size", ST["mopeo"], "archive size"); band(ax_, M_, "leaders", ST["pre"], "distinct leaders")
band(ax_, M_, "pre_size", ST["cf"], "union before truncation"); band(ax_, D_, "distinct", ST["moead"], "MOEA/D distinct solutions")
ax_.set_ylabel("members"); ax_.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, fontsize=8, columnspacing=1.0, handlelength=2.2)
ax_ = ax[2]; band(ax_, M_, "gd", ST["mopeo"], "MO-PEO gd"); band(ax_, D_, "gd", ST["moead"], "MOEA/D gd")
band(ax_, M_, "gap", ST["pre"], "MO-PEO largest hole"); band(ax_, D_, "gap", ST["cf"], "MOEA/D largest hole")
ax_.set_yscale("log"); ax_.set_ylabel("distance to front"); ax_.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, fontsize=8, columnspacing=1.0, handlelength=2.2)
ax_ = ax[3]; band(ax_, M_, "tgt_mean", ST["mopeo"], "MO-PEO leader"); band(ax_, D_, "tgt_mean", ST["moead"], "MOEA/D solution")
band(ax_, M_, "dir_mean", ST["pre"], "MO-PEO weight direction"); band(ax_, D_, "dir_mean", ST["cf"], "MOEA/D weight direction")
ax_.set_yscale("log"); ax_.set_ylabel("mean distance"); ax_.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, fontsize=8, columnspacing=1.0, handlelength=2.2)
ax_ = ax[4]; band(ax_, M_, "s", ST["mopeo"], "exploitation fraction $s$"); band(ax_, M_, "trunc", ST["pre"], "truncation triggered (fraction of runs)")
ax_.set_ylabel("fraction"); ax_.set_xlabel("Function evaluations"); ax_.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, fontsize=8, columnspacing=1.0, handlelength=2.2)
for i, a_ in enumerate(ax):
    a_.grid(True); a_.text(0.02, 0.05 if i not in (0,) else 0.05, "(%s)" % "abcde"[i], transform=a_.transAxes, ha="left", va="bottom",
                           bbox=dict(fc="white", ec="none", alpha=0.75, pad=2))
fig.subplots_adjust(left=0.19, right=0.97, top=0.965, bottom=0.05, hspace=0.62)
fig.savefig("../results/diag_dtlz2.png", dpi=600, bbox_inches="tight")
print("wrote diag_dtlz2.md, diag_dtlz2.png")
