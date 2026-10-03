#!/usr/bin/env python3
"""Summarise results/diag_dtlz4.pkl (diag_dtlz4.py): report results/diag_dtlz4.md and greyscale figure results/diag_dtlz4.png."""
import pickle
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gzip, os
_f = "../results/diag_dtlz4.pkl"
R = pickle.load(open(_f, "rb")) if os.path.exists(_f) else pickle.load(gzip.open(_f + ".gz", "rb"))   # the shipped copy is gzip-compressed
ALGS = ["MO-PEO", "GDE3", "NSGA-II"]; FAIL = 0.2
seeds = sorted({k[1] for k in R}); n = len(seeds)
fin = {a: np.array([R[(a, s)]["final_igd"] for s in seeds]) for a in ALGS}
fail = {a: fin[a] > FAIL for a in ALGS}
T = len(R[("MO-PEO", seeds[0])]["tr"])
L = []; P = L.append


def col(a, s, key):
    return np.array([r.get(key, np.nan) for r in R[(a, s)]["tr"]], float)


def grp(a, mask, key):
    return np.array([col(a, s, key) for s, m in zip(seeds, mask) if m])


def at(x, frac):
    return x[:, min(x.shape[1] - 1, max(0, int(round(frac * x.shape[1])) - 1))]


CP = [0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
P("# DTLZ4 diagnosis: instrumented runs, %d seeds (%d..%d), %d iterations of N = 91 evaluations\n" % (n, seeds[0], seeds[-1], T))
P("Position on the front is fixed by u1 = x1^100 (elevation) and u2 = x2^100 (azimuth); 'interior' = 0.05 < u1, u2 < 0.95, i.e. x1, x2 > 0.9705 and not at the bound. "
  "Failure = final IGD > %.1f (the final IGD is bimodal: see below).\n" % FAIL)
P("## Failure rates and the bimodality of the final IGD\n")
P("| algorithm | failed runs | median IGD of failed runs | median IGD of the others | largest IGD of the others | smallest IGD of the failed |\n|---|---|---|---|---|---|")
for a in ALGS:
    f = fail[a]; P("| %s | %d/%d | %s | %s | %s | %s |" % (a, f.sum(), n, "%.4f" % np.median(fin[a][f]) if f.any() else "-", "%.4f" % np.median(fin[a][~f]),
                                                          "%.4f" % fin[a][~f].max(), "%.4f" % fin[a][f].min() if f.any() else "-"))
P("\nFisher exact test, failure counts MO-PEO vs NSGA-II: p = %.3f; MO-PEO vs GDE3: p = %.4f." % (
    stats.fisher_exact([[fail["MO-PEO"].sum(), n - fail["MO-PEO"].sum()], [fail["NSGA-II"].sum(), n - fail["NSGA-II"].sum()]])[1],
    stats.fisher_exact([[fail["MO-PEO"].sum(), n - fail["MO-PEO"].sum()], [fail["GDE3"].sum(), n - fail["GDE3"].sum()]])[1]))

# ---- what the failing archive looks like
P("\n## Composition of the final archive (medians over runs; MO-PEO, failed vs not failed; the baselines' non-dominated set)\n")
P("| set | n runs | size | interior members | low-elevation (f3 ~ 0) | pole | interior elevation, azimuth at an edge | IGD | IGD on interior part of the front | IGD on the edge part |\n|---|---|---|---|---|---|---|---|---|---|")
for a in ALGS:
    for nm, m in (("failed", fail[a]), ("not failed", ~fail[a])):
        if m.sum() == 0:
            continue
        g = lambda k: np.median(at(grp(a, m, k), 1.0))
        P("| %s, %s | %d | %.0f | %.0f | %.0f | %.0f | %.0f | %.4f | %.4f | %.4f |" % (a, nm, m.sum(), g("ar_n"), g("ar_int"), g("ar_lowel"), g("ar_pole"), g("ar_azedge"), g("ar_igd"), g("ar_igd_int"), g("ar_igd_edge")))

# ---- time series
def series(title, key, algs_masks, fmt="%.3g"):
    P("\n**%s**\n" % title)
    P("| run group | " + " | ".join("%d%%" % (c * 100) for c in CP) + " |\n|---|" + "---|" * len(CP))
    for nm, a, m in algs_masks:
        if m.sum() == 0:
            continue
        x = grp(a, m, key); P("| %s (n=%d) | " % (nm, m.sum()) + " | ".join(fmt % np.nanmedian(at(x, c)) for c in CP) + " |")

G = [("MO-PEO, failed", "MO-PEO", fail["MO-PEO"]), ("MO-PEO, not failed", "MO-PEO", ~fail["MO-PEO"]),
     ("GDE3", "GDE3", np.ones(n, bool)), ("NSGA-II, failed", "NSGA-II", fail["NSGA-II"]), ("NSGA-II, not failed", "NSGA-II", ~fail["NSGA-II"])]
P("\n## Archive / non-dominated set over the run (medians over runs)")
series("Interior members of the archive (of 91)", "ar_int", G, "%.0f")
series("IGD restricted to the interior part of the front (%.0f%% of the reference points)" % 0, "ar_igd_int", G, "%.3f")
series("Archive size", "ar_n", G, "%.0f")
P("\n## The population: where do the agents stand in the informative window? (agents before the move; medians over runs)")
series("Fraction of agents with x1 > 0.9705 (elevation above 4.5 degrees)", "ag_x1_alive", G, "%.3f")
series("Fraction of agents at x1 = 1 (bound)", "ag_x1_clip", G, "%.3f")
series("Fraction of agents in the interior of the front", "ag_int", G, "%.3f")
M_ = [x for x in G if x[1] == "MO-PEO"]
# ---- joint zones of (x1, x2): dead (x <= 0.9705), window (0.9705 < x < 1), bound (x = 1)
ZN = ["dead", "win", "bound"]
def joint_tab(prefix, title, groups, cps=(0.02, 0.20, 1.00)):
    P("\n**%s** (mean over runs of the fraction of members in each (x1 zone, x2 zone) cell; cells with less than 0.5%% everywhere are omitted)\n" % title)
    cells = [(i, j) for i in range(3) for j in range(3)]
    keep = []
    for (i, j) in cells:
        mx = 0
        for nm, a, m in groups:
            if m.sum() == 0 or (prefix + "j%d%d" % (i, j)) not in R[(a, seeds[0])]["tr"][0]:
                continue
            x = grp(a, m, prefix + "j%d%d" % (i, j))
            mx = max(mx, max(np.nanmean(at(x, c)) for c in cps))
        if mx >= 0.005:
            keep.append((i, j))
    P("| run group | budget | " + " | ".join("x1 %s, x2 %s" % (ZN[i], ZN[j]) for i, j in keep) + " |\n|---|---|" + "---|" * len(keep))
    for nm, a, m in groups:
        if m.sum() == 0 or (prefix + "j00") not in R[(a, seeds[0])]["tr"][0]:
            continue
        for c in cps:
            P("| %s (n=%d) | %d%% | " % (nm, m.sum(), c * 100) + " | ".join("%.3f" % np.nanmean(at(grp(a, m, prefix + "j%d%d" % (i, j)), c)) for i, j in keep) + " |")

P("\n## Joint occupancy of (x1, x2): the interior is the cell 'window, window'")
P("Interior of the front = both x1 and x2 in the window (0.9705, 1). A point at the bound (x = 1) is on an edge of the octant (elevation or azimuth pi/2); a point in the dead zone (x <= 0.9705) has u = x^100 < 0.05, i.e. sits within 4.5 degrees of the corresponding edge.")
joint_tab("ag_", "Agents (before the move)", G)
joint_tab("ld_", "Leaders assigned to the agents (MO-PEO only)", M_)
joint_tab("am_", "Archive members (MO-PEO only)", M_)
P("\n## MO-PEO mechanics: leaders, normalisation, phase switching, truncation (medians over runs)")
M_ = [x for x in G if x[1] == "MO-PEO"]
series("Fraction of agents whose leader is an interior member", "lead_int", M_, "%.3f")
series("Fraction of agents whose leader has low elevation (f3 ~ 0)", "lead_low", M_, "%.3f")
series("Distinct leaders (of 91 agents)", "lead_n", M_, "%.0f")
series("Exploitation fraction s", "s", M_, "%.3f")
series("Smallest objective range of the archive (normalisation denominator)", "rmin", M_, "%.3g")
series("Largest normalised objective value of any agent", "nrm_max", M_, "%.3g")
series("Members removed by truncation per iteration", "cut", M_, "%.1f")
series("Offspring per iteration that land in the interior of the front", "off_int", M_, "%.1f")
series("... of which enter the archive", "off_int_kept", M_, "%.1f")

# ---- where do interior offspring come from
P("\n## Which branch produces interior offspring? (MO-PEO, all runs, all iterations)\n")
P("| branch | offspring | interior offspring | rate per 1000 offspring | share of interior offspring |\n|---|---|---|---|---|")
tot = {b: [0, 0] for b in ("exp", "rel", "abs")}
for s in seeds:
    for r in R[("MO-PEO", s)]["tr"]:
        for b in tot:
            tot[b][0] += r["n_" + b]; tot[b][1] += r["off_int_" + b]
allint = sum(v[1] for v in tot.values())
for b, nm in (("exp", "exploration (E + M + R)"), ("abs", "exploitation, absolute noise"), ("rel", "exploitation, relative noise")):
    P("| %s | %d | %d | %.2f | %.1f%% |" % (nm, tot[b][0], tot[b][1], 1000 * tot[b][1] / max(1, tot[b][0]), 100 * tot[b][1] / max(1, allint)))
# interior offspring: failed vs not failed, total counts per run
oi = np.array([sum(r["off_int"] for r in R[("MO-PEO", s)]["tr"]) for s in seeds]); ok_ = np.array([sum(r["off_int_kept"] for r in R[("MO-PEO", s)]["tr"]) for s in seeds])
P("\nInterior offspring over a whole run (median [min, max]): failed runs %s ; not failed runs %s." % (
    "%.0f [%d, %d]" % (np.median(oi[fail["MO-PEO"]]), oi[fail["MO-PEO"]].min(), oi[fail["MO-PEO"]].max()) if fail["MO-PEO"].any() else "-",
    "%.0f [%d, %d]" % (np.median(oi[~fail["MO-PEO"]]), oi[~fail["MO-PEO"]].min(), oi[~fail["MO-PEO"]].max())))
P("Of those, entering the archive: failed %s ; not failed %s." % ("%.0f" % np.median(ok_[fail["MO-PEO"]]) if fail["MO-PEO"].any() else "-", "%.0f" % np.median(ok_[~fail["MO-PEO"]])))

# ---- truncation: does it remove interior members preferentially?
ri = sum(sum(r["rem_int"] for r in R[("MO-PEO", s)]["tr"]) for s in seeds); pi = sum(sum(r["pre_int"] for r in R[("MO-PEO", s)]["tr"]) for s in seeds)
rc = sum(sum(r["cut"] for r in R[("MO-PEO", s)]["tr"]) for s in seeds); pp = sum(sum(r["pre"] for r in R[("MO-PEO", s)]["tr"]) for s in seeds)
P("\n## Does crowding truncation (D4) remove interior members preferentially? (MO-PEO, all iterations, all runs)\n")
P("Probability that a non-dominated member is removed by truncation: interior members %.3f (%d of %d), all members %.3f (%d of %d)." % (ri / max(1, pi), ri, pi, rc / pp, rc, pp))


# ---- discovery event: first iteration with an interior offspring
P("\n## The discovery event (MO-PEO): first iteration at which any offspring lands in the interior of the front\n")
firstit, br, xal = [], [], []
for sd in seeds:
    tr = R[("MO-PEO", sd)]["tr"]; i = next((k for k, r in enumerate(tr) if r["off_int"] > 0), None)
    firstit.append(np.nan if i is None else i + 1)
    br.append(None if i is None else {b: tr[i]["off_int_" + b] for b in ("exp", "abs", "rel")}); xal.append(np.nan if i is None else tr[i]["ag_x1_alive"])
firstit = np.array(firstit); f = fail["MO-PEO"]
P("Runs with at least one interior offspring: failed %d of %d, not failed %d of %d." % ((~np.isnan(firstit[f])).sum(), f.sum(), (~np.isnan(firstit[~f])).sum(), (~f).sum()))
v = firstit[~f]; P("First interior offspring in the runs that succeed: median iteration %.0f (%.2f of the budget), range %d-%d (%.2f-%.2f of the budget)." % (np.median(v), np.median(v) / T, v.min(), v.max(), v.min() / T, v.max() / T))
vf = firstit[f & ~np.isnan(firstit)]
if len(vf):
    P("Failed runs that had a first interior offspring: iterations %s (they never built an archive with more than %s interior members)." % (sorted(vf.astype(int)), int(max(max(col("MO-PEO", sd, "ar_int")) for sd, m in zip(seeds, f) if m))))
cnt = {b: 0 for b in ("exp", "abs", "rel")}
for b_ in br:
    if b_:
        for k_, x_ in b_.items(): cnt[k_] += int(x_ > 0)
P("At the discovery iteration, the branches that produced interior offspring (a run can count in several): exploration %d, exploitation with absolute noise %d, exploitation with relative noise %d (of %d runs with a discovery)." % (cnt["exp"], cnt["abs"], cnt["rel"], sum(1 for b_ in br if b_)))
P("\nLast iteration in which an interior offspring was produced, in the failed runs (0 = never): %s" % sorted(int(max([r["t"] for r in R[("MO-PEO", sd)]["tr"] if r["off_int"] > 0] or [0])) for sd, m in zip(seeds, f) if m))

# ---- time to success
P("\n## Time to success\n")
thr = 30   # at least 30 interior members in the archive
def first(a, s, key, thr):
    x = col(a, s, key); i = np.where(x >= thr)[0]; return (i[0] + 1) if len(i) else np.nan
P("First iteration with at least %d interior archive members (of %d iterations), by algorithm:\n" % (thr, T))
P("| algorithm | reached (of %d) | median iteration among those | fraction of budget | 10th / 90th percentile |\n|---|---|---|---|---|" % n)
tts = {}
for a in ALGS:
    x = np.array([first(a, s, "ar_int", thr) for s in seeds]); tts[a] = x; v = x[~np.isnan(x)]
    P("| %s | %d | %s | %s | %s |" % (a, len(v), "%.0f" % np.median(v) if len(v) else "-", "%.2f" % (np.median(v) / T) if len(v) else "-", "%.0f / %.0f" % tuple(np.percentile(v, [10, 90])) if len(v) else "-"))
# hazard: among MO-PEO runs, are the failed ones already distinguishable at the start?
P("\n## Is failure decided at initialisation? (MO-PEO)\n")
P("| initial-population feature | failed runs (median) | not failed (median) | Mann-Whitney p | AUC for failure |\n|---|---|---|---|---|")
for key, nm in (("x0_int", "fraction of initial agents in the interior"), ("x0_x1_alive", "fraction of initial agents with x1 > 0.9705"), ("x0_x2_alive", "fraction with x2 > 0.9705"),
                ("a0_int", "interior members of the initial archive"), ("a0_pole", "pole members of the initial archive"), ("a0_igd", "IGD of the initial archive")):
    v = np.array([R[("MO-PEO", s)]["init"][key] for s in seeds], float); f = fail["MO-PEO"]
    if f.sum() and (~f).sum():
        u = stats.mannwhitneyu(v[f], v[~f]); P("| %s | %.3g | %.3g | %.3f | %.2f |" % (nm, np.median(v[f]), np.median(v[~f]), u.pvalue, 1 - u.statistic / (f.sum() * (~f).sum())))
# does early progress predict?
for frac in (0.1, 0.3):
    v = at(np.array([col("MO-PEO", s, "ar_int") for s in seeds]), frac); f = fail["MO-PEO"]
    u = stats.mannwhitneyu(v[f], v[~f]); P("\nInterior archive members at %d%% of the budget: failed median %.0f, not failed median %.0f (Mann-Whitney p = %.3g)." % (frac * 100, np.median(v[f]), np.median(v[~f]), u.pvalue))
open("../results/diag_dtlz4.md", "w", encoding="utf-8").write("\n".join(L)); print("\n".join(L))

# ---- figure
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False,
                     "legend.fontsize": 7, "figure.dpi": 600, "savefig.dpi": 600, "mathtext.fontset": "dejavuserif", "axes.labelsize": 8, "axes.titlesize": 8})
ST = {"MO-PEO, failed": dict(color="black", ls="-", lw=1.4), "MO-PEO, not failed": dict(color="0.35", ls="--", lw=1.2), "GDE3": dict(color="0.6", ls="-.", lw=1.2),
      "NSGA-II, failed": dict(color="0.15", ls=":", lw=1.4), "NSGA-II, not failed": dict(color="0.5", ls=(0, (1, 1)), lw=1.0)}
x = np.arange(1, T + 1) / T
fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.4))
for (nm, a, m) in G:
    if m.sum() == 0:
        continue
    axs[0, 0].plot(x, np.nanmedian(grp(a, m, "ar_int"), 0), label="%s (n=%d)" % (nm, m.sum()), **ST[nm])
    axs[0, 1].plot(x, np.nanmedian(grp(a, m, "ag_int"), 0), **ST[nm])
    axs[1, 1].plot(x, np.nanmedian(grp(a, m, "ar_igd_int"), 0), **ST[nm])
axs[0, 0].set_ylabel("interior archive members"); axs[0, 1].set_ylabel("fraction of agents in the interior"); axs[1, 1].set_ylabel("IGD on the interior of the front")
axs[0, 1].set_yscale("symlog", linthresh=1e-3); axs[1, 1].set_yscale("log")
for a in ALGS:
    v = tts[a]; xs = np.sort(v[~np.isnan(v)]) / T
    axs[1, 0].step(np.r_[0, xs, 1], np.r_[0, np.arange(1, len(xs) + 1), len(xs)] / n, where="post", color={"MO-PEO": "black", "GDE3": "0.6", "NSGA-II": "0.3"}[a], ls={"MO-PEO": "-", "GDE3": "-.", "NSGA-II": ":"}[a], lw=1.3, label=a)
axs[1, 0].set_ylabel("runs with >= %d interior members" % thr); axs[1, 0].set_ylim(0, 1.02); axs[1, 0].legend(loc="upper left")
for ax, tg in zip(axs.ravel(), "abcd"):
    ax.set_xlabel("fraction of the evaluation budget"); ax.text(-0.17, 1.04, tg, transform=ax.transAxes, fontsize=10, fontweight="bold"); ax.tick_params(top=True, right=True)
h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
fig.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.90, hspace=0.32, wspace=0.30)
fig.savefig("../results/diag_dtlz4.png", dpi=600, bbox_inches="tight")
