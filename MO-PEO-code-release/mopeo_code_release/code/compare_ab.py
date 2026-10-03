#!/usr/bin/env python3
"""Option (a) [MO-PEO with the deferred-refinement schedule, tau = 0.1] against option (b) [no schedule], under the rule written in results/option_b_protocol.md.
Needs results/stats_seven_ab.json, results/stats_six.json (a), results/stats_six_b.json (b) (stats_six.py), results/tau_calib.pkl.
Writes results/option_b.md and results/option_b.json."""
import json, pickle
import numpy as np
from scipy import stats

PROBS = ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6", "DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]
A, B = "MO-PEO-final", "MO-PEO-noSchedule"
COMP = ["NSGA-II", "MOEA/D", "MOPSO", "GDE3", "MOGWO"]
ALL7 = [A, B] + COMP
rng = np.random.default_rng(12345)
r = json.load(open("../results/results_final.json"))
S7 = json.load(open("../results/stats_seven_ab.json")); SA = json.load(open("../results/stats_six.json")); SB = json.load(open("../results/stats_six_b.json"))
calib = pickle.load(open("../results/tau_calib.pkl", "rb")); proto = json.load(open("../results/tau_calib_protocol.json"))
L = ["# Option (a) versus option (b)\n", "(a) = MO-PEO with the schedule, tau = 0.1 (`MO-PEO-final`). (b) = localised perturbation only, no schedule (`MO-PEO-noSchedule`). 20 runs per problem, seeds 1000-1019. Rule: `results/option_b_protocol.md`.\n"]
out = {}


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (n - k) * p[i]); adj[i] = min(1.0, run)
    return adj


def ranksum(x, y):
    if np.ptp(np.r_[x, y]) == 0:
        return 1.0
    return float(stats.ranksums(x, y).pvalue)


def signrank(x, y):
    d = np.asarray(x) - np.asarray(y)
    return 1.0 if np.all(d == 0) else float(stats.wilcoxon(x, y).pvalue)


def ratio_ci(a, b, nboot=10000):
    """median(b)/median(a) with a paired bootstrap over seeds."""
    a, b = np.asarray(a), np.asarray(b); n = len(a); idx = rng.integers(0, n, (nboot, n))
    ra = np.median(b[idx], 1) / np.median(a[idx], 1)
    return float(np.median(b) / np.median(a)), [float(np.percentile(ra, 2.5)), float(np.percentile(ra, 97.5))]


# ------------------------------------------------------------------ 1 ranking
L.append("## 1. Ranking across the twelve problems (seven algorithms; Friedman on per-problem means)\n")
for m in ("IGD", "HV"):
    a = S7["across"][m]["all12"]; mr = a["mean_ranks"]
    L.append("**%s** (Friedman p = %.2g, Nemenyi CD = %.2f): " % (m, a["friedman_p"], a["nemenyi_cd"]) + ", ".join("%s %.2f" % (k if k != "MO-PEO" else "MO-PEO (a)", v) for k, v in mr.items())
             + ". Rank difference (b) - (a) = %.2f; signed-rank (a) vs (b) on the twelve means: raw p = %.3f.\n" % (mr["MO-PEO-noSchedule"] - mr["MO-PEO"], a["signed_rank_p_raw"]["MO-PEO-noSchedule"]))
    out["rank_" + m] = dict(mean_ranks=mr, cd=a["nemenyi_cd"], friedman_p=a["friedman_p"], signed_rank_p=a["signed_rank_p_raw"]["MO-PEO-noSchedule"])
    for lab, S in (("(a)", SA), ("(b)", SB)):
        a6 = S["across"][m]["all12"]; kk = "MO-PEO" if lab == "(a)" else "MO-PEO-noSchedule"
        L.append("Six algorithms with %s in place of MO-PEO: mean ranks " % lab + ", ".join("%s %.2f" % (k, v) for k, v in a6["mean_ranks"].items()) + " (CD %.2f)." % a6["nemenyi_cd"] + "\n")

# ------------------------------------------------------------------ 2 per problem
L.append("## 2. Problem by problem, (a) against (b)\n")
L.append("Ratio = median IGD of (b) / median IGD of (a), with a 95% bootstrap interval (10,000 resamples of the seeds, paired). p: rank-sum on the 20 runs, raw and Holm over the 12 problems; s = paired signed-rank (same seeds).\n")
L.append("| problem | (a) mean [median; worst] | (b) mean [median; worst] | ratio (b)/(a) | p raw | p Holm-12 | s | better |\n|---|---|---|---|---|---|---|---|")
rows = {}; ps = []
for p in PROBS:
    a = np.array(r["%s|%s" % (p, A)]["IGD"]); b = np.array(r["%s|%s" % (p, B)]["IGD"])
    rt, ci = ratio_ci(a, b); pv = ranksum(a, b); sr = signrank(a, b); ps.append(pv)
    rows[p] = dict(a=[a.mean(), np.median(a), a.max()], b=[b.mean(), np.median(b), b.max()], ratio=rt, ci=ci, p=pv, s=sr)
ad = holm(ps)
for (p, d), h in zip(rows.items(), ad):
    d["p_holm"] = float(h)
    better = "-" if d["p_holm"] >= 0.05 else ("(a)" if d["ratio"] > 1 else "(b)")
    d["better"] = better
    L.append("| %s | %.3g [%.3g; %.3g] | %.3g [%.3g; %.3g] | %.2f [%.2f, %.2f] | %.3g | %.3g | %.3g | %s |" % (p, *d["a"], *d["b"], d["ratio"], *d["ci"], d["p"], d["p_holm"], d["s"], better))
out["per_problem_IGD"] = rows
L.append("")
L.append("HV (mean of (a), mean of (b), rank-sum p raw / Holm-12):\n")
ps = []; hv = {}
for p in PROBS:
    a = np.array(r["%s|%s" % (p, A)]["HV"]); b = np.array(r["%s|%s" % (p, B)]["HV"]); ps.append(ranksum(a, b)); hv[p] = (a.mean(), b.mean())
ah = holm(ps)
L.append("| problem | HV (a) | HV (b) | p raw | p Holm-12 |\n|---|---|---|---|---|")
for (p, (x, y)), pr_, h in zip(hv.items(), ps, ah):
    L.append("| %s | %.3f | %.3f | %.3g | %.3g |" % (p, x, y, pr_, h))
out["per_problem_HV_holm"] = {p: float(h) for p, h in zip(hv, ah)}

# ------------------------------------------------------------------ 3 wins and losses
L.append("\n## 3. Significant wins and losses against the five competitors (rank-sum; Holm over the 60 comparisons of each six-algorithm analysis)\n")
L.append("| indicator | option | + raw | - raw | ~ raw | + Holm | - Holm | ~ Holm |\n|---|---|---|---|---|---|---|---|")
wl = {}
for m in ("IGD", "HV"):
    for lab, S in (("(a)", SA), ("(b)", SB)):
        cnt = {"raw": {"+": 0, "-": 0, "~": 0}, "holm": {"+": 0, "-": 0, "~": 0}}
        for p in PROBS:
            for c in COMP:
                t = S["per_problem"][m][p]["tests"][c]; cnt["raw"][t["sign"]] += 1; cnt["holm"][t["sign_holm60"]] += 1
        wl[m + lab] = cnt
        L.append("| %s | %s | %d | %d | %d | %d | %d | %d |" % (m, lab, *[cnt["raw"][k] for k in "+-~"], *[cnt["holm"][k] for k in "+-~"]))
out["wins_losses"] = wl
L.append("\nWhere the two options differ against a given competitor (IGD, Holm-60 sign; (a) -> (b)):\n")
for p in PROBS:
    ch = []
    for c in COMP:
        sa = SA["per_problem"]["IGD"][p]["tests"][c]["sign_holm60"]; sb = SB["per_problem"]["IGD"][p]["tests"][c]["sign_holm60"]
        if sa != sb:
            ch.append("%s %s -> %s" % (c, sa, sb))
    if ch:
        L.append("- %s: %s" % (p, "; ".join(ch)))

# ------------------------------------------------------------------ 4 failures
L.append("\n## 4. Failed runs\n")
L.append("A run fails if its IGD exceeds 5 x the smallest median IGD reached by any of the seven algorithms on that problem (DTLZ3 excluded: every algorithm fails). DTLZ4 also with IGD > 0.2.\n")
L.append("| problem | threshold | " + " | ".join(["(a)", "(b)"] + COMP) + " |\n|---|---|" + "---|" * 7)
fa = fb = 0; fail = {}
for p in PROBS:
    if p == "DTLZ3":
        continue
    thr = 5 * min(np.median(r["%s|%s" % (p, x)]["IGD"]) for x in ALL7)
    cs = [int((np.array(r["%s|%s" % (p, x)]["IGD"]) > thr).sum()) for x in ALL7]
    fail[p] = dict(threshold=float(thr), counts=dict(zip(ALL7, cs))); fa += cs[0]; fb += cs[1]
    L.append("| %s | %.3g | " % (p, thr) + " | ".join(str(c) for c in cs) + " |")
L.append("\nTotal over 11 problems (220 runs each): (a) %d, (b) %d." % (fa, fb))
d4 = {x: int((np.array(r["DTLZ4|%s" % x]["IGD"]) > 0.2).sum()) for x in ALL7}
L.append("DTLZ4, IGD > 0.2: " + ", ".join("%s %d" % (k, v) for k, v in d4.items()) + ".")
out["failures"] = fail; out["failures_total"] = dict(a=fa, b=fb); out["dtlz4_gt_0.2"] = d4

# ------------------------------------------------------------------ 5 replication on calibration seeds
L.append("\n## 5. Replication on the 40 calibration seeds (6000-6039): tau = 0.1 against tau = 1.0 (= option (b))\n")
L.append("tau = 0.1 was selected on the tuning half of these seeds, so the tuning half favours (a); the held-out half is the fairer reading. Ratio = median IGD(tau = 1) / median IGD(tau = 0.1), paired bootstrap interval.\n")
L.append("| problem | half | median (a) | median (b) | ratio (b)/(a) | p raw | p Holm-12 |\n|---|---|---|---|---|---|---|")
ps = []; rr = {}
for p in PROBS:
    a = np.array([calib[(p, 0.1, s)] for s in range(6000, 6040)]); b = np.array([calib[(p, 1.0, s)] for s in range(6000, 6040)])
    rt, ci = ratio_ci(a, b); ps.append(ranksum(a, b)); rr[p] = (np.median(a), np.median(b), rt, ci)
ah = holm(ps)
for (p, v), pr_, h in zip(rr.items(), ps, ah):
    L.append("| %s | %s | %.3g | %.3g | %.2f [%.2f, %.2f] | %.3g | %.3g |" % (p, "tune" if p in proto["tune"] else "held-out", v[0], v[1], v[2], *v[3], pr_, h))
out["calibration_replication"] = {p: dict(median_a=float(v[0]), median_b=float(v[1]), ratio=float(v[2]), ci=v[3]) for p, v in rr.items()}

# ------------------------------------------------------------------ 6 the rule
L.append("\n## 6. The pre-declared rule\n")
res = {}
for m in ("IGD", "HV"):
    a = S7["across"][m]["all12"]
    c1 = abs(a["mean_ranks"]["MO-PEO-noSchedule"] - a["mean_ranks"]["MO-PEO"]) < a["nemenyi_cd"]
    c2 = a["signed_rank_p_raw"]["MO-PEO-noSchedule"] >= 0.05
    res[m] = (c1, c2)
    L.append("- %s: (1) rank difference %.2f < CD %.2f: %s; (2) signed-rank p = %.3f >= 0.05: %s." % (m, a["mean_ranks"]["MO-PEO-noSchedule"] - a["mean_ranks"]["MO-PEO"], a["nemenyi_cd"], c1, a["signed_rank_p_raw"]["MO-PEO-noSchedule"], c2))
sig = [p for p, d in rows.items() if d["p_holm"] < 0.05]
L.append("- (3) problems with a significant (a)-(b) difference on IGD after Holm-12: %s." % (", ".join("%s (%s better)" % (p, rows[p]["better"]) for p in sig) or "none"))
c3 = len(sig) == 0
verdict = "INDISTINGUISHABLE" if all(all(v) for v in res.values()) and c3 else ("INDISTINGUISHABLE ON THE RANKING, DIFFERENT ON NAMED PROBLEMS" if all(all(v) for v in res.values()) else "DISTINGUISHABLE")
L.append("\n**Verdict under the rule: %s.**" % verdict)
out["verdict"] = verdict; out["significant_problems"] = sig
open("../results/option_b.md", "w").write("\n".join(L) + "\n"); json.dump(out, open("../results/option_b.json", "w"), indent=1, default=float)
print("\n".join(L))
