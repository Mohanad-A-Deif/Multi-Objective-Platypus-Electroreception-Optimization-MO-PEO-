#!/usr/bin/env python3
"""
stats_six.py -- statistical layer over the six algorithms of the final comparison.

Algorithms : MO-PEO, NSGA-II, MOEA/D, MOPSO (default configuration), GDE3, MOGWO
Problems   : the 12 ZDT / DTLZ problems, 20 runs each (results/results_final.json)
Indicators : IGD (lower is better) and normalised hypervolume (higher is better)

Per problem and indicator
    mean, median, standard deviation and WORST run of every algorithm (means alone are distorted by failed runs);
    Wilcoxon rank-sum test of MO-PEO against each other algorithm (two-sided, alpha = 0.05), reported as
        +  MO-PEO significantly better,  -  significantly worse,  ~  no significant difference,
    with the raw p-value, the Holm-adjusted p-value over all 60 comparisons of that indicator (the conservative
    reading: 12 problems x 5 opponents) and the Vargha-Delaney effect size A12 = P(MO-PEO better than the other).
    The direction of the sign is taken from the rank statistic, not from the means; the mean-based direction
    used in the earlier three-algorithm tables is also computed and any disagreement is listed.

Across problems, per indicator
    Friedman test on the 12 x 6 matrix of per-problem means (tie-corrected), mean ranks, the Nemenyi critical
    difference at alpha = 0.05, and pairwise Wilcoxon signed-rank tests of MO-PEO against each other algorithm on
    the 12 per-problem means with Holm adjustment over the 5 comparisons. Everything is repeated without DTLZ3
    (every algorithm fails there, so the problem carries almost no information).

Nothing here claims significance beyond what the tests give; the report states the p-values.

Usage:  python stats_six.py [--results-dir ../results] [--out ../results/stats_six]
Output: <out>.json  (all numbers)   <out>.md  (readable report)
"""
import argparse, json, os
import numpy as np
from scipy import stats

ALGS = ["MO-PEO-final", "NSGA-II", "MOEA/D", "MOPSO", "GDE3", "MOGWO"]
SHORT = {"MO-PEO-final": "MO-PEO", "NSGA-II": "NSGA-II", "MOEA/D": "MOEA/D", "MOPSO": "MOPSO", "GDE3": "GDE3", "MOGWO": "MOGWO"}
PROBS = ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6", "DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]
METRICS = {"IGD": True, "HV": False}      # metric -> lower_is_better
BASE = "MO-PEO-final"
ALPHA = 0.05


def holm(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); adj = np.empty(n); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (n - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def a12(x, y, lower_better):
    """P(x better than y) + 0.5 P(tie), x = MO-PEO sample."""
    x, y = np.asarray(x), np.asarray(y)
    gt = (x[:, None] < y[None]).mean() if lower_better else (x[:, None] > y[None]).mean()
    eq = (x[:, None] == y[None]).mean()
    return float(gt + 0.5 * eq)


def ranksum(x, y):
    if np.ptp(np.r_[x, y]) == 0:
        return 0.0, 1.0
    r = stats.ranksums(x, y)
    return float(r.statistic), float(r.pvalue)


def cell_sign(z, p, lower_better):
    if not (p < ALPHA):
        return "~"
    better = (z < 0) if lower_better else (z > 0)       # statistic < 0: MO-PEO sample tends to be lower
    return "+" if better else "-"


def friedman_block(mat, lower_better):
    """mat: problems x algorithms matrix of means. Returns dict with Friedman, ranks, Nemenyi CD, signed-rank vs base."""
    n, k = mat.shape
    ranks = np.array([stats.rankdata(row if lower_better else -row) for row in mat])
    mean_ranks = ranks.mean(0)
    chi, p = stats.friedmanchisquare(*[mat[:, j] for j in range(k)])
    q = stats.studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2)
    cd = float(q * np.sqrt(k * (k + 1) / (6.0 * n)))
    b = ALGS.index(BASE)
    raw, names = [], []
    for j in range(k):
        if j == b:
            continue
        d = mat[:, b] - mat[:, j]
        pw = 1.0 if np.all(d == 0) else float(stats.wilcoxon(mat[:, b], mat[:, j]).pvalue)
        raw.append(pw); names.append(SHORT.get(ALGS[j], ALGS[j]))
    adj = holm(raw)
    return dict(n_problems=int(n), friedman_chi2=float(chi), friedman_p=float(p), mean_ranks={SHORT.get(a, a): float(m) for a, m in zip(ALGS, mean_ranks)},
                nemenyi_cd=cd, rank_diff_vs_MOPEO={SHORT.get(ALGS[j], ALGS[j]): float(mean_ranks[j] - mean_ranks[b]) for j in range(k) if j != b},
                signed_rank_p_raw=dict(zip(names, raw)), signed_rank_p_holm=dict(zip(names, map(float, adj))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="../results"); ap.add_argument("--out", default=None)
    ap.add_argument("--algs", nargs="+", default=None,
                    help="override the algorithm set (first = MO-PEO); used to re-derive the earlier three-algorithm statistics as a check of this script")
    a = ap.parse_args()
    global ALGS, BASE
    if a.algs:
        ALGS = a.algs; BASE = ALGS[0]
    out = a.out or os.path.join(a.results_dir, "stats_six")
    r = json.load(open(os.path.join(a.results_dir, "results_final.json")))
    res = {"per_problem": {}, "across": {}, "mean_direction_disagreements": []}
    md = ["# Statistical layer, six algorithms (20 runs per problem)", ""]

    for metric, lower in METRICS.items():
        rows = {}
        raws, keys = [], []
        for pn in PROBS:
            base = np.array(r["%s|%s" % (pn, BASE)][metric])
            row = {"desc": {}, "tests": {}}
            for alg in ALGS:
                v = np.array(r["%s|%s" % (pn, alg)][metric])
                row["desc"][SHORT.get(alg, alg)] = dict(mean=float(v.mean()), median=float(np.median(v)), std=float(v.std()),
                                                worst=float(v.max() if lower else v.min()))
            for alg in ALGS:
                if alg == BASE:
                    continue
                v = np.array(r["%s|%s" % (pn, alg)][metric])
                z, p = ranksum(base, v)
                mean_dir = "+" if ((base.mean() < v.mean()) if lower else (base.mean() > v.mean())) else "-"
                sg = cell_sign(z, p, lower)
                if sg in "+-" and sg != mean_dir:
                    res["mean_direction_disagreements"].append(dict(metric=metric, problem=pn, algorithm=SHORT.get(alg, alg), rank_sign=sg, mean_sign=mean_dir))
                row["tests"][SHORT.get(alg, alg)] = dict(p_raw=p, sign=sg, A12=a12(base, v, lower))
                raws.append(p); keys.append((pn, SHORT.get(alg, alg)))
            rows[pn] = row
        adj = holm(raws)
        for (pn, an), pa in zip(keys, adj):
            t = rows[pn]["tests"][an]; t["p_holm60"] = float(pa)
            t["sign_holm60"] = t["sign"] if (t["sign"] != "~" and pa < ALPHA) else "~"
        res["per_problem"][metric] = rows

        # ------------------------------------------------ across problems
        for label, plist in (("all12", PROBS), ("no_DTLZ3", [p for p in PROBS if p != "DTLZ3"])):
            mat = np.array([[rows[pn]["desc"][SHORT.get(al, al)]["mean"] for al in ALGS] for pn in plist])
            res["across"].setdefault(metric, {})[label] = friedman_block(mat, lower)

        # ------------------------------------------------ report
        md += ["## %s (%s is better)" % (metric, "lower" if lower else "higher"), ""]
        opp = [SHORT.get(x, x) for x in ALGS if x != BASE]
        md += ["Mean [median; worst] of every algorithm.", ""]
        md += ["| Problem | " + " | ".join(SHORT.get(x, x) for x in ALGS) + " |", "|---|" + "---|" * len(ALGS)]
        for pn in PROBS:
            best = min if lower else max
            bm = best(rows[pn]["desc"][SHORT.get(al, al)]["mean"] for al in ALGS)
            cells = []
            for al in ALGS:
                d = rows[pn]["desc"][SHORT.get(al, al)]
                c = "%.3g [%.3g; %.3g]" % (d["mean"], d["median"], d["worst"])
                cells.append("**%s**" % c if d["mean"] == bm else c)
            md.append("| %s | %s |" % (pn, " | ".join(cells)))
        md += ["", "Wilcoxon rank-sum, MO-PEO against each opponent: sign (raw p; Holm-60 p; A12). "
               "`+` MO-PEO significantly better, `-` worse, `~` no significant difference at 0.05 (raw). "
               "The Holm-60 column marks whether the result survives correction over all 60 comparisons of this indicator.", ""]
        md += ["| Problem | " + " | ".join(opp) + " |", "|---|" + "---|" * len(opp)]
        cnt = {o: [0, 0, 0] for o in opp}; cnt_h = {o: [0, 0, 0] for o in opp}
        for pn in PROBS:
            cells = []
            for o in opp:
                t = rows[pn]["tests"][o]
                cells.append("%s (%.2g; %.2g; %.2f)%s" % (t["sign"], t["p_raw"], t["p_holm60"], t["A12"], "" if t["sign"] == "~" or t["sign_holm60"] != "~" else " *"))
                cnt[o][{"+": 0, "-": 1, "~": 2}[t["sign"]]] += 1; cnt_h[o][{"+": 0, "-": 1, "~": 2}[t["sign_holm60"]]] += 1
            md.append("| %s | %s |" % (pn, " | ".join(cells)))
        md += ["", "`*` = significant at 0.05 uncorrected but not after Holm correction over 60 comparisons.", ""]
        md += ["Win/loss/tie counts for MO-PEO (+/-/~), uncorrected | Holm-60:", ""]
        md += ["| " + " | ".join(opp) + " |", "|" + "---|" * len(opp), "| " + " | ".join("%d/%d/%d \\| %d/%d/%d" % (*cnt[o], *cnt_h[o]) for o in opp) + " |", ""]
        for label in ("all12", "no_DTLZ3"):
            f = res["across"][metric][label]
            md += ["**Friedman across problems, %s (n = %d):** chi2 = %.3f, p = %.3g. Mean ranks: %s. Nemenyi critical difference at 0.05: %.2f."
                   % (label, f["n_problems"], f["friedman_chi2"], f["friedman_p"], ", ".join("%s %.2f" % kv for kv in f["mean_ranks"].items()), f["nemenyi_cd"]),
                   "Rank difference to MO-PEO (positive = MO-PEO ranks better; a pair is separated by Nemenyi only if the difference exceeds the CD): "
                   + ", ".join("%s %+.2f" % kv for kv in f["rank_diff_vs_MOPEO"].items()) + ".",
                   "Signed-rank test of MO-PEO on the per-problem means, raw p / Holm-5 p: "
                   + ", ".join("%s %.3g / %.3g" % (k, f["signed_rank_p_raw"][k], f["signed_rank_p_holm"][k]) for k in f["signed_rank_p_raw"]) + ".", ""]
    # ------------------------------------------------------------ supplementary: MOPSO tuned vs default
    if all("%s|MOPSO-tuned" % pn in r for pn in PROBS):
        md += ["## Supplementary: MOPSO default (official) against the coarse-grid tuned configuration", "",
               "| Problem | IGD default | IGD tuned | Wilcoxon p (IGD) | HV default | HV tuned |", "|---|---|---|---|---|---|"]
        for pn in PROBS:
            d, t = np.array(r["%s|MOPSO" % pn]["IGD"]), np.array(r["%s|MOPSO-tuned" % pn]["IGD"])
            hd, ht = np.mean(r["%s|MOPSO" % pn]["HV"]), np.mean(r["%s|MOPSO-tuned" % pn]["HV"])
            md.append("| %s | %.3g | %.3g | %.2g | %.3f | %.3f |" % (pn, d.mean(), t.mean(), ranksum(d, t)[1], hd, ht))
        md.append("")
    if res["mean_direction_disagreements"]:
        md += ["## Cases where the rank-based sign and the mean-based sign disagree", ""]
        md += ["- %(metric)s %(problem)s vs %(algorithm)s: rank sign %(rank_sign)s, mean sign %(mean_sign)s" % d for d in res["mean_direction_disagreements"]]
    json.dump(res, open(out + ".json", "w"), indent=1)
    open(out + ".md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
