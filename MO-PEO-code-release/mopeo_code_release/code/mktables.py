#!/usr/bin/env python3
"""
mktables.py -- LaTeX tables of the main study, for ANY number of algorithms.

    python mktables.py                                   # the six algorithms of the final comparison
    python mktables.py --algs MO-PEO-final NSGA-II MOEA/D   # any list of keys of results_final.json; the first is
                                                          # the reference algorithm the Wilcoxon tests are made against

Reads   results/results_final.json, results/tau.json, results/timing.json
Writes  tables.tex   (IGD, HV, ablation, tau sensitivity, timing)

Statistics (all computed here, nothing is typed in by hand; the tests are those of stats_six.py, which this file imports):
  * symbols in the IGD / HV tables: Wilcoxon rank-sum test of the reference algorithm against each other
    algorithm, two-sided, alpha = 0.05 AFTER Holm correction over all problems x opponents of that table
    (60 comparisons for the six-algorithm tables); the direction is taken from the rank statistic;
  * Friedman test over the per-problem means (tie-corrected, Iman-Davenport not applied), with the
    Nemenyi critical difference, reported in the caption;
  * the mean-rank row is the mean rank of the Friedman test.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats_six as S

NAMES = S.PROBS
DISPLAY = {"MO-PEO-final": "MO-PEO", "A1-sparse": "A1 (sparse only)", "B2-delayOnly": "B2 (schedule only)",
           "A0-noSparse-noUnion": "A0 (neither)", "MOEA/D": "MOEA/D", "NSGA-II": "NSGA-II", "MOPSO": "MOPSO",
           "GDE3": "GDE3", "MOGWO": "MOGWO", "MOPSO-tuned": "MOPSO (tuned)"}
ABL = ["MO-PEO-final", "A1-sparse", "B2-delayOnly", "A0-noSparse-noUnion"]
MAIN = ["MO-PEO-final", "NSGA-II", "MOEA/D", "MOPSO", "GDE3", "MOGWO"]


def _p(f):
    return os.path.join("results", f) if os.path.exists(os.path.join("results", f)) else f


def sci(x, d=2):
    """1.27e-2 style (no zero padding); plain 0 for exact zeros."""
    if x == 0:
        return "0"
    m, e = ("%.*e" % (d, x)).split("e")
    return "%se%d" % (m, int(e))


def pstr(p):
    if p >= 0.01:
        return "%.3f" % p
    m, e = ("%.2e" % p).split("e")
    return "%s\\times10^{%d}" % (m, int(e))


def analyse(r, algs, metric):
    """Everything the table needs for one metric. algs[0] is the reference algorithm."""
    S.ALGS, S.BASE = list(algs), algs[0]
    lower = S.METRICS[metric]
    rows, raws, keys = {}, [], []
    for pn in NAMES:
        base = np.array(r["%s|%s" % (pn, algs[0])][metric])
        row = dict(desc={}, sign={})
        for a in algs:
            v = np.array(r["%s|%s" % (pn, a)][metric])
            row["desc"][a] = dict(mean=v.mean(), std=v.std(), median=np.median(v), worst=v.max() if lower else v.min())
        for a in algs[1:]:
            v = np.array(r["%s|%s" % (pn, a)][metric])
            z, p = S.ranksum(base, v)
            row["sign"][a] = S.cell_sign(z, p, lower); raws.append(p); keys.append((pn, a))
        rows[pn] = row
    adj = S.holm(raws)
    for (pn, a), pa in zip(keys, adj):
        if rows[pn]["sign"][a] != "~" and pa >= S.ALPHA:
            rows[pn]["sign"][a] = "~"
    mat = np.array([[rows[pn]["desc"][a]["mean"] for a in algs] for pn in NAMES])
    fr = S.friedman_block(mat, lower)
    return rows, fr, lower


SYM = {"+": " $+$", "-": " $-$", "~": " $\\approx$"}


def mean_std_table(r, algs, metric, caption, label, header=None, resize=True):
    rows, fr, lower = analyse(r, algs, metric)
    header = header or [DISPLAY.get(a, a) for a in algs]
    L, cnt = [], {a: [0, 0, 0] for a in algs[1:]}
    disagree = []
    for pn in NAMES:
        means = [rows[pn]["desc"][a]["mean"] for a in algs]
        best = int(np.argmin(means) if lower else np.argmax(means))
        if sum(m == means[best] for m in means) > 1:      # a shared best value (e.g. HV = 0 for every algorithm) is not marked
            best = -1
        cells = []
        for i, a in enumerate(algs):
            d = rows[pn]["desc"][a]
            top = sci(d["mean"], 2)
            top = "\\textbf{%s}" % top if i == best else top
            sg = ""
            if a != algs[0]:
                s = rows[pn]["sign"][a]; sg = SYM[s]; cnt[a][{"+": 0, "-": 1, "~": 2}[s]] += 1
                m0, m1 = rows[pn]["desc"][algs[0]]["mean"], d["mean"]
                if s != "~" and s != ("+" if ((m0 < m1) if lower else (m0 > m1)) else "-"):
                    disagree.append("%s against %s" % (pn, DISPLAY.get(a, a)))
            cells.append("\\makecell{%s%s\\\\(%s)}" % (top, sg, sci(d["std"], 1)))
        L.append("%s & %s \\\\" % (pn, " & ".join(cells)))
    L.append("\\midrule")
    mr = fr["mean_ranks"]
    L.append("Mean rank & " + " & ".join("%.2f" % mr[S.SHORT.get(a, a)] for a in algs) + " \\\\")
    L.append("$+/-/\\approx$ & --- & " + " & ".join("%d/%d/%d" % tuple(cnt[a]) for a in algs[1:]) + " \\\\")
    k = len(algs)
    note = (" The symbol is that of the rank-sum test, not of the comparison of the two means; the two point in opposite directions for %s "
            "(a few failed runs move a mean without moving the bulk of the distribution)." % ", ".join(disagree)) if disagree else ""
    cap = (caption + note + " Symbols: Wilcoxon rank-sum test of MO-PEO against each algorithm, $\\alpha=0.05$ after Holm correction over the %d comparisons of this table; "
           "$+$ MO-PEO significantly better, $-$ significantly worse, $\\approx$ not significant. "
           "Friedman test over the %d per-problem means: $\\chi^2_{%d}=%.2f$, $p=%s$; Nemenyi critical difference %.2f (a pair of mean ranks closer than this is not separated)."
           % ((len(algs) - 1) * len(NAMES), fr["n_problems"], k - 1, fr["friedman_chi2"], pstr(fr["friedman_p"]), fr["nemenyi_cd"]))
    body = ("\\begin{tabular}{l%s}\n\\toprule\nProblem & %s \\\\\n\\midrule\n%s\n\\bottomrule\n\\end{tabular}"
            % ("c" * k, " & ".join(header), "\n".join(L)))
    if resize:
        body = "\\resizebox{\\textwidth}{!}{%s}" % body
    return "\\begin{table}[htbp]\n\\centering\n\\caption{%s}\n\\label{%s}\n%s\n\\end{table}" % (cap, label, body), fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algs", nargs="+", default=MAIN)
    ap.add_argument("--out", default="tables.tex")
    a = ap.parse_args()
    r = json.load(open(_p("results_final.json")))
    algs = a.algs

    T1, fr_igd = mean_std_table(
        r, algs, "IGD", "Mean IGD (standard deviation in parentheses) over 20 independent runs; lower is better, best mean per problem in bold.", "tab:igd")
    T2, fr_hv = mean_std_table(
        r, algs, "HV", "Mean hypervolume (standard deviation in parentheses) over 20 independent runs, computed after normalisation with reference point $\\mathbf{1.1}$; higher is better, best mean per problem in bold.", "tab:hv")
    T3, fr_abl = mean_std_table(
        r, ABL, "IGD", "Ablation study, mean IGD (standard deviation) over 20 runs. A1 retains only the localised perturbation, B2 only the deferred-refinement schedule, A0 neither.",
        "tab:ablation", resize=False)

    # sensitivity to tau: the calibration table on fresh seeds (analyse_tau_calib.py writes paper/table_tau_calib.tex)
    T4 = "\\input{table_tau_calib.tex}"

    tm = json.load(open(_p("timing.json")))
    short = [S.SHORT.get(x, x) for x in algs]
    rows = ["%s & %s \\\\" % (n, " & ".join("%.2f (%.2f)" % tuple(tm["%s|%s" % (n, s)]) for s in short)) for n in ["ZDT1", "DTLZ2"]]
    T5 = ("\\begin{table}[htbp]\n\\centering\n\\caption{Mean wall-clock time in seconds (standard deviation) over 5 runs at the budgets of Section~\\ref{sec:setup}, "
          "single-threaded NumPy on one CPU core of the same machine for all algorithms. Implementation effects dominate these figures; they are indicative only.}\n"
          "\\label{tab:time}\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{l%s}\n\\toprule\nProblem & %s \\\\\n\\midrule\n%s\n\\bottomrule\n\\end{tabular}}\n\\end{table}"
          % ("c" * len(short), " & ".join(DISPLAY.get(x, x) for x in algs), "\n".join(rows)))

    open(a.out, "w", encoding="utf-8").write("\n\n".join([T1, T2, T3, T4, T5]))
    print("wrote", a.out)
    print("Friedman IGD: chi2 %.3f p %.3g | HV: chi2 %.3f p %.3g | ablation IGD: chi2 %.3f p %.3g"
          % (fr_igd["friedman_chi2"], fr_igd["friedman_p"], fr_hv["friedman_chi2"], fr_hv["friedman_p"], fr_abl["friedman_chi2"], fr_abl["friedman_p"]))


if __name__ == "__main__":
    main()
