#!/usr/bin/env python3
"""
mktables_supp.py -- supplementary LaTeX tables.

Writes tables_supp.tex with
  S1  validation of the NSGA-II baseline against Deb et al. (2002)                 (values measured earlier, see HANDOVER.md)
  S2  the two alternative diversification designs (S1 / S2 variants); the Friedman p-values are COMPUTED here
  S3  robustness view: median / worst run of every algorithm of the main comparison
  S4  MOPSO: default (reported in the main comparison) against the coarse-grid tuned configuration
  and \\input lines for the internal-validation tables of MOPSO, GDE3 and MOGWO (paper/table_validation_*.tex,
  written by validate_baselines.py).
"""
import json, os, sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats_six as S
from mktables import sci, pstr, DISPLAY, MAIN, analyse


def _p(f):
    return os.path.join("results", f) if os.path.exists(os.path.join("results", f)) else f


r = json.load(open(_p("results_final.json")))
names = S.PROBS

# ---- S1: NSGA-II validation (measured in this work, six runs; published values from Deb et al. 2002)
deb = {"ZDT1": (0.0415, 0.033), "ZDT2": (0.0405, 0.072), "ZDT3": (0.0421, 0.114), "ZDT4": (0.3446, 0.513), "ZDT6": (0.0719, 0.296)}
rows = ["%s & %.4f & %.3f \\\\" % (k, v[0], v[1]) for k, v in deb.items()]
T_S1 = """\\begin{table}[htbp]
\\centering
\\caption{Validation of the NSGA-II baseline. Convergence metric $\\gamma$ (mean distance of the obtained non-dominated set to 500 uniformly sampled Pareto-optimal points) at 25{,}000 evaluations, six runs, compared with the values reported by Deb et al. (2002) for real-coded NSGA-II under the same protocol. Lower is better.}
\\label{tab:validation}
\\begin{tabular}{lcc}
\\toprule
Problem & This implementation & Deb et al. (2002) \\\\
\\midrule
%s
\\bottomrule
\\end{tabular}
\\end{table}""" % "\n".join(rows)

# ---- S2: variants, Friedman computed
vs = ["MO-PEO-final", "S1-splitDensity", "S2-spatialPrel"]
rows_i, fr_i = None, None
rows, fr_i, _ = analyse(r, vs, "IGD")
_, fr_h, _ = analyse(r, vs, "HV")
L = []
for pn in names:
    cells = []
    means = [rows[pn]["desc"][a]["mean"] for a in vs]
    b = int(np.argmin(means))
    for i, a in enumerate(vs):
        d = rows[pn]["desc"][a]
        c = "%.3e (%.1e)" % (d["mean"], d["std"])
        if i == b:
            c = "\\textbf{%s}" % c
        if a != vs[0]:
            c += {"+": " $+$", "-": " $-$", "~": " $\\approx$"}[rows[pn]["sign"][a]]
        cells.append(c)
    L.append("%s & %s \\\\" % (pn, " & ".join(cells)))
mr = fr_i["mean_ranks"]
L.append("\\midrule")
L.append("Mean rank & " + " & ".join("%.2f" % mr[S.SHORT.get(a, a)] for a in vs) + " \\\\")
T_S2 = """\\begin{table}[htbp]
\\centering
\\caption{Mean IGD (standard deviation) over 20 runs for the two alternative diversification designs. S1 assigns dense perturbation to half of the agents during refinement; S2 replaces the temporal schedule of $p_{\\mathrm{rel}}$ by a fixed per-agent value. Symbols: Wilcoxon rank-sum test against MO-PEO, $\\alpha=0.05$ after Holm correction over the 24 comparisons. Friedman test over the three variants (12 per-problem means): IGD $p=%s$, hypervolume $p=%s$; the three variants are not separable.}
\\label{tab:variants}
\\begin{tabular}{lccc}
\\toprule
Problem & MO-PEO & S1 (split density) & S2 (spatial $p_{\\mathrm{rel}}$) \\\\
\\midrule
%s
\\bottomrule
\\end{tabular}
\\end{table}""" % (pstr(fr_i["friedman_p"]), pstr(fr_h["friedman_p"]), "\n".join(L))

# ---- S3: median / worst for the main algorithms
S.ALGS, S.BASE = list(MAIN), MAIN[0]
L = []
for pn in names:
    c = []
    for a in MAIN:
        v = np.array(r["%s|%s" % (pn, a)]["IGD"])
        c.append("\\makecell{%s\\\\/ %s}" % (sci(np.median(v), 2), sci(v.max(), 2)))
    L.append("%s & %s \\\\" % (pn, " & ".join(c)))
T_S3 = """\\begin{table}[htbp]
\\centering
\\caption{Robustness view of the IGD results: median (upper line) and worst value over 20 runs (lower line). A large gap between median and worst identifies problems on which occasional runs fail (notably DTLZ6 for MO-PEO, ZDT2 for MOGWO).}
\\label{tab:robust}
\\resizebox{\\textwidth}{!}{\\begin{tabular}{l%s}
\\toprule
Problem & %s \\\\
\\midrule
%s
\\bottomrule
\\end{tabular}}
\\end{table}""" % ("c" * len(MAIN), " & ".join(DISPLAY[a] for a in MAIN), "\n".join(L))

# ---- S4: MOPSO default vs tuned
L = []
for pn in names:
    d, t = np.array(r["%s|MOPSO" % pn]["IGD"]), np.array(r["%s|MOPSO-tuned" % pn]["IGD"])
    hd, ht = np.mean(r["%s|MOPSO" % pn]["HV"]), np.mean(r["%s|MOPSO-tuned" % pn]["HV"])
    z, p = S.ranksum(d, t)
    L.append("%s & %s & %s & %s & %.3f & %.3f \\\\" % (pn, sci(d.mean(), 2), sci(t.mean(), 2), ("$%s$" % pstr(p)) if p < 0.05 else "%.2f" % p, hd, ht))
T_S4 = """\\begin{table}[htbp]
\\centering
\\caption{MOPSO with its default configuration ($w=0.4$, $c_1=c_2=1$, 30 grid divisions for $M=2$), which is the configuration reported in the main comparison, and with the configuration selected by a coarse grid search ($w=0.2$, $c_1=c_2=2$, 6 divisions) on ZDT4 and ZDT6 over five non-protocol seeds. The search improved MOPSO on some problems and degraded it on others, so the default configuration is reported in the main comparison. Mean IGD and hypervolume over 20 runs; the $p$-value is that of a two-sided Wilcoxon rank-sum test of the IGD samples (uncorrected).}
\\label{tab:mopso_tuned}
\\begin{tabular}{lccccc}
\\toprule
Problem & IGD default & IGD tuned & $p$ & HV default & HV tuned \\\\
\\midrule
%s
\\bottomrule
\\end{tabular}
\\end{table}""" % "\n".join(L)

inputs = "\n".join("\\input{table_validation_%s.tex}" % k for k in ("mopso", "gde3", "mogwo"))
inputs += "\n\\input{table_tau_fresh.tex}"   # written by analyse_tau.py

open("tables_supp.tex", "w", encoding="utf-8").write("\n\n".join([T_S1, T_S2, T_S3, T_S4, inputs]))
print("written; variants Friedman p: IGD %.3g HV %.3g" % (fr_i["friedman_p"], fr_h["friedman_p"]))
