#!/usr/bin/env python3
"""
probe_mopso.py -- give MOPSO its best shot on the two problems where it is weakest (ZDT4, ZDT6).

Purpose: evidence, not tuning of the mutation operator. Polynomial mutation (eta_m = 20, p = 1/D) is FIXED
because it is what NSGA-II and MOEA/D use in this study. Only the swarm coefficients and the grid
resolution are probed, on NON-PROTOCOL seeds (5000 .. 5004), so the protocol seeds (1000 + run) are untouched.

Grid:   w in {0.2, 0.4, 0.6, 0.729}   c1 = c2 in {1.0, 1.5, 2.0}   grid divisions (M = 2) in {6, 10}
        archive size = N (fixed).  Reference row: the untuned default (w = 0.4, c = 1.0, 30 divisions).
Score (declared before running): geometric mean over {ZDT4, ZDT6} of  mean IGD / mean IGD of NSGA-II
        (NSGA-II means taken from results_final.json).  Lower is better.  The configuration with the
        lowest score is reported as "MOPSO (tuned)".

Usage:  python probe_mopso.py [--grid coarse|extended] [--results-dir ../results]
        -> results/probe_mopso.json (coarse) or results/probe_mopso_extended.json (robustness check, see below)

The coarse-grid optimum lay on a corner of the grid (lowest w, highest c). The "extended" grid
(w in {0.05, 0.1, 0.2}, c in {2.0, 2.5}) only checks whether that changes the conclusion; the official
"tuned" configuration remains the best of the coarse grid unless stated otherwise.
"""
import argparse, itertools, json, os, sys
import numpy as np
from multiprocessing import Pool

from metrics import igd
from mopso import mopso
from validate_baselines import problem

SEEDS = [5000 + i for i in range(5)]
PROBS = ["ZDT4", "ZDT6"]


def evaluate(cfg):
    out = {}
    for pn in PROBS:
        pr, P, N, FE = problem(pn)
        v, sz = [], []
        for s in SEEDS:
            A = mopso(pr, N, FE, np.random.default_rng(s), **cfg)
            v.append(float(igd(A, P))); sz.append(len(A))
        out[pn] = dict(igd_mean=float(np.mean(v)), igd_median=float(np.median(v)), igd_all=v, size_mean=float(np.mean(sz)))
    return cfg, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results-dir", default="../results"); ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--grid", choices=["coarse", "extended"], default="coarse")
    a = ap.parse_args()
    ref = json.load(open(os.path.join(a.results_dir, "results_final.json")))
    n2 = {p: float(np.mean(ref["%s|NSGA-II" % p]["IGD"])) for p in PROBS}
    if a.grid == "coarse":
        cfgs = [dict(w=w, c1=c, c2=c, ndiv2=nd) for w, c, nd in itertools.product([0.2, 0.4, 0.6, 0.729], [1.0, 1.5, 2.0], [6, 10])]
    else:   # robustness check: the coarse optimum sat on the corner (lowest w, highest c) of the grid
        cfgs = [dict(w=w, c1=c, c2=c, ndiv2=nd) for w, c, nd in itertools.product([0.05, 0.1, 0.2], [2.0, 2.5], [6, 10])]
    cfgs.append(dict(w=0.4, c1=1.0, c2=1.0, ndiv2=30))          # untuned default, for reference
    with Pool(a.procs) as p:
        res = p.map(evaluate, cfgs)
    rows = []
    for cfg, o in res:
        score = float(np.exp(np.mean([np.log(o[p]["igd_mean"] / n2[p]) for p in PROBS])))
        rows.append(dict(cfg=cfg, score=score, **{p: o[p] for p in PROBS}))
    rows.sort(key=lambda r: r["score"])
    json.dump(dict(seeds=SEEDS, nsga2=n2, rows=rows), open(os.path.join(a.results_dir, "probe_mopso.json" if a.grid == "coarse" else "probe_mopso_extended.json"), "w"), indent=1)
    print("score = geometric mean of (MOPSO / NSGA-II) IGD over ZDT4, ZDT6   [NSGA-II: %s]" % {k: "%.3g" % v for k, v in n2.items()})
    print("%-6s %-5s %-6s | %10s %10s | %10s %10s | %8s" % ("w", "c", "ndiv", "ZDT4 IGD", "arch", "ZDT6 IGD", "arch", "score"))
    for r in rows:
        c = r["cfg"]
        print("%-6g %-5g %-6d | %10.3e %10.1f | %10.3e %10.1f | %8.1f%s" % (c["w"], c["c1"], c["ndiv2"], r["ZDT4"]["igd_mean"], r["ZDT4"]["size_mean"],
              r["ZDT6"]["igd_mean"], r["ZDT6"]["size_mean"], r["score"], "   <- untuned default" if (c["w"], c["c1"], c["ndiv2"]) == (0.4, 1.0, 30) else ""))


if __name__ == "__main__":
    main()
