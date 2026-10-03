#!/usr/bin/env python3
"""DTLZ4: dynamic ablations of ONE term of the MO-PEO update at a time, as a probe of the mechanism (NOT candidate fixes; they are not evaluated on any other problem).
For each configuration: 40 runs (seeds 7000..7039, tau = 0.1), failure = final IGD > 0.2, and the number of interior members (x1, x2 in the window) in the final archive.
Configurations: reported; no leader attraction (gamma = lambda = 0); no directional-sensing term (alpha = 0); noise on every coordinate (sparse=False); noise only (alpha = gamma = lambda = 0).
Usage: python diag_dtlz4_dyn.py [--procs 2]   ->  results/diag_dtlz4_dyn.json / .md"""
import argparse, json
import numpy as np
from multiprocessing import Pool
from problems import dtlz, dtlz_pf
from metrics import igd
from mopeo4 import mopeo4

LO, HI = 0.05 ** 0.01, 0.95 ** 0.01
CFG = {"reported": dict(), "no leader attraction (gamma = lambda = 0)": dict(gamma=0.0, lam=0.0), "no directional-sensing term (alpha = 0)": dict(alpha=0.0),
       "noise on every coordinate": dict(sparse=False), "noise only (alpha = gamma = lambda = 0)": dict(alpha=0.0, gamma=0.0, lam=0.0)}


def one(a):
    name, seed = a
    prob = dtlz("DTLZ4"); P = dtlz_pf("DTLZ4")
    # mopeo4 returns only the archive objectives; interior membership is read from the objective vector: interior <=> both angles in (0.05, 0.95) of pi/2 (g ~ 0)
    AF = mopeo4(prob, 91, 30000, np.random.default_rng(seed), sparse=CFG[name].get("sparse", True), mode="delay", rspeed=0.1, **{k: v for k, v in CFG[name].items() if k != "sparse"})
    return name, seed, float(igd(AF, P)), int(len(AF))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=2); a = ap.parse_args()
    jobs = [(n, 7000 + i) for n in CFG for i in range(40)]
    with Pool(a.procs) as pool:
        out = pool.map(one, jobs, chunksize=1)
    res = {}
    for n, s, v, m in out:
        res.setdefault(n, []).append(v)
    md = ["# DTLZ4: dynamic ablations of one term of the update (probe; 40 seeds, 7000-7039, tau = 0.1)\n", "| configuration | failed runs (IGD > 0.2) | median IGD of the others | median IGD overall |\n|---|---|---|---|"]
    for n, v in res.items():
        v = np.array(v); f = v > 0.2
        md.append("| %s | %d/40 | %s | %.3g |" % (n, f.sum(), "%.4f" % np.median(v[~f]) if (~f).any() else "-", np.median(v)))
    json.dump({n: list(map(float, v)) for n, v in res.items()}, open("../results/diag_dtlz4_dyn.json", "w"))
    open("../results/diag_dtlz4_dyn.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
