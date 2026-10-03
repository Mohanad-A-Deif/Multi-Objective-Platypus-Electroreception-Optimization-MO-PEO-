#!/usr/bin/env python3
"""Reproducibility audit: re-run the first N_SEEDS protocol seeds (1000, 1001, ...) of every stored (algorithm, problem) key with the CURRENT code
and compare the IGD with results_final.json bit for bit. Writes results/audit_repro.json and results/audit_repro.md."""
import json, sys, numpy as np
from multiprocessing import Pool
import reproduce as R
from metrics import igd

N_SEEDS = 1
ALGS = ["MO-PEO-final", "MO-PEO-tau0.4", "MO-PEO-noSchedule", "A1-sparse", "A0-noSparse-noUnion", "B2-delayOnly", "B2-delayOnly-tau0.4", "S1-splitDensity", "S1-splitDensity-tau0.4",
        "S2-spatialPrel", "NSGA-II", "MOEA/D", "MOPSO", "GDE3", "MOGWO", "MOPSO-tuned"]


def job(a):
    alg, prob, i = a
    pr, P, N, FE = R.problem(prob)
    A = R.ALGORITHMS[alg](pr, N, FE, np.random.default_rng(R.SEED0 + i))
    return alg, prob, i, float(igd(A, P))


if __name__ == "__main__":
    stored = json.load(open("../results/results_final.json"))
    jobs = [(a, p, i) for a in ALGS for p in R.ALL_PROBLEMS for i in range(N_SEEDS) if "%s|%s" % (p, a) in stored]
    import time; t0 = time.time(); out = []
    with Pool(2) as pool:
        for k, o in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            out.append(o); print("%d/%d %.0fs %s %s %s" % (k + 1, len(jobs), time.time() - t0, o[0], o[1], o[3] == stored["%s|%s" % (o[1], o[0])]["IGD"][o[2]]), flush=True)
    rows = {}
    for alg, prob, i, v in out:
        s = stored["%s|%s" % (prob, alg)]["IGD"][i]
        rows.setdefault(alg, []).append((prob, i, v, s, v == s))
    res = {a: dict(n=len(r), identical=sum(x[4] for x in r), problems_not_identical=sorted({x[0] for x in r if not x[4]})) for a, r in rows.items()}
    json.dump(res, open("../results/audit_repro.json", "w"), indent=1)
    md = ["# Reproducibility audit: first %d protocol seeds of every stored key re-run with the current code\n" % N_SEEDS, "| algorithm key | runs checked | bit-identical | problems with a mismatch |", "|---|---|---|---|"]
    for a, d in res.items():
        md.append("| %s | %d | %d | %s |" % (a, d["n"], d["identical"], ", ".join(d["problems_not_identical"]) or "-"))
    open("../results/audit_repro.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
