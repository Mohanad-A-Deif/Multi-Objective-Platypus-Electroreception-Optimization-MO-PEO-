#!/usr/bin/env python3
"""Dose-response of the perturbation density (fraction of D coordinates perturbed per agent) on DTLZ4 and on
four problems already known/suspected to be harmed by denser perturbation. Protocol and acceptance rule are
pre-declared in results/dtlz4_dose_protocol.md -- READ THAT FIRST; nothing here may change the rule after the
fact. Shipped configuration is density = 1/D (mode="delay", rspeed=TAU=0.1); this script only varies density.
40 calibration seeds (6000-6039), NOT the protocol seeds and NOT the diagnosis seeds.
Usage: python diag_dtlz4_dose.py [--procs 2]  ->  results/dtlz4_dose.json / .md"""
import argparse, json
import numpy as np
from multiprocessing import Pool
import reproduce as R
from metrics import igd
from mopeo4 import mopeo4

PROBLEMS = ["DTLZ4", "ZDT4", "DTLZ1", "DTLZ3", "DTLZ6"]
DENS_LABEL = ["1/D", "0.1", "0.25", "0.5", "1.0"]
SEEDS = list(range(6000, 6040))


def densities_for(name):
    _, _, _, _ = R.problem(name)  # sanity: problem exists
    pr, P, N, FE = R.problem(name)
    D = len(pr[1])
    return D, [1.0 / D, 0.1, 0.25, 0.5, 1.0]


def one(job):
    name, dlabel, density, seed = job
    pr, P, N, FE = R.problem(name)
    AF = mopeo4(pr, N, FE, np.random.default_rng(seed), sparse=True, mode="delay", rspeed=0.1, density=density)
    return name, dlabel, seed, float(igd(AF, P))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=2); a = ap.parse_args()
    jobs = []
    Ds = {}
    for name in PROBLEMS:
        D, dens = densities_for(name)
        Ds[name] = D
        for dlabel, density in zip(DENS_LABEL, dens):
            for s in SEEDS:
                jobs.append((name, dlabel, density, s))
    print("D per problem:", Ds, flush=True)
    print("%d jobs on %d procs" % (len(jobs), a.procs), flush=True)
    out = []
    with Pool(a.procs) as pool:
        for k, o in enumerate(pool.imap_unordered(one, jobs, chunksize=4)):
            out.append(o)
            if (k + 1) % 50 == 0:
                print("  %d/%d" % (k + 1, len(jobs)), flush=True)

    res = {}  # res[problem][dlabel] = list of IGD over the 40 seeds
    for name, dlabel, seed, v in out:
        res.setdefault(name, {}).setdefault(dlabel, {})[seed] = v
    res = {name: {dlabel: [res[name][dlabel][s] for s in SEEDS] for dlabel in DENS_LABEL} for name in PROBLEMS}

    json.dump({"D": Ds, "igd": res, "seeds": SEEDS, "density_labels": DENS_LABEL}, open("../results/dtlz4_dose.json", "w"), indent=1)

    md = ["# DTLZ4 dose-response: perturbation density (fraction of D coordinates perturbed per agent)\n",
          "Pre-declared protocol and acceptance rule: `results/dtlz4_dose_protocol.md`. Shipped configuration = "
          "density 1/D (reported). 40 calibration seeds (6000-6039). D per problem: %s.\n" % ", ".join("%s=%d" % (n, Ds[n]) for n in PROBLEMS)]

    md.append("## DTLZ4 failure rate (IGD > 0.2) by density\n")
    md.append("| density | failed / 40 | median IGD |\n|---|---|---|")
    dtlz4_fail = {}
    for dlabel in DENS_LABEL:
        v = np.array(res["DTLZ4"][dlabel])
        fail = int((v > 0.2).sum())
        dtlz4_fail[dlabel] = fail
        md.append("| %s | %d | %.4g |" % (dlabel, fail, np.median(v)))

    md.append("\n## Median IGD on the four problems already known/suspected to be harmed by denser noise, and the ratio to density = 1/D\n")
    md.append("| problem | " + " | ".join(DENS_LABEL) + " |\n|---|" + "---|" * len(DENS_LABEL))
    ratios = {}
    for name in ["ZDT4", "DTLZ1", "DTLZ3", "DTLZ6"]:
        meds = {dlabel: float(np.median(res[name][dlabel])) for dlabel in DENS_LABEL}
        base = meds["1/D"]
        ratios[name] = {dlabel: meds[dlabel] / base for dlabel in DENS_LABEL}
        md.append("| %s | " % name + " | ".join("%.4g (%.2fx)" % (meds[dlabel], ratios[name][dlabel]) for dlabel in DENS_LABEL) + " |")

    md.append("\n## Acceptance rule (pre-declared; see protocol file) applied to each density\n")
    md.append("| density | DTLZ4 failed < 10/40 | ZDT4 <= 1.5x | DTLZ1 <= 1.5x | DTLZ6 <= 1.5x | ADOPT |\n|---|---|---|---|---|---|")
    adopted = []
    for dlabel in DENS_LABEL:
        c1 = dtlz4_fail[dlabel] < 10
        c2 = ratios["ZDT4"][dlabel] <= 1.5
        c3 = ratios["DTLZ1"][dlabel] <= 1.5
        c4 = ratios["DTLZ6"][dlabel] <= 1.5
        ok = c1 and c2 and c3 and c4
        if ok:
            adopted.append(dlabel)
        md.append("| %s | %s | %s | %s | %s | %s |" % (dlabel, c1, c2, c3, c4, "YES" if ok else "no"))

    md.append("\n## Verdict\n")
    if adopted:
        chosen = adopted[0]  # DENS_LABEL is already in increasing order, smallest density preferred
        md.append("**Densities satisfying all four conditions: %s. Smallest, %s, would be preferred if adopted.** "
                   "Per the pre-declared protocol this still requires the full 12-problem/20-run protocol and the "
                   "standing reproducibility audit before it could replace the shipped configuration." % (", ".join(adopted), chosen))
    else:
        md.append("**No density on the grid satisfies all four conditions. Per the pre-declared protocol, none is "
                   "adopted.** The DTLZ4 discovery failure is reported as a characterised limitation (mechanism "
                   "analysis in `results/diag_dtlz4_mech.md`); MO-PEO-final (density = 1/D) remains shipped.")

    open("../results/dtlz4_dose.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
