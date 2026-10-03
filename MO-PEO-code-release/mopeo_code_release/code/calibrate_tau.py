#!/usr/bin/env python3
"""
calibrate_tau.py -- calibration of the deferred-refinement threshold tau on FRESH seeds, with a declared split of the suite.

PRE-REGISTRATION (written to results/tau_calib_protocol.json BEFORE any calibration run; not to be changed after the results are seen)

  * Calibration seeds 6000..6039 (40 runs per problem and tau). Never the protocol seeds 1000..1019. The protocol seeds are used only for
    the final reported run with the selected tau.
  * tau grid: 0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0 (1.0 = schedule disabled: mode="fixed", prel=0). Everything else is the final configuration.
  * Split of the 12 problems into a TUNING half and a HELD-OUT half: stratified by structure of the problem definition (strata listed below),
    random within strata with numpy default_rng(20260920). The split was computed once and is not re-drawn. The strata are the author's judgement;
    the author had already seen exploratory tau results on seven problems (seeds 5000..5059) when defining them.
  * Selection rule (tuning half only): tau minimising the geometric mean over the tuning problems of the regret ratio
    r_p(tau) = median IGD_p(tau) / min over tau' of median IGD_p(tau').  The held-out half is looked at only after tau has been fixed.
  * 'Defensible' (declared): on the held-out problems (a) the geometric-mean regret of the selected tau is <= 1.5 and (b) at most 2 of the 6
    held-out problems have a regret ratio > 2. If not, we report that no single tau is defensible.
  * Secondary analyses: the same rule with the two halves swapped, and leave-one-problem-out (tau chosen on the other 11 problems, regret
    measured on the one left out). Sensitivity of every problem to tau: medians with 95% bootstrap intervals.
Limitations of the regret measure: the per-problem minimum over eight noisy medians is optimistically low, so all regrets are biased upward by an
amount that grows with the noise of the problem (largest for ZDT4, DTLZ1, DTLZ3, DTLZ6).
"""
import argparse, json, os, pickle, sys, time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tau_diag import setup, kw_of
from metrics import igd
from mopeo4 import mopeo4

PROBS = ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6", "DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]
TAUS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
SEEDS = list(range(6000, 6040))
STRATA = {"smooth unimodal 2-objective": ["ZDT1", "ZDT2"],
          "disconnected front": ["ZDT3", "DTLZ7"],
          "biased density / hard distance function": ["ZDT6", "DTLZ4", "DTLZ6"],
          "many local fronts": ["ZDT4", "DTLZ1", "DTLZ3"],
          "smooth unimodal 3-objective": ["DTLZ2", "DTLZ5"]}


def make_split():
    rng = np.random.default_rng(20260920); coin = int(rng.integers(2)); tune, test = [], []
    for name, mem in STRATA.items():
        perm = [str(x) for x in rng.permutation(mem)]
        k = len(mem) // 2 if len(mem) % 2 == 0 else ((2 if coin else 1) if name.startswith("biased") else (1 if coin else 2))
        tune += perm[:k]; test += perm[k:]
    return sorted(tune), sorted(test)


_REF = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "ref_fronts.pkl"), "rb"))
# true fronts are computed once and cached in ref_fronts.pkl (building DTLZ7's takes more than a minute)


def one(a):
    name, tau, seed = a
    if name.startswith("ZDT"):
        from problems import zdt
        pr, N, FE = zdt(name, 30 if name in ("ZDT1", "ZDT2", "ZDT3") else 10), 100, 25000
    else:
        from problems import dtlz
        pr, N, FE = dtlz(name), 91, 30000
    P = _REF[name] if name in _REF else _REF["DTLZ2"]     # DTLZ3 and DTLZ4 share the unit-sphere front (5050 points)
    return (name, tau, seed), float(igd(mopeo4(pr, N, FE, np.random.default_rng(seed), **kw_of(tau)), P))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../results/tau_calib.pkl"); ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--resume", action="store_true"); a = ap.parse_args()
    tune, test = make_split()
    proto = dict(seeds=[SEEDS[0], SEEDS[-1]], taus=TAUS, tune=tune, heldout=test, strata=STRATA, split_seed=20260920,
                 rule="geometric mean over tuning problems of median-IGD regret ratio", defensible="held-out GM regret <= 1.5 and at most 2 of 6 held-out regrets > 2",
                 written=time.strftime("%Y-%m-%d %H:%M:%S"))
    pp = "../results/tau_calib_protocol.json"
    if not os.path.exists(pp):
        json.dump(proto, open(pp, "w"), indent=1)
    print("TUNE", tune, "HELD-OUT", test, flush=True)
    out = pickle.load(open(a.out, "rb")) if (a.resume and os.path.exists(a.out)) else {}
    jobs = [(n, t, s) for s in SEEDS for n in PROBS for t in TAUS if (n, t, s) not in out]
    t0 = time.time()
    with Pool(a.procs) as p:
        for k, (key, v) in enumerate(p.imap_unordered(one, jobs, chunksize=4)):
            out[key] = v
            if (k + 1) % 240 == 0:
                pickle.dump(out, open(a.out, "wb")); print("%d/%d %.0fs" % (k + 1, len(jobs), time.time() - t0), flush=True)
    pickle.dump(out, open(a.out, "wb")); print("saved", len(out))


if __name__ == "__main__":
    main()
