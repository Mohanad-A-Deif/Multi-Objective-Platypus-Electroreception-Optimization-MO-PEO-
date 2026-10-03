#!/usr/bin/env python3
"""
reproduce.py — run the whole MO-PEO study end to end.

Everything in the manuscript is produced by this script: the baseline validation,
the main comparison, the ablation, the two alternative designs, the tau sweep,
the convergence curves, the timing measurements, the LaTeX tables and the figures.

Usage
-----
    python reproduce.py                       # every stage, full protocol (~35 min, 1 core)
    python reproduce.py --stage main          # one stage only
    python reproduce.py --stage main ablation
    python reproduce.py --runs 3 --problems ZDT1 DTLZ2   # quick smoke test
    python reproduce.py --list                # show stages and exit

Stages
------
    validate    NSGA-II convergence metric against Deb et al. (2002)   -> stdout + validation.json
    main        MO-PEO, NSGA-II, MOEA/D, MOPSO, GDE3, MOGWO on the full suite -> results_final.json
    ablation    A0 / A1 / B2 (components removed)                      -> results_final.json
    variants    S1 (split density), S2 (spatial p_rel)                 -> results_final.json
    supp_mopso  MOPSO with the coarse-grid "tuned" configuration (supplementary) -> results_final.json
    tau         sensitivity to the deferred-refinement threshold       -> tau.json
    convergence IGD against budget                                     -> conv.pkl
    fronts      median-run fronts used by figures 1 and 3              -> fronts_med.pkl
    timing      wall-clock time                                        -> timing.json
    audit       standing reproducibility check: re-run seed 1000 of every stored key and flag any
                mismatch against results_final.json; exits non-zero on a mismatch, before tables/figures
                are built (added 2026-09-23 after A1-sparse/A0-noSparse-noUnion were found stale)
                                                                        -> audit_repro.json, audit_repro.md
    tables      LaTeX tables from the stored results                   -> tables.tex, tables_supp.tex
    figures     all five figures                                       -> figs/

Every stage is resumable: results are written after each (problem, algorithm)
pair, and a rerun skips whatever is already complete. Delete the corresponding
file in results/ to force recomputation.

Seeds are fixed at 1000 + run index, so repeated executions give identical
numbers on any machine. Requires numpy, scipy and matplotlib only.
"""
import argparse, json, os, pickle, sys, time
import numpy as np

from problems import zdt, zdt_pf, dtlz, dtlz_pf
from metrics import igd, hv_norm
from algorithms import nsga2, moead
from mopeo4 import mopeo4
from mopeo5 import mopeo5
from mopso import mopso, TUNED as MOPSO_TUNED
from gde3 import gde3
from mogwo import mogwo

# ------------------------------------------------------------------- protocol --
RESULTS_DIR = "results"
ZDT_NAMES = ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6"]
DTLZ_NAMES = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]
ALL_PROBLEMS = ZDT_NAMES + DTLZ_NAMES
RUNS = 20
SEED0 = 1000

# MO-PEO as reported: localised perturbation + deferred refinement. TAU is the value selected by the pre-registered
# calibration on seeds 6000..6039 (calibrate_tau.py, analyse_tau_calib.py; tuning half of the suite only). The protocol
# seeds (1000+i) are used only for this final run. TAU_OLD = 0.4 was the value fixed earlier on protocol seeds; its
# results are kept under the '-tau0.4' keys for comparison.
TAU = 0.1
TAU_OLD = 0.4
MOPEO = dict(sparse=True, mode="delay", rspeed=TAU)

ALGORITHMS = {
    # main comparison
    "MO-PEO-final":        lambda p, N, FE, r: mopeo4(p, N, FE, r, **MOPEO),
    "NSGA-II":             nsga2,
    "MOEA/D":              moead,
    # additional baselines (validated with validate_baselines.py; see HANDOVER.md section 7.1)
    "MOPSO":               mopso,                                                    # default configuration = official
    "GDE3":                gde3,
    "MOGWO":               mogwo,
    "MOPSO-tuned":         lambda p, N, FE, r: mopso(p, N, FE, r, **MOPSO_TUNED),   # supplementary only
    # ablation: A1 keeps only the localised perturbation, B2 only the schedule,
    # A0 neither (i.e. the plain multi-objective port of PEO)
    "A1-sparse":           lambda p, N, FE, r: mopeo4(p, N, FE, r, sparse=True,  mode="fixed", prel=0.0),
    "B2-delayOnly":        lambda p, N, FE, r: mopeo4(p, N, FE, r, sparse=False, mode="delay", rspeed=TAU),
    "A0-noSparse-noUnion": lambda p, N, FE, r: mopeo4(p, N, FE, r, sparse=False, mode="fixed", prel=0.0),
    # alternative diversification designs discussed in the appendix
    "S1-splitDensity":     lambda p, N, FE, r: mopeo5(p, N, FE, r, kmode="split",  pmode="time", tau=TAU),
    "S2-spatialPrel":      lambda p, N, FE, r: mopeo5(p, N, FE, r, kmode="sparse", pmode="spatial"),
    # option (b): MO-PEO without the deferred-refinement schedule (localised perturbation only; identical in definition to A1)
    "MO-PEO-noSchedule":   lambda p, N, FE, r: mopeo4(p, N, FE, r, sparse=True, mode="fixed", prel=0.0),
    # the same designs at the earlier value tau = 0.4 (kept for comparison; not part of the main tables)
    "MO-PEO-tau0.4":       lambda p, N, FE, r: mopeo4(p, N, FE, r, sparse=True, mode="delay", rspeed=TAU_OLD),
    "B2-delayOnly-tau0.4": lambda p, N, FE, r: mopeo4(p, N, FE, r, sparse=False, mode="delay", rspeed=TAU_OLD),
    "S1-splitDensity-tau0.4": lambda p, N, FE, r: mopeo5(p, N, FE, r, kmode="split", pmode="time", tau=TAU_OLD),
}

STAGE_ALGS = dict(
    main=["MO-PEO-final", "NSGA-II", "MOEA/D", "MOPSO", "GDE3", "MOGWO"],
    supp_mopso=["MOPSO-tuned"],
    ablation=["A1-sparse", "B2-delayOnly", "A0-noSparse-noUnion"],
    variants=["S1-splitDensity", "S2-spatialPrel"],
    option_b=["MO-PEO-noSchedule"],
    tau_old=["MO-PEO-tau0.4", "B2-delayOnly-tau0.4", "S1-splitDensity-tau0.4"],
)


def problem(name):
    """Return (problem tuple, true front, population size, evaluation budget)."""
    if name.startswith("ZDT"):
        D = 30 if name in ("ZDT1", "ZDT2", "ZDT3") else 10
        return zdt(name, D), zdt_pf(name), 100, 25000
    return dtlz(name), dtlz_pf(name), 91, 30000


# ---------------------------------------------------------------- persistence --
def rpath(fname):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, fname)


def load_json(fname, default=None):
    p = rpath(fname)
    return json.load(open(p)) if os.path.exists(p) else (default if default is not None else {})


def save_json(obj, fname):
    json.dump(obj, open(rpath(fname), "w"))


def load_pkl(fname, default=None):
    p = rpath(fname)
    return pickle.load(open(p, "rb")) if os.path.exists(p) else (default if default is not None else {})


def save_pkl(obj, fname):
    pickle.dump(obj, open(rpath(fname), "wb"))


# -------------------------------------------------------------------- stages --
ALG_FILTER = None      # set by --algs: restrict the algorithms of a stage (e.g. to run several processes in parallel)


def stage_runs(alg_names, problems, runs):
    """Shared driver for main / ablation / variants: fills results_final.json."""
    if ALG_FILTER:
        alg_names = [a for a in alg_names if a in ALG_FILTER]
    res = load_json("results_final.json")
    for name in problems:
        pr, P, N, FE = problem(name)
        for a in alg_names:
            key = "%s|%s" % (name, a)
            d = res.get(key, {"IGD": [], "HV": []})
            if len(d["IGD"]) >= runs:
                print("  skip %-28s (already %d runs)" % (key, len(d["IGD"])))
                continue
            t0 = time.time()
            while len(d["IGD"]) < runs:
                A = ALGORITHMS[a](pr, N, FE, np.random.default_rng(SEED0 + len(d["IGD"])))
                d["IGD"].append(float(igd(A, P)))
                d["HV"].append(float(hv_norm(A, P)))
                res[key] = d
                save_json(res, "results_final.json")      # checkpoint after every run
            print("  %-28s IGD %.4g   HV %.4f   (%.0f s)"
                  % (key, np.mean(d["IGD"]), np.mean(d["HV"]), time.time() - t0), flush=True)


def stage_validate(problems, runs):
    """NSGA-II convergence metric gamma against the published values."""
    published = {"ZDT1": 0.033, "ZDT2": 0.072, "ZDT3": 0.114, "ZDT4": 0.513, "ZDT6": 0.296}
    out = load_json("validation.json")
    for name in [p for p in problems if p in published]:
        if name in out:
            continue
        pr, P_, _, FE = problem(name)
        P = zdt_pf(name, 500)                              # 500 reference points, as in Deb et al.
        g = []
        for s in range(max(6, runs // 3)):
            A = nsga2(pr, 100, FE, np.random.default_rng(s))
            g.append(float(np.sqrt(((A[:, None, :] - P[None]) ** 2).sum(2)).min(1).mean()))
        out[name] = [float(np.mean(g)), published[name]]
        save_json(out, "validation.json")
        print("  %-6s gamma %.4f   (published %.3f)" % (name, np.mean(g), published[name]), flush=True)


def stage_tau(problems, runs):
    """Calibration of tau on fresh seeds 6000..6039 (pre-registered protocol; see calibrate_tau.py)."""
    os.system("%s calibrate_tau.py --resume && %s analyse_tau_calib.py" % (sys.executable, sys.executable))


def stage_convergence(problems, runs):
    """Independent runs at increasing budgets; each run honours its own schedule."""
    fracs = [0.08, 0.16, 0.28, 0.44, 0.64, 0.84, 1.0]
    sel = [p for p in problems if p in ("ZDT1", "ZDT4", "DTLZ1", "DTLZ6")]
    out = load_pkl("conv.pkl")
    n_runs = max(5, runs // 4)
    for name in sel:
        pr, P, N, FEmax = problem(name)
        for a in STAGE_ALGS["main"]:
            for fc in fracs:
                key = (name, a, fc)
                if key in out:
                    continue
                FE = max(3 * N, int(FEmax * fc))
                v = [igd(ALGORITHMS[a](pr, N, FE, np.random.default_rng(SEED0 + s)), P) for s in range(n_runs)]
                out[key] = (FE, float(np.mean(v)))
                save_pkl(out, "conv.pkl")
            print("  %-6s %-14s done" % (name, a), flush=True)


def stage_fronts(problems, runs):
    """Store the front of the median run (by IGD) for the plotted problems."""
    sel = [p for p in problems if p in ("ZDT1", "ZDT3", "ZDT4", "ZDT6", "DTLZ2")]
    out = load_pkl("fronts_med.pkl")
    n_runs = max(5, runs // 4)
    for name in sel:
        pr, P, N, FE = problem(name)
        for a in STAGE_ALGS["main"]:
            if (name, a) in out:
                continue
            got = []
            for s in range(n_runs):
                A = ALGORITHMS[a](pr, N, FE, np.random.default_rng(SEED0 + s))
                got.append((igd(A, P), A))
            got.sort(key=lambda x: x[0])
            out[(name, a)] = got[len(got) // 2][1]
            save_pkl(out, "fronts_med.pkl")
            print("  %-6s %-14s median IGD %.4g" % (name, a, got[len(got) // 2][0]), flush=True)
    # reference fronts, so the plotting script can run standalone
    ref = {n: (zdt_pf(n) if n.startswith("ZDT") else dtlz_pf(n))
           for n in ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6", "DTLZ1", "DTLZ2", "DTLZ5", "DTLZ6", "DTLZ7"]}
    save_pkl(ref, "ref_fronts.pkl")


def stage_timing(problems, runs):
    out = load_json("timing.json")
    sel = [p for p in problems if p in ("ZDT1", "DTLZ2")]
    for name in sel:
        pr, P, N, FE = problem(name)
        for a in STAGE_ALGS["main"]:
            key = "%s|%s" % (name, a.replace("-final", ""))
            if key in out:
                continue
            ts = []
            for s in range(5):
                t = time.perf_counter()
                ALGORITHMS[a](pr, N, FE, np.random.default_rng(s))
                ts.append(time.perf_counter() - t)
            out[key] = [float(np.mean(ts)), float(np.std(ts))]
            save_json(out, "timing.json")
            print("  %-22s %.2f +- %.2f s" % (key, np.mean(ts), np.std(ts)), flush=True)


def stage_audit(problems, runs):
    """Standing reproducibility check (added 2026-09-23, after A1-sparse/A0-noSparse-noUnion were found stale):
    re-run the FIRST protocol seed (SEED0 = 1000) of every stored (problem, algorithm) key whose algorithm name
    is still in ALGORITHMS, and compare the IGD bit for bit against results_final.json. A key kept only for the
    record under a "-stored-old" (or similarly renamed) suffix is not in ALGORITHMS and is skipped on purpose.
    Any mismatch means the stored numbers no longer match the code that would produce them -- this must never
    reach a table or the manuscript silently, so a mismatch is printed AND the stage exits non-zero, which stops
    `python reproduce.py` before the tables/figures stages run on stale data."""
    stored = load_json("results_final.json")
    per_alg, mism, checked = {}, [], 0
    for key, d in sorted(stored.items()):
        prob, alg = key.split("|", 1)
        if alg not in ALGORITHMS or prob not in problems or not d.get("IGD"):
            continue
        pr, P, N, FE = problem(prob)
        v = float(igd(ALGORITHMS[alg](pr, N, FE, np.random.default_rng(SEED0)), P))
        ok = (v == d["IGD"][0]); checked += 1
        per_alg.setdefault(alg, []).append((prob, ok))
        if not ok:
            mism.append((prob, alg, v, d["IGD"][0]))
        print("  %-24s %-8s %s" % (alg, prob, "OK" if ok else "MISMATCH (got %.6g, stored %.6g)" % (v, d["IGD"][0])), flush=True)
    res = {a: dict(n=len(r), identical=sum(ok for _, ok in r), problems_not_identical=sorted(p for p, ok in r if not ok))
           for a, r in per_alg.items()}
    save_json(res, "audit_repro.json")
    md = ["# Reproducibility audit: seed %d of every stored key re-run with the current code\n" % SEED0,
          "| algorithm key | runs checked | bit-identical | problems with a mismatch |", "|---|---|---|---|"]
    for a, d in res.items():
        md.append("| %s | %d | %d | %s |" % (a, d["n"], d["identical"], ", ".join(d["problems_not_identical"]) or "-"))
    open(rpath("audit_repro.md"), "w").write("\n".join(md) + "\n")
    if mism:
        print("\nAUDIT FAILED: %d of %d stored keys did not reproduce bit-for-bit at seed %d:" % (len(mism), checked, SEED0))
        for prob, alg, v, s in mism:
            print("  %s|%s: got %r, results_final.json has %r" % (prob, alg, v, s))
        sys.exit(1)
    print("\naudit passed: %d stored keys reproduce bit-for-bit at seed %d" % (checked, SEED0))


def stage_tables(problems, runs):
    for mod in ("mktables", "mktables_supp"):
        if os.path.exists(mod + ".py"):
            os.system("%s %s.py" % (sys.executable, mod))
            print("  ran", mod)


def stage_figures(problems, runs):
    if os.path.exists("make_figures.py"):
        os.system("%s make_figures.py" % sys.executable)


STAGES = {
    "validate":    stage_validate,
    "main":        lambda p, r: stage_runs(STAGE_ALGS["main"], p, r),
    "ablation":    lambda p, r: stage_runs(STAGE_ALGS["ablation"], p, r),
    "variants":    lambda p, r: stage_runs(STAGE_ALGS["variants"], p, r),
    "supp_mopso":  lambda p, r: stage_runs(STAGE_ALGS["supp_mopso"], p, r),
    "tau_old":     lambda p, r: stage_runs(STAGE_ALGS["tau_old"], p, r),
    "option_b":    lambda p, r: stage_runs(STAGE_ALGS["option_b"], p, r),
    "tau":         stage_tau,
    "convergence": stage_convergence,
    "fronts":      stage_fronts,
    "timing":      stage_timing,
    "audit":       stage_audit,
    "tables":      stage_tables,
    "figures":     stage_figures,
}
ORDER = ["validate", "main", "ablation", "variants", "supp_mopso", "tau_old", "option_b", "tau", "convergence",
         "fronts", "timing", "audit", "tables", "figures"]


def main():
    global RESULTS_DIR, ALG_FILTER
    ap = argparse.ArgumentParser(description="Reproduce the MO-PEO study.")
    ap.add_argument("--stage", nargs="+", default=ORDER, choices=ORDER + ["all"])
    ap.add_argument("--problems", nargs="+", default=ALL_PROBLEMS)
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--algs", nargs="+", default=None, help="restrict the algorithms of the main/ablation/variants/supp stages")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("stages:", ", ".join(ORDER)); print("problems:", ", ".join(ALL_PROBLEMS)); return
    RESULTS_DIR = a.results_dir
    ALG_FILTER = a.algs
    stages = ORDER if "all" in a.stage else [s for s in ORDER if s in a.stage]
    t0 = time.time()
    for s in stages:
        print("\n== %s ==" % s, flush=True)
        STAGES[s](a.problems, a.runs)
    print("\nfinished in %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
