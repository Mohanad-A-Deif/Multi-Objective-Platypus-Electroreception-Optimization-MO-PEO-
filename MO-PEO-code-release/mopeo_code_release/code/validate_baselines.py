#!/usr/bin/env python3
"""
validate_baselines.py -- internal validation of the additional baselines (MOPSO, GDE3, MOGWO).

Two parts, both run under the protocol of the main study
(25,000 evaluations on ZDT with N = 100; 30,000 on DTLZ with N = 91; 20 runs; seeds 1000 + run).

1. Sanity checks (each one caught a real bug earlier in this project)
     sphere   the search operators alone, in single-objective mode (M = 1) on a 30-D sphere, must converge
     box      no candidate handed to the objective function lies outside [L, U]; objectives are finite;
              the evaluation counter never exceeds the budget
     archive  the returned set is non-dominated, free of duplicates, no larger than N, and every
              returned objective vector equals f(x) of the decision vector stored with it
2. Range check against the stored results of NSGA-II and MOEA/D on ZDT1-ZDT3 and DTLZ2.
     A baseline that is more than about 2x worse than BOTH reference algorithms is treated as
     misimplemented or mis-parameterised.  Both readings of the rule are printed:
        lenient : IGD <= 2 * max(NSGA-II, MOEA/D)      (fails only if worse than both by > 2x)
        strict  : IGD <= 2 * min(NSGA-II, MOEA/D)      (within 2x of the better of the two)

   The multimodal / precision problems ZDT4, ZDT6 and DTLZ1 are run under the same protocol as EVIDENCE
   and reported (with archive sizes), but they do not gate the verdict: choosing settings on the problems
   where MO-PEO claims its wins would manufacture the outcome in either direction.
   MOPSO is reported in two configurations: "default" (w = 0.4, c1 = c2 = 1.0, 30 divisions) and "tuned"
   (best of the coarse probe in probe_mopso.py, run on non-protocol seeds 5000+ on ZDT4/ZDT6 only).

Usage:   python validate_baselines.py MOPSO [--config default|tuned|both] [--runs 20] [--results-dir results]
Output:  results/validation_new.json   (per run: IGD, HV, archive size; checkpointed, resumable)
         paper/table_validation_<alg>.tex (LaTeX snippet for the supplementary material)
"""
import argparse, json, os, sys
import numpy as np

from problems import zdt, zdt_pf, dtlz, dtlz_pf, nd_mask
from metrics import igd, hv_norm

SEED0 = 1000
CHECK_PROBLEMS = ["ZDT1", "ZDT2", "ZDT3", "DTLZ2"]      # acceptance set (gates the verdict)
EVIDENCE_PROBLEMS = ["ZDT4", "ZDT6", "DTLZ1"]            # reported with the same protocol, NOT used for the verdict


def problem(name):
    if name.startswith("ZDT"):
        D = 30 if name in ("ZDT1", "ZDT2", "ZDT3") else 10
        return zdt(name, D), zdt_pf(name), 100, 25000
    return dtlz(name), dtlz_pf(name), 91, 30000


def get_algorithm(name):
    if name == "MOPSO":
        from mopso import mopso; return mopso
    if name == "GDE3":
        from gde3 import gde3; return gde3
    if name == "MOGWO":
        from mogwo import mogwo; return mogwo
    raise SystemExit("unknown algorithm " + name)


# ------------------------------------------------------------------ sanity checks --
# Sphere-check specification per algorithm.
#   core_kw / core_FE / core_tol : the bare search operator (textbook coefficients, no extra mutation) must converge
#                                  to below core_tol within core_FE evaluations;
#   ref                          : reference the configured algorithm is compared with at the standard 25,000 evaluations:
#                                  "ga"  -> at least as good as an elitist GA with the NSGA-II operators (median);
#                                  "de"  -> agrees with an independent, loop-based DE/rand/1/bin (median within a factor 3).
# DE/rand/1/bin with N = 100 is simply slow on a 30-D sphere in 250 generations (the independent loop implementation
# also ends at ~1.6e-1), so for GDE3 the convergence test uses 10x the budget and the comparison test uses an
# independent implementation of the same operator instead of a GA. The first version of this check applied the
# MOPSO thresholds to GDE3 and failed it for that reason, not because of a defect (see HANDOVER).
SPHERE_SPEC = {
    "MOPSO": dict(core_kw=dict(w=0.7298, c1=1.49445, c2=1.49445, mutation="decay", pm0=0.0), core_FE=25000, core_tol=1e-3, ref="ga"),
    "GDE3": dict(core_kw={}, core_FE=250000, core_tol=1e-6, ref="de"),
    "MOGWO": dict(core_kw={}, core_FE=25000, core_tol=1e-3, ref="ga"),
}


def _sphere_reference_ga(seed, D=30, N=100, FE=25000):
    """Elitist GA with the SBX + polynomial-mutation operators of the NSGA-II baseline, for context."""
    from algorithms import sbx_pm
    rng = np.random.default_rng(seed)
    L, U = -5.0 * np.ones(D), 5.0 * np.ones(D)
    X = L + rng.random((N, D)) * (U - L); F = (X ** 2).sum(1); fe = N
    while fe + N <= FE:
        a, b = rng.integers(0, N, N), rng.integers(0, N, N)
        C = sbx_pm(X[np.where(F[a] < F[b], a, b)], L, U, rng); FC = (C ** 2).sum(1); fe += N
        i = np.argsort(np.r_[F, FC])[:N]; X, F = np.r_[X, C][i], np.r_[F, FC][i]
    return float(F.min())


def _sphere_reference_de(seed, D=30, N=100, FE=25000, F=0.5, CR=0.9):
    """Independent, deliberately un-vectorised generational DE/rand/1/bin with greedy selection."""
    rng = np.random.default_rng(seed)
    L, U = -5.0 * np.ones(D), 5.0 * np.ones(D)
    X = L + rng.random((N, D)) * (U - L); fx = (X ** 2).sum(1); fe = N
    while fe + N <= FE:
        Xn, fn = X.copy(), fx.copy()
        for i in range(N):
            r1, r2, r3 = rng.choice([j for j in range(N) if j != i], 3, replace=False)
            jr = rng.integers(D)
            u = np.clip(np.where((rng.random(D) < CR) | (np.arange(D) == jr), X[r1] + F * (X[r2] - X[r3]), X[i]), L, U)
            fu = float((u ** 2).sum()); fe += 1
            if fu <= fx[i]:
                Xn[i], fn[i] = u, fu
        X, fx = Xn, fn
    return float(fx.min())


def check_sphere(alg, name="MOPSO", verbose=True):
    """Two single-objective (M = 1) tests on the 30-D sphere in [-5, 5]^30 (start value ~ 1.5e2 - 2.5e2); see SPHERE_SPEC."""
    sp = SPHERE_SPEC[name]
    D = 30
    f = lambda X: (X ** 2).sum(1)[:, None]
    L, U = -5.0 * np.ones(D), 5.0 * np.ones(D)
    run = lambda kw, FE: [float(alg((f, L, U, 1), 100, FE, np.random.default_rng(SEED0 + s), **kw).min()) for s in range(5)]
    core, conf = run(sp["core_kw"], sp["core_FE"]), run({}, 25000)
    if sp["ref"] == "ga":
        ref = [_sphere_reference_ga(SEED0 + s) for s in range(5)]; rname = "reference GA with the NSGA-II operators"
        ok_b = np.median(conf) <= np.median(ref)
    else:
        ref = [_sphere_reference_de(SEED0 + s) for s in range(5)]; rname = "independent loop-based DE/rand/1/bin"
        ok_b = np.median(ref) / 3 <= np.median(conf) <= 3 * np.median(ref)
    ok_a = max(core) < sp["core_tol"]
    if verbose:
        print("  sphere-30D, 5 seeds (worst / median):")
        print("     (a) core update, %d evals, textbook coefficients      : %.2e / %.2e  (< %.0e)  -> %s"
              % (sp["core_FE"], max(core), np.median(core), sp["core_tol"], "PASS" if ok_a else "FAIL"))
        print("     (b) configured algorithm, 25,000 evals               : %.2e / %.2e" % (max(conf), np.median(conf)))
        print("         %-40s: %.2e / %.2e  -> %s" % (rname, max(ref), np.median(ref), "PASS" if ok_b else "FAIL"))
    return ok_a and ok_b, dict(core=core, configured=conf, reference=ref)


def check_box_and_archive(alg, verbose=True):
    ok_all = True
    for name in ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6", "DTLZ2"]:
        pr, P, N, FE = problem(name)
        f0, L, U = pr[0], pr[1], pr[2]
        stat = dict(calls=0, out=0, bad=0)

        def f(X):
            stat["calls"] += len(X)
            stat["out"] += int(((X < L) | (X > U)).any(1).sum())
            F = f0(X)
            stat["bad"] += int((~np.isfinite(F)).any(1).sum())
            return F

        info = {}
        A = alg((f,) + tuple(pr[1:]), N, FE, np.random.default_rng(SEED0), info=info)
        nd = bool(nd_mask(A).all())
        uniq = len(np.unique(np.round(A, 12), axis=0)) == len(A)
        size = len(A) <= N
        align = bool(np.allclose(f0(info["AX"]), A, rtol=1e-9, atol=1e-12)) if "AX" in info else None
        budget = stat["calls"] <= FE
        ok = stat["out"] == 0 and stat["bad"] == 0 and nd and uniq and size and budget and (align is not False)
        ok_all &= ok
        if verbose:
            print("  %-5s box-violations %d  non-finite %d | archive: non-dominated %s  unique %s  size %d<=%d  "
                  "aligned %s | evals %d<=%d -> %s"
                  % (name, stat["out"], stat["bad"], nd, uniq, len(A), N, align, stat["calls"], FE,
                     "PASS" if ok else "FAIL"))
    return ok_all


# ------------------------------------------------------------------- range check --
def label_of(name, cfg):
    return name if cfg == "default" else name + "-" + cfg.replace("_", "")


def config_kwargs(name, cfg):
    if cfg == "default":
        return {}
    if name == "MOPSO":
        import mopso as m
        return dict(m.TUNED if cfg == "tuned" else m.TUNED_EXT)
    raise SystemExit("no tuned configuration recorded for " + name)


def range_check(name, alg, runs, rdir, cfg="default"):
    kw, label = config_kwargs(name, cfg), label_of(name, cfg)
    ref = json.load(open(os.path.join(rdir, "results_final.json")))
    path = os.path.join(rdir, "validation_new.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for pn in CHECK_PROBLEMS + EVIDENCE_PROBLEMS:
        key = "%s|%s" % (pn, label)
        d = out.get(key, {})
        if "SIZE" not in d:                       # older entries lacked archive sizes; rerun is exact (fixed seeds)
            d = {"IGD": [], "HV": [], "SIZE": []}
        pr, P, N, FE = problem(pn)
        while len(d["IGD"]) < runs:
            A = alg(pr, N, FE, np.random.default_rng(SEED0 + len(d["IGD"])), **kw)
            d["IGD"].append(float(igd(A, P)))
            d["HV"].append(float(hv_norm(A, P)))
            d["SIZE"].append(int(len(A)))
            out[key] = d
            json.dump(out, open(path, "w"))
    return out, ref


def report(name, cfg, out, ref, tex_path=None):
    label = label_of(name, cfg)
    rows = []
    print("\n  == %s ==" % label)
    print("  %-6s %10s %10s %10s | %10s %10s | %6s %6s | %-17s | %s" %
          ("", "mean", "median", "max", "NSGA-II", "MOEA/D", "x N2", "x MD", "archive size", "rule"))
    all_len = all_str = True
    for grp, probs in (("acceptance set", CHECK_PROBLEMS), ("evidence only (no verdict)", EVIDENCE_PROBLEMS)):
        print("  -- %s" % grp)
        for pn in probs:
            d = out["%s|%s" % (pn, label)]
            v = np.array(d["IGD"]); sz = np.array(d["SIZE"])
            n2 = np.mean(ref["%s|NSGA-II" % pn]["IGD"]); md = np.mean(ref["%s|MOEA/D" % pn]["IGD"])
            lenient = v.mean() <= 2 * max(n2, md); strict = v.mean() <= 2 * min(n2, md)
            gate = pn in CHECK_PROBLEMS
            if gate:
                all_len &= lenient; all_str &= strict
            rows.append((pn, v.mean(), np.median(v), v.max(), n2, md, v.mean() / n2, v.mean() / md, sz.min(), sz.max(), gate))
            print("  %-6s %10.3e %10.3e %10.3e | %10.3e %10.3e | %6.2f %6.2f | %3d-%-3d (mean %5.1f) | %s" %
                  (pn, v.mean(), np.median(v), v.max(), n2, md, v.mean() / n2, v.mean() / md, sz.min(), sz.max(), sz.mean(),
                   ("pass" if lenient else "FAIL") + "/" + ("pass" if strict else "FAIL") if gate else "-"))
    print("  verdict on the acceptance set: lenient rule %s, strict rule %s" %
          ("PASS" if all_len else "FAIL", "PASS" if all_str else "FAIL"))
    if tex_path:
        os.makedirs(os.path.dirname(tex_path) or ".", exist_ok=True)
        L = [r"\begin{table}[htbp]\centering",
             r"\caption{Internal validation of %s under the main protocol (20 runs). Mean IGD; ratio columns divide the %s "
             r"mean by that of NSGA-II and MOEA/D. The first four problems form the acceptance set (ratio at most 2 against at "
             r"least one reference algorithm); the last three are reported as evidence only. The last column gives the range of "
             r"the number of archive members returned.}" % (label, label),
             r"\label{tab:validation_%s}" % label.lower().replace("/", "").replace("-", "_"),
             r"\begin{tabular}{lrrrrrrr}\toprule",
             r"Problem & mean & median & NSGA-II & MOEA/D & ratio N & ratio M & archive \\\midrule"]
        for r in rows:
            if r[0] == EVIDENCE_PROBLEMS[0]:
                L.append(r"\midrule")
            L.append("%s & %.2e & %.2e & %.2e & %.2e & %.2f & %.2f & %d--%d \\\\" % (r[0], r[1], r[2], r[4], r[5], r[6], r[7], r[8], r[9]))
        L += [r"\bottomrule\end{tabular}\end{table}"]
        open(tex_path, "w").write("\n".join(L) + "\n")
    return all_len, all_str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("algorithm", choices=["MOPSO", "GDE3", "MOGWO"])
    ap.add_argument("--config", choices=["default", "tuned", "tuned_ext", "both", "all"], default="default")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--paper-dir", default="../paper")
    ap.add_argument("--skip-checks", action="store_true")
    a = ap.parse_args()
    alg = get_algorithm(a.algorithm)
    if not a.skip_checks:
        print("== sanity checks: %s ==" % a.algorithm)
        ok1, sph = check_sphere(alg, a.algorithm)
        ok2 = check_box_and_archive(alg)
        print("  checks: %s" % ("ALL PASS" if ok1 and ok2 else "SOMETHING FAILED"))
    for cfg in {"both": ["default", "tuned"], "all": ["default", "tuned", "tuned_ext"]}.get(a.config, [a.config]):
        print("\n== range check, %s (%d runs, seeds %d+i) ==" % (label_of(a.algorithm, cfg), a.runs, SEED0))
        out, ref = range_check(a.algorithm, alg, a.runs, a.results_dir, cfg)
        report(a.algorithm, cfg, out, ref, os.path.join(a.paper_dir, "table_validation_%s.tex" % label_of(a.algorithm, cfg).lower()))


if __name__ == "__main__":
    main()
