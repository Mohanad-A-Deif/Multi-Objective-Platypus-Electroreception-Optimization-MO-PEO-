#!/usr/bin/env python3
"""
tau_diag.py -- does the deferred-refinement schedule hurt well-conditioned unimodal fronts?

Runs MO-PEO (final configuration, only the schedule threshold tau varied) on DTLZ2, DTLZ4, DTLZ5 (and DTLZ6 as a positive
control, where the schedule is known to help) with NON-PROTOCOL seeds (>= 5000) and records the archive IGD and the
generic convergence measure gd_ref (mean distance of the archive to the nearest point of the reference front) along the run.

    tau = 1.0  ->  schedule disabled: mode="fixed", prel=0.0 (absolute noise only; identical to ablation A1)
    tau < 1.0  ->  mode="delay", rspeed=tau

`mopeo4_traced` is a line-for-line copy of mopeo4.mopeo4 (measurements never touch the random generator); --verify checks that it
reproduces mopeo4 bit for bit on the protocol seeds 1000..1002.

Common random numbers: for a given seed the random stream is consumed identically for every tau (both noise branches are always
drawn), and p_rel = 0 until t/T = tau, so runs with different tau are IDENTICAL up to the iteration t = tau*T. Any difference
afterwards is caused by the relative-noise branch alone; the analysis therefore uses paired (per-seed) differences.

Statistics fixed before the runs (see analyse_tau.py): paired median difference with a 95% percentile bootstrap interval
(10,000 resamples of the seeds); Wilcoxon signed-rank as a secondary check. If the interval of a primary contrast contains 0
on 20 seeds, that problem is extended with seeds 5020..5059 (60 in total), for all values of tau.
"""
import argparse, os, pickle, sys, time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import dtlz, dtlz_pf, zdt, zdt_pf, nd_mask
from algorithms import weights, trunc_archive
from metrics import igd
from mopeo4 import mopeo4

TAUS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
PROBS = ["DTLZ2", "DTLZ4", "DTLZ5", "DTLZ6"]
EVERY = 4


def kw_of(tau):
    return dict(sparse=True, mode="fixed", prel=0.0) if tau >= 1.0 else dict(sparse=True, mode="delay", rspeed=tau)


def gd_ref(A, P):
    return float(np.sqrt(((A[:, None, :] - P[None, :, :]) ** 2).sum(2)).min(1).mean())


def mopeo4_traced(prob, N, FE, rng, P, alpha=1.5, beta=1.0, gamma=0.5, lam=0.8, zeta=0.9, eps=1e-6, theta=1e-3, NA=None,
                  bs=0.1, zs=0.1, mode="fixed", sparse=True, prel=0.7, floor=0.01, rspeed=1.0, up=0.05, dn=0.05, lo=0.3, hi=0.9):
    f, L, U, M = prob[:4]; D = len(L); span = U - L; NA = NA or N
    W = np.maximum(weights(M, N), 1e-6); N = len(W); b = beta * bs * span; z = zeta * zs * span
    X = L + rng.random((N, D)) * span; F = f(X); fe = N; AX, AF = trunc_archive(X, F, NA); Fp = None
    T = FE // N; t = 0; pr_i = np.full(N, prel)
    tr = dict(t=[], igd=[], gd=[], s=[])
    while fe + N <= FE:
        zmin, zmax = AF.min(0), AF.max(0); r = np.maximum(zmax - zmin, 1e-12)
        nrm = lambda G: (G - zmin) / (r + eps); phi = lambda G, w: (w * nrm(G)).max(1)
        pc = phi(F, W); s = np.zeros(N, bool) if Fp is None else np.abs(pc - phi(Fp, W)) > theta
        if mode == "delay": pr_i[:] = max(0.0, (t / T - rspeed) / (1 - rspeed))
        j = (np.arange(N) + rng.integers(1, N, N)) % N; dXn = (X[j] - X) / span; dP = phi(F[j], W) - pc
        E = -alpha * (dP / (np.linalg.norm(dXn, axis=1) + eps))[:, None] * dXn * span
        lead = AX[((W[:, None, :] * nrm(AF)[None]).max(2)).argmin(1)]
        if sparse:
            K = np.zeros((N, D), bool); K[np.arange(N), rng.integers(0, D, N)] = True
        else:
            K = np.ones((N, D), bool)
        Mt = b * (2 * rng.random((N, D)) - 1) * K
        use = (rng.random(N) < pr_i)[:, None]
        Xi = np.where(use, np.abs(X - lead) * rng.standard_normal((N, D)), z * rng.standard_normal((N, D))) * K
        Xn = np.clip(np.where(s[:, None], X + lam * (lead - X) + Xi, X + E + Mt + gamma * (lead - X)), L, U)
        Fp = F; X = Xn; F = f(X); fe += N; t += 1
        AX, AF = trunc_archive(np.r_[AX, X], np.r_[AF, F], NA)
        if t % EVERY == 0 or fe + N > FE:
            tr["t"].append(t); tr["igd"].append(float(igd(AF, P))); tr["gd"].append(gd_ref(AF, P)); tr["s"].append(float(s.mean()))
    tr["T"] = T
    return AF, tr


def setup(name):
    """(problem, reference front, N, FE) exactly as in reproduce.problem"""
    if name.startswith("ZDT"):
        return zdt(name, 30 if name in ("ZDT1", "ZDT2", "ZDT3") else 10), zdt_pf(name), 100, 25000
    return dtlz(name), dtlz_pf(name), 91, 30000


def one(args):
    name, tau, seed = args
    pr, P, N, FE = setup(name)
    AF, tr = mopeo4_traced(pr, N, FE, np.random.default_rng(seed), P, **kw_of(tau))
    tr["final_igd"] = float(igd(AF, P)); tr["size"] = len(AF)
    return (name, tau, seed), tr


def verify():
    ok = True
    for name in ("DTLZ2", "DTLZ6", "ZDT4", "DTLZ3"):
        pr, P, N, FE = setup(name)
        for tau in (0.4, 1.0):
            for s in (1000, 1001, 1002):
                A0 = mopeo4(pr, N, FE, np.random.default_rng(s), **kw_of(tau))
                A1, _ = mopeo4_traced(pr, N, FE, np.random.default_rng(s), P, **kw_of(tau))
                same = A0.shape == A1.shape and np.array_equal(A0, A1)
                ok &= same
                print(name, tau, s, "IDENTICAL" if same else "DIFFERENT", "IGD %.6f" % igd(A0, P), flush=True)
    print("VERIFY", "OK" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--problems", nargs="+", default=PROBS)
    ap.add_argument("--taus", nargs="+", type=float, default=TAUS)
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--out", default="../results/tau_diag_a.pkl")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    if a.verify:
        sys.exit(0 if verify() else 1)
    assert a.seed0 >= 5000, "non-protocol seeds only"
    jobs = [(n, tau, a.seed0 + i) for n in a.problems for tau in a.taus for i in range(a.runs)]
    out, t0 = {}, time.time()
    with Pool(a.procs) as p:
        for k, (key, tr) in enumerate(p.imap_unordered(one, jobs, chunksize=2)):
            out[key] = tr
            if (k + 1) % 20 == 0:
                print("%d/%d  %.0fs" % (k + 1, len(jobs), time.time() - t0), flush=True)
    pickle.dump(out, open(a.out, "wb"))
    print("saved", a.out, len(out))


if __name__ == "__main__":
    main()
