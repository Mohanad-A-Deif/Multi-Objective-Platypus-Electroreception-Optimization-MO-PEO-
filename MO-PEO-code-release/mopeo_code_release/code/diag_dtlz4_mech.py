#!/usr/bin/env python3
"""DTLZ4: WHERE does the interior of the front get lost in the update rule of MO-PEO? (observation only; nothing is changed)

The interior of the DTLZ4 front is 0.05 < u1, u2 < 0.95 with u = x^100, i.e. x1, x2 in the window (0.9705, 0.99949). The clipping to the box acts coordinate by coordinate and is the identity
on the window, so a candidate is interior BEFORE the projection if and only if it is interior after it: interior candidates cannot be "generated and then projected away". What can be
measured is (i) how often interior candidates are generated, per branch, and (ii) which term of the update limits that rate.

mopeo_mech is a line-for-line copy of mopeo4 (sparse=True, mode="delay") with recorders. Random numbers of the run are drawn in the same order as in mopeo4 (verified: identical archives).
In every iteration, from the same state, R counterfactual candidate sets are drawn with a SEPARATE random stream (they never influence the run) under variants of the move:
  full      the reported move
  no_att    no leader attraction (gamma = lambda = 0)
  no_E      no directional-sensing term (E = 0)
  no_noise  no noise term (Mt = 0 and Xi = 0)
  dense     the noise acts on every coordinate instead of a single one
  rnd_lead  the leader of each agent is a random archive member instead of the archive member with the smallest weighted Tchebycheff value
For each variant and branch (s = 0 exploration, s = 1 exploitation) the recorder stores: candidates, candidates with both x1 and x2 in the window (interior), with x1 in the window, with x2 in the window.
Usage: python diag_dtlz4_mech.py [--runs 40] [--seed0 7000] [--R 50] [--procs 2] [--tau 0.1] [--out ../results/diag_dtlz4_mech.pkl] [--verify]
"""
import argparse, pickle, sys, time
import numpy as np
from multiprocessing import Pool
from problems import dtlz, dtlz_pf, nd_mask
from algorithms import weights, trunc_archive
from metrics import igd
from mopeo4 import mopeo4

LO, HI = 0.05 ** 0.01, 0.95 ** 0.01            # 0.97049..., 0.99949...
VARIANTS = ["full", "no_att", "no_E", "no_noise", "dense", "rnd_lead"]


def win(x):
    return (x > LO) & (x < HI)


def gen(rng, X, F, Fp, s, lead, AX, W, nrm_AF_dummy, params, variant, phi, pc, span, L, U, prel):
    """One candidate set for all agents from the state (X, lead, s, ...), exactly the formulas of mopeo4 with the variant's ablation. Returns the RAW (unclipped) candidates."""
    N, D = X.shape
    alpha, gamma, lam, b, z = params
    if variant == "rnd_lead":
        lead = AX[rng.integers(0, len(AX), N)]
    j = (np.arange(N) + rng.integers(1, N, N)) % N
    dXn = (X[j] - X) / span
    dP = phi(F[j], W) - pc
    E = -alpha * (dP / (np.linalg.norm(dXn, axis=1) + 1e-6))[:, None] * dXn * span
    if variant == "dense":
        K = np.ones((N, D), bool)
    else:
        K = np.zeros((N, D), bool); K[np.arange(N), rng.integers(0, D, N)] = True
    Mt = b * (2 * rng.random((N, D)) - 1) * K
    use = (rng.random(N) < prel)[:, None]
    Xi = np.where(use, np.abs(X - lead) * rng.standard_normal((N, D)), z * rng.standard_normal((N, D))) * K
    g_, l_ = (0.0, 0.0) if variant == "no_att" else (gamma, lam)
    if variant == "no_E":
        E = 0.0 * E
    if variant == "no_noise":
        Mt = 0.0 * Mt; Xi = 0.0 * Xi
    return np.where(s[:, None], X + l_ * (lead - X) + Xi, X + E + Mt + g_ * (lead - X))


def mopeo_mech(prob, N, FE, rng, hrng=None, R=50, tau=0.1, record=True):
    f, L, U, M = prob[:4]; D = len(L); span = U - L
    alpha, gamma, lam, beta, zeta, eps, theta, bs, zs = 1.5, 0.5, 0.8, 1.0, 0.9, 1e-6, 1e-3, 0.1, 0.1
    W = np.maximum(weights(M, N), 1e-6); N = len(W); b = beta * bs * span; z = zeta * zs * span
    X = L + rng.random((N, D)) * span; F = f(X); fe = N; AX, AF = trunc_archive(X, F, N); Fp = None
    T = FE // N; t = 0; tr = []
    while fe + N <= FE:
        zmin, zmax = AF.min(0), AF.max(0); r = np.maximum(zmax - zmin, 1e-12)
        nrm = lambda G: (G - zmin) / (r + eps); phi = lambda G, w: (w * nrm(G)).max(1)
        pc = phi(F, W); s = np.zeros(N, bool) if Fp is None else np.abs(pc - phi(Fp, W)) > theta
        prel = max(0.0, (t / T - tau) / (1 - tau))
        j = (np.arange(N) + rng.integers(1, N, N)) % N; dXn = (X[j] - X) / span; dP = phi(F[j], W) - pc
        E = -alpha * (dP / (np.linalg.norm(dXn, axis=1) + eps))[:, None] * dXn * span
        lead = AX[((W[:, None, :] * nrm(AF)[None]).max(2)).argmin(1)]
        K = np.zeros((N, D), bool); K[np.arange(N), rng.integers(0, D, N)] = True
        Mt = b * (2 * rng.random((N, D)) - 1) * K
        use = (rng.random(N) < prel)[:, None]
        Xi = np.where(use, np.abs(X - lead) * rng.standard_normal((N, D)), z * rng.standard_normal((N, D))) * K
        Xraw = np.where(s[:, None], X + lam * (lead - X) + Xi, X + E + Mt + gamma * (lead - X))
        Xn = np.clip(Xraw, L, U)
        if record:
            rec = dict(t=t, n_exp=int((~s).sum()), n_expl=int(s.sum()))
            for br, msk in (("e", ~s), ("x", s)):
                xr = Xraw[msk]; xc = Xn[msk]
                rec[br + "_raw_int"] = int((win(xr[:, 0]) & win(xr[:, 1])).sum())
                rec[br + "_clip_int"] = int((win(xc[:, 0]) & win(xc[:, 1])).sum())
                rec[br + "_raw_w1"] = int(win(xr[:, 0]).sum()); rec[br + "_raw_w2"] = int(win(xr[:, 1]).sum())
                rec[br + "_over1"] = int(((xr[:, 0] > 1) | (xr[:, 0] < 0)).sum()); rec[br + "_over2"] = int(((xr[:, 1] > 1) | (xr[:, 1] < 0)).sum())
                # interior after clipping, but the raw point was NOT (a candidate that the projection turned into an interior point) and vice versa
                rec[br + "_proj_gain"] = int(((win(xc[:, 0]) & win(xc[:, 1])) & ~(win(xr[:, 0]) & win(xr[:, 1]))).sum())
                rec[br + "_proj_loss"] = int((~(win(xc[:, 0]) & win(xc[:, 1])) & (win(xr[:, 0]) & win(xr[:, 1]))).sum())
            # counterfactual candidate sets from the same state, separate random stream
            cf = np.zeros((len(VARIANTS), 2, 4), np.int64)      # variant, branch(e,x), [cands, interior, w1, w2]
            params = (alpha, gamma, lam, b, z)
            for vi, v in enumerate(VARIANTS):
                for _ in range(R):
                    C = gen(hrng, X, F, Fp, s, lead, AX, W, None, params, v, phi, pc, span, L, U, prel)
                    C = np.clip(C, L, U)
                    for bi, msk in enumerate((~s, s)):
                        c = C[msk]
                        cf[vi, bi, 0] += len(c); cf[vi, bi, 1] += int((win(c[:, 0]) & win(c[:, 1])).sum())
                        cf[vi, bi, 2] += int(win(c[:, 0]).sum()); cf[vi, bi, 3] += int(win(c[:, 1]).sum())
            rec["cf"] = cf
            # state: agents' marginal window occupancy and the leaders'
            rec["ag_w1"] = float(win(X[:, 0]).mean()); rec["ag_w2"] = float(win(X[:, 1]).mean()); rec["ag_int"] = float((win(X[:, 0]) & win(X[:, 1])).mean())
            rec["ld_w1"] = float(win(lead[:, 0]).mean()); rec["ld_w2"] = float(win(lead[:, 1]).mean())
            rec["ar_int"] = int((win(AX[:, 0]) & win(AX[:, 1])).sum())
            tr.append(rec)
        Fp = F; X = Xn; F = f(X); fe += N; t += 1
        AX, AF = trunc_archive(np.r_[AX, X], np.r_[AF, F], N)
    return AF, tr


def one(a):
    seed, R, tau = a
    prob = dtlz("DTLZ4"); P = dtlz_pf("DTLZ4")
    AF, tr = mopeo_mech(prob, 91, 30000, np.random.default_rng(seed), np.random.default_rng(10 ** 6 + seed), R=R, tau=tau)
    return seed, dict(final_igd=float(igd(AF, P)), tr=tr)


def verify(tau):
    prob = dtlz("DTLZ4"); ok = True
    for seed in (7000, 7001):
        A = mopeo4(prob, 91, 30000, np.random.default_rng(seed), sparse=True, mode="delay", rspeed=tau)
        B, _ = mopeo_mech(prob, 91, 30000, np.random.default_rng(seed), np.random.default_rng(1), R=2, tau=tau)
        same = A.shape == B.shape and np.array_equal(A, B); ok &= same
        print("seed", seed, "identical to mopeo4:", same, flush=True)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=40); ap.add_argument("--seed0", type=int, default=7000); ap.add_argument("--R", type=int, default=50)
    ap.add_argument("--procs", type=int, default=2); ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--out", default="../results/diag_dtlz4_mech.pkl"); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        sys.exit(0 if verify(a.tau) else 1)
    t0 = time.time(); res = {}
    with Pool(a.procs) as pool:
        for k, (seed, d) in enumerate(pool.imap_unordered(one, [(a.seed0 + i, a.R, a.tau) for i in range(a.runs)])):
            res[seed] = d; print("%d/%d %.0fs" % (k + 1, a.runs, time.time() - t0), flush=True)
    pickle.dump(res, open(a.out, "wb")); print("saved", a.out, len(res))
