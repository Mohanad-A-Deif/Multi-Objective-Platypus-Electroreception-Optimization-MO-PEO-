#!/usr/bin/env python3
"""
diag_dtlz2.py -- diagnosis of MO-PEO on DTLZ2 (observation only; no modification of the algorithm is proposed or tried).

Question: MO-PEO is significantly worse than every opponent on DTLZ2 (mean IGD 0.0962 against 0.0758 for MOEA/D).
Is it (D2) weight-coherent leader selection concentrating the agents on few archive members, (D4) crowding-distance
truncation of the archive discarding useful spread, or neither (e.g. a convergence deficit)?

What is measured, per iteration, for MO-PEO (the final configuration of the paper, replicated line for line from
mopeo4.py) and, with the same definitions, for MOEA/D (replicated from algorithms.moead):

  size            archive size after the update (MO-PEO) / number of non-dominated members of the population (MOEA/D)
  trunc           MO-PEO: 1 if the non-dominated union A u P exceeded N_A, so that crowding truncation removed members
  pre_size, cut   MO-PEO: size of the non-dominated union before truncation and number of members removed
  churn           MO-PEO: number of members of the previous archive that are no longer in the new archive
  igd             IGD of the set (against the 5050-point true front used everywhere in the study)
  igd_pre         MO-PEO: IGD of the non-dominated union BEFORE truncation (what the run held and then cut down)
  igd_cf          MO-PEO: IGD of the same pre-truncation set cut to N_A by iterative crowding truncation (the member
                  with the smallest crowding distance removed, distances recomputed after every removal). A measurement
                  made on the same set at every step; the run itself is NOT changed.
  gd              convergence: mean | ||f|| - 1 | of the members (DTLZ2: the true front is the unit sphere)
  gap             coverage: max over true-front points of the distance to the nearest member (largest hole)
  nn_mean/std/max spacing of the set: nearest-neighbour distance between members
  leaders         MO-PEO: number of DISTINCT archive members chosen as leader by the N agents; lead_max = the largest
                  number of agents sharing one leader; idle = archive members that are nobody's leader
  distinct        MOEA/D: number of distinct objective vectors in the population (subproblems can hold copies)
  phi_*           spread of the Tchebycheff values phi(x_i | w_i) of the agents (MOEA/D: of the subproblem solutions),
                  normalised with the ideal / nadir of the set as in the algorithm: mean, std, min, max
  exc_*           excess of the agent over the best value attainable for its weight,
                  e_i = phi(x_i | w_i) - min_{p in true front} phi(p | w_i); quantiles across agents and the number of
                  agents with e_i > 0.05 ("unserved weight")
  excL_*          the same for the LEADER of agent i (MO-PEO): e^L_i = phi(a_i | w_i) - min_p phi(p | w_i)
  tgt_*           distance in objective space between the leader (MO-PEO) / subproblem solution (MOEA/D) of weight i and
                  the Tchebycheff-optimal true-front point of that weight
  dir_*           distance from the direction of each weight vector, w_i / ||w_i||, to the nearest member of the set
                  (mean and maximum over the 91 weights)
  s               MO-PEO: fraction of agents in the exploitation phase (s_i = 1)
  fe              evaluations spent

Seeds: 5000 + i (probe seeds; the reported protocol seeds 1000 + i are used only to verify that the instrumented copies
reproduce the stored results exactly).
"""
import argparse, json, os, pickle, sys, time
import numpy as np
from problems import dtlz, dtlz_pf, nd_mask, crowding
from algorithms import weights, trunc_archive, nsga2, moead
from mopeo4 import mopeo4
from metrics import igd

FIN = dict(sparse=True, mode="delay", rspeed=0.4)


# ---------------------------------------------------------------- measurements --
def nn_stats(A):
    if len(A) < 2:
        return 0.0, 0.0, 0.0
    d = np.sqrt(((A[:, None] - A[None]) ** 2).sum(2)); np.fill_diagonal(d, np.inf); m = d.min(1)
    return float(m.mean()), float(m.std()), float(m.max())


def set_metrics(A, P):
    D = np.sqrt(((P[:, None] - A[None]) ** 2).sum(2))
    nn = nn_stats(A)
    return dict(igd=float(D.min(1).mean()), gd=float(np.abs(np.linalg.norm(A, axis=1) - 1).mean()), gap=float(D.min(1).max()),
                nn_mean=nn[0], nn_std=nn[1], nn_max=nn[2], size=len(A))


def dir_dist(A, W):
    a = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    w = W / np.linalg.norm(W, axis=1, keepdims=True)
    d = np.sqrt(((w[:, None] - a[None]) ** 2).sum(2)).min(1)
    return float(d.mean()), float(d.max())


def iterative_truncation(F, NA):
    """Cut F to NA members: repeatedly remove the member with the smallest crowding distance (recomputed each time)."""
    idx = np.arange(len(F))
    while len(idx) > NA:
        cd = crowding(F[idx]); idx = np.delete(idx, int(np.argmin(cd)))
    return F[idx]


def q(x, qs=(50, 90)):
    return [float(np.percentile(x, p)) for p in qs]


class Target:
    """Best Tchebycheff value and optimal true-front point for every weight (true front normalised as the algorithm does)."""
    def __init__(self, W, P):
        Z = (W[:, None, :] * P[None]).max(2)          # (N, |P|)  ideal 0, nadir 1, eps ~ 0
        k = Z.argmin(1)
        self.best = Z[np.arange(len(W)), k]; self.pt = P[k]


# ------------------------------------------------------------- MO-PEO traced ----
def mopeo_traced(prob, N, FE, rng, P, alpha=1.5, beta=1.0, gamma=0.5, lam=0.8, zeta=0.9, eps=1e-6, theta=1e-3, NA=None,
                 bs=0.1, zs=0.1, rspeed=0.4):
    """Line-for-line copy of mopeo4(..., sparse=True, mode='delay', rspeed=0.4); the only additions are measurements
    that read the state and never touch rng."""
    f, L, U, M = prob[:4]; D = len(L); span = U - L; NA = NA or N
    W = np.maximum(weights(M, N), 1e-6); N = len(W); b = beta * bs * span; z = zeta * zs * span
    tgt = Target(W, P)
    X = L + rng.random((N, D)) * span; F = f(X); fe = N; AX, AF = trunc_archive(X, F, NA); Fp = None
    T = FE // N; t = 0
    tr = []
    while fe + N <= FE:
        zmin, zmax = AF.min(0), AF.max(0); r = np.maximum(zmax - zmin, 1e-12)
        nrm = lambda G: (G - zmin) / (r + eps); phi = lambda G, w: (w * nrm(G)).max(1)
        pc = phi(F, W); s = np.zeros(N, bool) if Fp is None else np.abs(pc - phi(Fp, W)) > theta
        pr = max(0.0, (t / T - rspeed) / (1 - rspeed))
        j = (np.arange(N) + rng.integers(1, N, N)) % N; dXn = (X[j] - X) / span; dP = phi(F[j], W) - pc
        E = -alpha * (dP / (np.linalg.norm(dXn, axis=1) + eps))[:, None] * dXn * span
        lead_idx = ((W[:, None, :] * nrm(AF)[None]).max(2)).argmin(1)
        lead = AX[lead_idx]
        K = np.zeros((N, D), bool); K[np.arange(N), rng.integers(0, D, N)] = True
        Mt = b * (2 * rng.random((N, D)) - 1) * K
        use = (rng.random(N) < pr)[:, None]
        Xi = np.where(use, np.abs(X - lead) * rng.standard_normal((N, D)), z * rng.standard_normal((N, D))) * K
        Xn = np.clip(np.where(s[:, None], X + lam * (lead - X) + Xi, X + E + Mt + gamma * (lead - X)), L, U)
        relstep = (np.abs(X - lead) * K).sum(1)              # size of the relative-branch step of every agent (before the move)
        xdev_ag = float(np.abs(X[:, M - 1:] - 0.5).mean())   # DTLZ2 distance variables: deviation from the optimum 0.5
        grp_rel, grp_abs, grp_exp = s & use[:, 0], s & ~use[:, 0], ~s   # exploitation with the relative / absolute noise, exploration
        Fp = F; X = Xn; F = f(X); fe += N; t += 1

        # ---- archive update (the real one) and the measurements around it
        UX, UF = np.r_[AX, X], np.r_[AF, F]
        m = nd_mask(UF); PF = UF[m]
        _, u = np.unique(np.round(PF, 12), axis=0, return_index=True); pre = PF[u]          # non-dominated union before truncation
        oldF = AF
        AX, AF = trunc_archive(UX, UF, NA)
        rec = dict(fe=fe, s=float(s.mean()), pre_size=len(pre), trunc=int(len(pre) > NA), cut=max(0, len(pre) - NA))
        key = lambda G: set(map(tuple, np.round(G, 12)))
        rec["churn"] = len(key(oldF) - key(AF))
        kept = key(AF); gdrow = lambda G: np.abs(np.linalg.norm(G, axis=1) - 1)
        rem = np.array([row for row in pre if tuple(np.round(row, 12)) not in kept])
        rec["gd_kept"] = float(gdrow(AF).mean()); rec["gd_removed"] = float(gdrow(rem).mean()) if len(rem) else float("nan")
        rec["entered"] = len(key(F) & kept)                                # offspring of this iteration that are in the new archive
        newkey = [tuple(np.round(row, 12)) for row in F]
        entered_i = np.array([k in kept for k in newkey])
        improved_i = phi(F, W) < pc                                        # own Tchebycheff value improved (same normalisation as the move)
        for gname, g in (("rel", grp_rel), ("abs", grp_abs), ("exp", grp_exp)):
            rec["n_" + gname] = int(g.sum())
            rec["imp_" + gname] = float(improved_i[g].mean()) if g.any() else float("nan")
            rec["ent_" + gname] = float(entered_i[g].mean()) if g.any() else float("nan")
        rec["relstep_med"] = float(np.median(relstep)); rec["xdev_agents"] = xdev_ag
        rec["xdev_arch"] = float(np.abs(AX[:, M - 1:] - 0.5).mean()); rec["pr"] = pr
        rec.update(set_metrics(AF, P))
        pm = set_metrics(pre, P); rec["igd_pre"] = pm["igd"]; rec["gap_pre"] = pm["gap"]
        if len(pre) > NA:
            cf = set_metrics(iterative_truncation(pre, NA), P); rec["igd_cf"] = cf["igd"]; rec["gap_cf"] = cf["gap"]
        else:
            rec["igd_cf"] = rec["igd"]; rec["gap_cf"] = rec["gap"]
        cnt = np.bincount(lead_idx, minlength=len(oldF))
        rec.update(leaders=int((cnt > 0).sum()), lead_max=int(cnt.max()), idle=int((cnt == 0).sum()), archive_before=len(oldF))
        pcn = pc                                       # phi of the agents that were acted on in this iteration
        rec.update(phi_mean=float(pcn.mean()), phi_std=float(pcn.std()), phi_min=float(pcn.min()), phi_max=float(pcn.max()))
        e = pcn - tgt.best
        rec.update(exc_med=q(e)[0], exc_p90=q(e)[1], exc_max=float(e.max()), unserved=int((e > 0.05).sum()))
        LF = oldF[lead_idx]                                # objective vectors of the leaders
        eL = phi(LF, W) - tgt.best
        rec.update(excL_med=q(eL)[0], excL_p90=q(eL)[1], excL_max=float(eL.max()), unservedL=int((eL > 0.05).sum()))
        dl = np.linalg.norm(LF - tgt.pt, axis=1)
        rec.update(tgt_mean=float(dl.mean()), tgt_p90=q(dl)[1], tgt_max=float(dl.max()))
        rec["dir_mean"], rec["dir_max"] = dir_dist(AF, W)
        tr.append(rec)
    return AF, tr


# ------------------------------------------------------------- MOEA/D traced ----
def moead_traced(prob, N, FE, rng, P, T=20, delta=0.9, nr=2):
    """Line-for-line copy of algorithms.moead; the measurements are taken after every sweep of N children."""
    from algorithms import sbx_pm
    f, L, U, M = prob[:4]; D = len(L)
    W = np.maximum(weights(M, N), 1e-6); N = len(W)
    tgt = Target(W, P)
    B = np.argsort(((W[:, None] - W[None]) ** 2).sum(2), 1)[:, :T]
    X = L + rng.random((N, D)) * (U - L); F = f(X); fe = N; zs = F.min(0)
    tr = []
    while fe + N <= FE:
        for i in rng.permutation(N):
            pool = B[i] if rng.random() < delta else np.arange(N)
            p = rng.choice(pool, 2, replace=False)
            c = sbx_pm(X[p], L, U, rng)[:1]; fc = f(c)[0]; fe += 1
            zs = np.minimum(zs, fc)
            g_old = (W[pool] * np.abs(F[pool] - zs)).max(1); g_new = (W[pool] * np.abs(fc - zs)).max(1)
            upd = pool[rng.permutation(np.where(g_new <= g_old)[0])][:nr]
            X[upd] = c[0]; F[upd] = fc
        nd = F[nd_mask(F)]
        _, u = np.unique(np.round(nd, 12), axis=0, return_index=True); nd = nd[u]
        rec = dict(fe=fe)
        rec.update(set_metrics(nd, P))
        rec["distinct"] = len(np.unique(np.round(F, 12), axis=0))
        zmin, zmax = nd.min(0), nd.max(0); r = np.maximum(zmax - zmin, 1e-12)
        phi = lambda G, w: (w * (G - zmin) / (r + 1e-6)).max(1)
        pc = phi(F, W)
        rec.update(phi_mean=float(pc.mean()), phi_std=float(pc.std()), phi_min=float(pc.min()), phi_max=float(pc.max()))
        e = pc - tgt.best
        rec.update(exc_med=q(e)[0], exc_p90=q(e)[1], exc_max=float(e.max()), unserved=int((e > 0.05).sum()))
        dl = np.linalg.norm(F - tgt.pt, axis=1)
        rec.update(tgt_mean=float(dl.mean()), tgt_p90=q(dl)[1], tgt_max=float(dl.max()))
        rec["dir_mean"], rec["dir_max"] = dir_dist(nd, W)
        tr.append(rec)
    return F[nd_mask(F)], tr


# ------------------------------------------------------------------- driver ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--verify", action="store_true", help="check the instrumented copies against the stored protocol results (seeds 1000+i)")
    ap.add_argument("--out", default="../results/diag_dtlz2.pkl")
    a = ap.parse_args()
    prob = dtlz("DTLZ2"); P = dtlz_pf("DTLZ2"); N, FE = 91, 30000

    if a.verify:
        stored = json.load(open("../results/results_final.json"))
        for i in range(3):
            A, _ = mopeo_traced(prob, N, FE, np.random.default_rng(1000 + i), P)
            B, _ = moead_traced(prob, N, FE, np.random.default_rng(1000 + i), P)
            g1, g2 = igd(A, P), igd(B, P)
            s1, s2 = stored["DTLZ2|MO-PEO-final"]["IGD"][i], stored["DTLZ2|MOEA/D"]["IGD"][i]
            print("seed %d  MO-PEO traced %.15g stored %.15g  %s | MOEA/D traced %.15g stored %.15g  %s"
                  % (1000 + i, g1, s1, "IDENTICAL" if g1 == s1 else "DIFFERENT", g2, s2, "IDENTICAL" if g2 == s2 else "DIFFERENT"), flush=True)
        return

    out = dict(mopeo=[], moead=[], final=[])
    t0 = time.time()
    for i in range(a.runs):
        seed = a.seed0 + i
        A, tm = mopeo_traced(prob, N, FE, np.random.default_rng(seed), P)
        B, td = moead_traced(prob, N, FE, np.random.default_rng(seed), P)
        C = nsga2(prob, N, FE, np.random.default_rng(seed))
        out["mopeo"].append(tm); out["moead"].append(td)
        out["final"].append(dict(seed=seed, MOPEO=A, MOEAD=B, NSGA2=C))
        pickle.dump(out, open(a.out, "wb"))
        print("seed %d: IGD MO-PEO %.4f  MOEA/D %.4f  NSGA-II %.4f  (%.0f s)" % (seed, igd(A, P), igd(B, P), igd(C, P), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
