"""Constrained-domination archive/selection utilities (Deb 2000: feasible beats infeasible; among two
infeasible solutions the one with lower total violation wins; among two feasible solutions, ordinary
Pareto dominance) and the six optimizers run on the radiotherapy case study (HANDOVER 7.2).

These are separate from problems.py/algorithms.py/mopeo4.py on purpose: the 12-problem benchmark
suite has no constraints, its archive/selection code is exactly what the standing reproducibility
audit (reproduce.py --stage audit) checks bit-for-bit, and none of it is touched here.

mopeo_rt uses exactly the shipped MO-PEO-final hyperparameters and update rule (sparse=True,
mode="delay", rspeed=TAU=0.1 -- see reproduce.py's MOPEO dict) applied to a bi-objective, two-
constraint problem; the only change from mopeo4.py is the constrained archive. nsga2_rt is the
textbook constrained-NSGA-II (Deb 2002) with the same constrained-domination rule in the tournament
and in environmental selection. moead_rt, mopso_rt, gde3_rt and mogwo_rt apply the same
constrained-domination principle (CDP: feasible beats infeasible; infeasible-infeasible decided by
total violation; feasible-feasible decided by ordinary dominance / the algorithm's own scalar
criterion) to the exact update rules of algorithms.py's moead(), mopso.py's mopso(), gde3.py's gde3()
and mogwo.py's mogwo(), so all six algorithms are compared under the same constraint-handling rule
and none is given an advantage by a different one. Every optimizer also records a violation
trajectory (fe, min total violation of that generation's population/archive) so progress toward
feasibility -- not just the final feasibility count -- can be read off.
"""
import numpy as np
from problems import nd_mask, nd_sort, crowding
from algorithms import weights, sbx_pm
from mopso import _leaders as _mopso_leaders, _poly_mutation
from mogwo import _cell_keys, _leaders_for_pack


def nd_mask_con(F, V):
    """Non-dominated mask under constrained-domination. If any point is feasible (V<=0), the
    non-dominated set is the ordinary Pareto front of the feasible subset (every infeasible point
    is constrained-dominated by any feasible one). If none is feasible, it is the objective-space
    non-dominated subset among those with (numerically) the smallest total violation."""
    n = len(F)
    feas = V <= 0
    mask = np.zeros(n, bool)
    if feas.any():
        mask[feas] = nd_mask(F[feas])
    else:
        vmin = V.min()
        cand = np.where(V <= vmin + 1e-12)[0]
        mask[cand[nd_mask(F[cand])]] = True
    return mask


def trunc_archive_con(X, F, V, NA):
    m = nd_mask_con(F, V)
    X, F, V = X[m], F[m], V[m]
    _, u = np.unique(np.round(np.c_[F, V], 12), axis=0, return_index=True)
    X, F, V = X[u], F[u], V[u]
    if len(F) > NA:
        cd = crowding(F)
        keep = np.argsort(-cd)[:NA]
        X, F, V = X[keep], F[keep], V[keep]
    return X, F, V


def cdom_rank_crowd_select(F, V, n):
    """Constrained analogue of select_by_rank_crowd: feasible solutions are ranked and selected
    first (ordinary non-dominated sorting + crowding among them), then infeasible solutions fill any
    remaining slots in increasing order of violation."""
    feas = np.where(V <= 0)[0]
    infeas = np.where(V > 0)[0]
    idx = []
    if len(feas):
        rank = nd_sort(F[feas])
        for r in range(rank.max() + 1):
            fr = feas[rank == r]
            if len(idx) + len(fr) <= n:
                idx.extend(fr.tolist())
            else:
                cd = crowding(F[fr]); idx.extend(fr[np.argsort(-cd)[:n - len(idx)]].tolist())
                break
    if len(idx) < n and len(infeas):
        order = infeas[np.argsort(V[infeas])]
        idx.extend(order[:n - len(idx)].tolist())
    return np.array(idx[:n], dtype=int)


# ---------------------------------------------------------------- MO-PEO (constrained), as shipped --
def mopeo_rt(rt, N, FE, rng, alpha=1.5, beta=1.0, gamma=0.5, lam=0.8, zeta=0.9, eps=1e-6, theta=1e-3,
             NA=None, bs=0.1, zs=0.1, mode="delay", sparse=True, rspeed=0.1):
    """Bookkeeping note: `vtrace` (returned as a 4th value) is a running record of the minimum total
    violation in that generation's population, appended after computing quantities the update rule
    already needs; it consumes no random numbers and changes no line of the update itself, so this
    function is bit-identical to the version without it."""
    L, U, D = rt.L, rt.U, rt.n_beamlets; span = U - L; NA = NA or N
    W = np.maximum(weights(2, N), 1e-6); N = len(W); b = beta * bs * span; z = zeta * zs * span
    X = L + rng.random((N, D)) * span
    F = rt.objectives(X); Vt, _, _ = rt.violation(X); fe = N
    AX, AF, AV = trunc_archive_con(X, F, Vt, NA); Fp = None
    T = FE // N; t = 0; pr_i = np.zeros(N)
    vtrace = [(fe, float(Vt.min()))]
    while fe + N <= FE:
        zmin, zmax = AF.min(0), AF.max(0); r = np.maximum(zmax - zmin, 1e-12)
        nrm = lambda G: (G - zmin) / (r + eps); phi = lambda G, w: (w * nrm(G)).max(1)
        pc = phi(F, W); s = np.zeros(N, bool) if Fp is None else np.abs(pc - phi(Fp, W)) > theta
        if mode == "delay":
            pr_i[:] = max(0.0, (t / T - rspeed) / (1 - rspeed))
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
        Fp = F; X = Xn; F = rt.objectives(X); Vt, _, _ = rt.violation(X); fe += N; t += 1
        AX, AF, AV = trunc_archive_con(np.r_[AX, X], np.r_[AF, F], np.r_[AV, Vt], NA)
        vtrace.append((fe, float(Vt.min())))
    return AX, AF, AV, vtrace


# ---------------------------------------------------------------- NSGA-II (constrained), Deb 2002 --
def nsga2_rt(rt, N, FE, rng):
    L, U, D = rt.L, rt.U, rt.n_beamlets
    X = L + rng.random((N, D)) * (U - L)
    F = rt.objectives(X); Vt, _, _ = rt.violation(X); fe = N
    vtrace = [(fe, float(Vt.min()))]
    while fe + N <= FE:
        a, bb = rng.integers(0, N, N), rng.integers(0, N, N)
        # binary tournament under constrained-domination: feasible beats infeasible; among two
        # infeasible, lower violation wins; among two feasible, better rank then higher crowding.
        rank = np.full(N, np.inf)
        feas = np.where(Vt <= 0)[0]
        if len(feas):
            r = nd_sort(F[feas])
            rank[feas] = r
        cd = np.zeros(N)
        if len(feas):
            for rr in range(int(rank[feas].max()) + 1 if len(feas) else 0):
                m = feas[rank[feas] == rr]
                cd[m] = crowding(F[m])
        def wins(i, k):
            fi, fk = Vt[i] <= 0, Vt[k] <= 0
            if fi and not fk: return True
            if fk and not fi: return False
            if (not fi) and (not fk): return Vt[i] < Vt[k]
            if rank[i] != rank[k]: return rank[i] < rank[k]
            return cd[i] > cd[k]
        better = np.array([a[i] if wins(a[i], bb[i]) else bb[i] for i in range(N)])
        par = X[better]
        C = sbx_pm(par, L, U, rng); FC = rt.objectives(C); VC, _, _ = rt.violation(C); fe += N
        XX, FF, VV = np.r_[X, C], np.r_[F, FC], np.r_[Vt, VC]
        idx = cdom_rank_crowd_select(FF, VV, N)
        X, F, Vt = XX[idx], FF[idx], VV[idx]
        vtrace.append((fe, float(Vt.min())))
    feas = Vt <= 0
    if feas.any():
        m = nd_mask(F[feas])
        return X[feas][m], F[feas][m], Vt[feas][m], vtrace
    m = nd_mask_con(F, Vt)
    return X[m], F[m], Vt[m], vtrace


def _prune_con(X, F, V, N):
    """Constrained analogue of gde3.py's _prune: keep whole constrained-rank fronts (feasible-first)
    while they fit, trimming the least-crowded members of the last one; if fewer than N feasible
    members exist, keep all of them and fill the remainder with the lowest-violation infeasible ones."""
    feas = np.where(V <= 0)[0]
    if len(feas) >= N:
        rank = nd_sort(F[feas])
        idx = []
        for r in range(rank.max() + 1):
            fr = list(feas[rank == r])
            if len(idx) + len(fr) <= N:
                idx.extend(fr); continue
            while len(idx) + len(fr) > N:
                fr.pop(int(np.argmin(crowding(F[fr]))))
            idx.extend(fr)
            break
        return X[idx], F[idx], V[idx]
    infeas = np.where(V > 0)[0]
    order = infeas[np.argsort(V[infeas])]
    keep = list(feas) + list(order[:N - len(feas)])
    return X[keep], F[keep], V[keep]


# ---------------------------------------------------------------- MOEA/D (constrained), CDP --
def moead_rt(rt, N, FE, rng, T=20, delta=0.9, nr=2):
    """Constrained MOEA/D: the neighbourhood-replacement rule of algorithms.py's moead() (compare
    Tchebycheff scalar values) is replaced by Deb's constraint-dominance principle (CDP) -- feasible
    beats infeasible; between two infeasible neighbours the lower total violation wins; between two
    feasible neighbours the original scalar comparison decides -- everything else (neighbourhoods,
    SBX+PM, the ideal point) is unchanged from moead()."""
    L, U, D = rt.L, rt.U, rt.n_beamlets
    W = np.maximum(weights(2, N), 1e-6); N = len(W)
    B = np.argsort(((W[:, None] - W[None]) ** 2).sum(2), 1)[:, :T]
    X = L + rng.random((N, D)) * (U - L)
    F = rt.objectives(X); Vt, _, _ = rt.violation(X); fe = N
    zs = F.min(0)
    vtrace = [(fe, float(Vt.min()))]
    while fe + N <= FE:
        for i in rng.permutation(N):
            pool = B[i] if rng.random() < delta else np.arange(N)
            p = rng.choice(pool, 2, replace=False)
            c = sbx_pm(X[p], L, U, rng)[:1]; fc = rt.objectives(c)[0]
            vc, _, _ = rt.violation(c); vc = float(vc[0]); fe += 1
            zs = np.minimum(zs, fc)
            g_old = (W[pool] * np.abs(F[pool] - zs)).max(1)
            g_new = (W[pool] * np.abs(fc - zs)).max(1)
            feas_c = vc <= 0; feas_nb = Vt[pool] <= 0
            better = np.where(feas_c & feas_nb, g_new <= g_old,
                      np.where(feas_c & ~feas_nb, True,
                      np.where(~feas_c & feas_nb, False,
                               vc <= Vt[pool])))
            upd = pool[rng.permutation(np.where(better)[0])][:nr]
            X[upd] = c[0]; F[upd] = fc; Vt[upd] = vc
        vtrace.append((fe, float(Vt.min())))
    AX, AF, AV = trunc_archive_con(X, F, Vt, N)
    return AX, AF, AV, vtrace


# ---------------------------------------------------------------- MOPSO (constrained), CDP --
def mopso_rt(rt, N, FE, rng, w=0.4, c1=1.0, c2=1.0, ndiv=30, NA=None, bounce=0.5):
    """Constrained MOPSO: personal-best replacement and the archive filter of mopso.py's mopso() are
    replaced by constraint-dominance (feasible beats infeasible; infeasible-infeasible by violation;
    feasible-feasible by the original dominance/coin-flip rule); velocity update, bound handling and
    polynomial mutation are unchanged, and leader selection still runs mopso.py's own grid roulette
    (_leaders), now over the constrained-dominated archive."""
    L, U, D = rt.L, rt.U, rt.n_beamlets; span = U - L; NA = NA or N
    X = L + rng.random((N, D)) * span; V = np.zeros((N, D))
    F = rt.objectives(X); Vt, _, _ = rt.violation(X); fe = N
    PX, PF, PV = X.copy(), F.copy(), Vt.copy()
    AX, AF, AV = trunc_archive_con(X, F, Vt, NA)
    vtrace = [(fe, float(Vt.min()))]
    while fe + N <= FE:
        lead = _mopso_leaders(AX, AF, N, ndiv, rng)
        r1, r2 = rng.random((N, D)), rng.random((N, D))
        Vv = w * V + c1 * r1 * (PX - X) + c2 * r2 * (lead - X)
        Xn = X + Vv
        out = (Xn < L) | (Xn > U)
        Xn = np.clip(Xn, L, U)
        Vv = np.where(out, -bounce * Vv, Vv)
        Xn = _poly_mutation(Xn, L, U, rng, 1.0 / D, 20.0)
        X, V = Xn, Vv
        F = rt.objectives(X); Vt, _, _ = rt.violation(X); fe += N
        feas_new, feas_old = Vt <= 0, PV <= 0
        new_dom = (F <= PF).all(1) & (F < PF).any(1)
        old_dom = (PF <= F).all(1) & (PF < F).any(1)
        repl = np.where(feas_new & ~feas_old, True,
               np.where(~feas_new & feas_old, False,
               np.where(feas_new & feas_old, new_dom | (~new_dom & ~old_dom & (rng.random(N) < 0.5)),
                        Vt < PV)))
        PX[repl], PF[repl], PV[repl] = X[repl], F[repl], Vt[repl]
        AX, AF, AV = trunc_archive_con(np.r_[AX, X], np.r_[AF, F], np.r_[AV, Vt], NA)
        vtrace.append((fe, float(Vt.min())))
    return AX, AF, AV, vtrace


# ---------------------------------------------------------------- GDE3 (constrained), CDP --
def gde3_rt(rt, N, FE, rng, CRate=0.9, Fscale=0.5):
    """Constrained GDE3: the trial-vs-target selection and the final pruning of gde3.py's gde3() are
    replaced by constraint-dominance; DE/rand/1/bin variation is unchanged."""
    L, U, D = rt.L, rt.U, rt.n_beamlets; span = U - L
    eye2 = 2.0 * np.eye(N); ar = np.arange(N)
    X = L + rng.random((N, D)) * span
    F = rt.objectives(X); Vt, _, _ = rt.violation(X); fe = N
    vtrace = [(fe, float(Vt.min()))]
    while fe + N <= FE:
        r = np.argsort(rng.random((N, N)) + eye2, axis=1)[:, :3]
        Vm = X[r[:, 0]] + Fscale * (X[r[:, 1]] - X[r[:, 2]])
        cross = rng.random((N, D)) < CRate
        cross[ar, rng.integers(0, D, N)] = True
        Ut = np.clip(np.where(cross, Vm, X), L, U)
        Fu = rt.objectives(Ut); Vu, _, _ = rt.violation(Ut); fe += N
        feas_u, feas_x = Vu <= 0, Vt <= 0
        u_dom = (Fu <= F).all(1)
        x_dom = (F <= Fu).all(1) & (F < Fu).any(1)
        both = ~u_dom & ~x_dom
        u_wins = np.where(feas_u & ~feas_x, True,
                 np.where(~feas_u & feas_x, False,
                 np.where(feas_u & feas_x, u_dom, Vu <= Vt)))
        grow = feas_u & feas_x & both
        Xn = np.where(u_wins[:, None], Ut, X)
        Fn = np.where(u_wins[:, None], Fu, F)
        Vn = np.where(u_wins, Vu, Vt)
        if grow.any():
            Xn = np.r_[Xn, Ut[grow]]; Fn = np.r_[Fn, Fu[grow]]; Vn = np.r_[Vn, Vu[grow]]
        if len(Fn) > N:
            Xn, Fn, Vn = _prune_con(Xn, Fn, Vn, N)
        X, F, Vt = Xn, Fn, Vn
        vtrace.append((fe, float(Vt.min())))
    AX, AF, AV = trunc_archive_con(X, F, Vt, N)
    return AX, AF, AV, vtrace


# ---------------------------------------------------------------- MOGWO (constrained), CDP --
def _update_archive_con(AX, AF, AV, NA, ngrid, inflate, gamma, rng):
    m = nd_mask_con(AF, AV)
    AX, AF, AV = AX[m], AF[m], AV[m]
    _, u = np.unique(np.round(np.c_[AF, AV], 12), axis=0, return_index=True)
    u = np.sort(u)
    AX, AF, AV = AX[u], AF[u], AV[u]
    while len(AF) > NA:
        _, inv, cnt = np.unique(_cell_keys(AF, ngrid, inflate), return_inverse=True, return_counts=True)
        p = np.exp(gamma * (cnt - cnt.max()))
        cell = rng.choice(len(cnt), p=p / p.sum())
        drop = rng.choice(np.where(inv == cell)[0])
        keep = np.ones(len(AF), bool); keep[drop] = False
        AX, AF, AV = AX[keep], AF[keep], AV[keep]
    return AX, AF, AV


def mogwo_rt(rt, N, FE, rng, ngrid=10, inflate=0.1, beta=4.0, gamma=2.0, NA=None):
    """Constrained MOGWO: mogwo.py's mogwo() archive filter (nd_mask) is replaced by nd_mask_con;
    leader selection (_leaders_for_pack) and the position update are unchanged, now drawing from the
    constrained-dominated archive. ngrid=10 matches the M<=2 setting used for MOGWO elsewhere in this
    paper (Section 4)."""
    L, U, D = rt.L, rt.U, rt.n_beamlets; span = U - L; NA = NA or N
    X = L + rng.random((N, D)) * span
    F = rt.objectives(X); Vt, _, _ = rt.violation(X); fe = N
    AX, AF, AV = _update_archive_con(X, F, Vt, NA, ngrid, inflate, gamma, rng)
    T = max(FE // N, 1); t = 0
    vtrace = [(fe, float(Vt.min()))]
    while fe + N <= FE:
        a = 2.0 * (1.0 - t / T)
        lead = _leaders_for_pack(AF, N, ngrid, inflate, beta, rng)
        Xn = np.zeros((N, D))
        for k in range(3):
            P = AX[lead[:, k]]
            A = 2.0 * a * rng.random((N, D)) - a
            C = 2.0 * rng.random((N, D))
            Xn += P - A * np.abs(C * P - X)
        X = np.clip(Xn / 3.0, L, U)
        F = rt.objectives(X); Vt, _, _ = rt.violation(X); fe += N; t += 1
        AX, AF, AV = _update_archive_con(np.r_[AX, X], np.r_[AF, F], np.r_[AV, Vt], NA, ngrid, inflate, gamma, rng)
        vtrace.append((fe, float(Vt.min())))
    return AX, AF, AV, vtrace
