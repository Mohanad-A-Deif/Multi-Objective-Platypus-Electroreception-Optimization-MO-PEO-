"""
MOGWO -- multi-objective grey wolf optimiser (Mirjalili, Saremi, Mirjalili & Coelho, ESWA 2016),
box-constrained, M >= 1 objectives.

Interface identical to the other baselines in algorithms.py:

    mogwo(prob, N, FE, rng) -> (k, M) array of non-dominated objective vectors

    prob : (f, L, U, M[, D])  f maps an (n, D) array to an (n, M) array
    N    : number of wolves
    FE   : evaluation budget (every call to f is counted; the loop never exceeds it)
    rng  : numpy Generator, seeded by the caller

Components (settings of the original MATLAB implementation: 10 grids per objective, inflation 0.1,
leader-selection pressure beta = 4, deletion pressure gamma = 2, archive size N_A = N)
* archive     non-dominated, de-duplicated. The objective space spanned by the archive is inflated by 10 %
              on every side and cut into ngrid divisions per objective. When the archive exceeds N_A,
              members are removed one at a time: a cell is drawn with probability proportional to
              exp(gamma * members_in_cell) (crowded cells first) and a random member of it is deleted.
* leaders     drawn afresh FOR EVERY WOLF in every iteration, as in the original implementation (a first
              version of this file drew them once per iteration for the whole pack; that deviates from the
              original and was corrected, although it did not remove the ZDT2 behaviour described in HANDOVER.md).
              Delta is drawn from the archive: a cell is chosen with probability proportional to
              exp(-beta * members_in_cell) (sparse cells first) and a random member of that cell is taken.
              Beta is drawn the same way from the archive without Delta, alpha from the archive without
              Delta and Beta, so the three leaders are distinct whenever the archive has three members.
* position    a = 2 (1 - t / T);  for each leader  A = 2 a r1 - a,  C = 2 r2,  X_k = leader - A |C leader - x|;
              (A and C are drawn independently for each leader) x <- mean(X_alpha, X_beta, X_delta), clipped to the box. There is no personal best and no
              mutation, so nothing but the leaders carries information between iterations.

Grid resolution: ngrid = 10 (the original setting) for M <= 2. For M >= 3, 10^3 = 1000 cells for an archive of 91
is too fine (nearly every occupied cell holds one member, so crowding-based deletion is close to random and the
front is covered poorly: DTLZ2 IGD 0.40 with 10 divisions against 0.14 with 6); the same rule as in mopso.py is
used, ngrid = round((N_A / 3) ** (1 / (M - 1))) = 6. Pass ngrid explicitly to override.

Not verified against the reference source: the exact leader-selection weighting. exp(-beta * count) is used (the
weighting of the MOPSO code family this implementation follows); a power law count**(-beta) was tried on the
tuning seeds and behaves similarly (ZDT2 still collapses in some runs).

Budget convention as in the rest of the package: N evaluations for the initial wolves, then N per iteration
while fe + N <= FE; T = FE // N iterations sets the linear decrease of a.
"""
import numpy as np
from problems import nd_mask


def _cell_keys(F, ngrid, inflate):
    lo, hi = F.min(0), F.max(0)
    r = np.maximum(hi - lo, 1e-12)
    lo, hi = lo - inflate * r, hi + inflate * r
    idx = np.minimum(((F - lo) / (hi - lo) * ngrid).astype(np.int64), ngrid - 1)
    return idx @ (ngrid ** np.arange(F.shape[1], dtype=np.int64))


def _update_archive(AX, AF, NA, ngrid, inflate, gamma, rng):
    m = nd_mask(AF)
    AX, AF = AX[m], AF[m]
    _, u = np.unique(np.round(AF, 12), axis=0, return_index=True)
    u = np.sort(u)
    AX, AF = AX[u], AF[u]
    while len(AF) > NA:
        _, inv, cnt = np.unique(_cell_keys(AF, ngrid, inflate), return_inverse=True, return_counts=True)
        p = np.exp(gamma * (cnt - cnt.max()))
        cell = rng.choice(len(cnt), p=p / p.sum())
        drop = rng.choice(np.where(inv == cell)[0])
        keep = np.ones(len(AF), bool)
        keep[drop] = False
        AX, AF = AX[keep], AF[keep]
    return AX, AF


def _pick(pool, cid, beta, u1, u2):
    """Roulette over the occupied cells of `pool` (weight exp(-beta * count)), then a random member of the cell."""
    c = cid[pool]
    cnt = np.bincount(c)
    occ = np.nonzero(cnt)[0]
    cum = np.cumsum(np.exp(-beta * cnt[occ]))
    cell = occ[min(int(np.searchsorted(cum, u1 * cum[-1])), len(occ) - 1)]
    members = pool[c == cell]
    return members[min(int(u2 * len(members)), len(members) - 1)]


def _leaders_for_pack(AF, N, ngrid, inflate, beta, rng):
    """(N, 3) archive indices (delta, beta, alpha) for every wolf, following the original implementation."""
    n = len(AF)
    _, cid = np.unique(_cell_keys(AF, ngrid, inflate), return_inverse=True)
    allm = np.arange(n)
    U = rng.random((N, 6))
    out = np.empty((N, 3), dtype=np.int64)
    for i in range(N):
        d = _pick(allm, cid, beta, U[i, 0], U[i, 1])
        b = _pick(allm, cid, beta, U[i, 2], U[i, 3])
        al = _pick(allm, cid, beta, U[i, 4], U[i, 5])
        if n > 1:
            rep2 = allm[allm != d]
            b = _pick(rep2, cid, beta, U[i, 2], U[i, 3])
            if n > 2:
                al = _pick(rep2[rep2 != b], cid, beta, U[i, 4], U[i, 5])
        out[i] = (d, b, al)
    return out


def mogwo(prob, N, FE, rng, ngrid=None, inflate=0.1, beta=4.0, gamma=2.0, NA=None, info=None):
    f, L, U, M = prob[:4]
    D = len(L)
    span = U - L
    NA = NA or N
    if ngrid is None:
        ngrid = 10 if M <= 2 else max(2, int(round((NA / 3.0) ** (1.0 / (M - 1)))))

    X = L + rng.random((N, D)) * span
    F = f(X)
    fe = N
    AX, AF = _update_archive(X, F, NA, ngrid, inflate, gamma, rng)
    T = max(FE // N, 1)
    t = 0

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
        F = f(X)
        fe += N
        t += 1
        AX, AF = _update_archive(np.r_[AX, X], np.r_[AF, F], NA, ngrid, inflate, gamma, rng)

    if info is not None:
        info["fe"], info["AX"] = fe, AX
    return AF
