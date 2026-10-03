"""
MOPSO -- multi-objective particle swarm optimisation with an adaptive-grid archive
(after Coello Coello, Pulido & Salazar Lechuga, IEEE TEC 2004).

Interface identical to the other baselines in algorithms.py:

    mopso(prob, N, FE, rng) -> (k, M) array of non-dominated objective vectors

    prob : (f, L, U, M[, D])  f maps an (n, D) array to an (n, M) array
    N    : swarm size
    FE   : evaluation budget (every call to f is counted; the loop never exceeds it)
    rng  : numpy Generator, seeded by the caller

Components
----------
* velocity   v <- w v + c1 r1 (pbest - x) + c2 r2 (leader - x),  w = 0.4, c1 = c2 = 1.0
* bounds     a coordinate that leaves the box is clipped to it and its velocity is reversed and
             halved (bounce = 0.5). Full reversal (bounce = 1, the textbook rule) is unstable when the
             velocity exceeds the box width: on a 30-D sphere in [-5, 5] the mean |v| stayed at ~10 and
             the swarm never contracted (best value 23 against 5e-6 without reversal).
* pbest      replaced when the new position dominates it; replaced with probability 1/2 when the
             two are mutually non-dominated; kept otherwise
* archive    non-dominated, de-duplicated, size <= N_A (= N). When over capacity, members are
             removed at random from the most crowded cell of an adaptive grid whose range is
             recomputed from the archive each time
* leaders    roulette over occupied grid cells with weight 10 / (members in the cell), so sparse
             cells are favoured; the leader is a uniformly random member of the chosen cell
* mutation   Deb's polynomial mutation (eta_m = 20, probability 1/D per coordinate) on every
             particle at every iteration -- the same operator and rate as the NSGA-II and MOEA/D
             baselines. The alternative mutation="decay" (one coordinate redrawn uniformly with
             probability pm0 (1 - t/T)^mexp) is kept for reference; it reached IGD 0.12 on ZDT1
             against 0.038 for "poly", because its rate is negligible after the first third of the run.

Grid resolution (deviation from the usual "30 divisions", recorded in the validation notes)
    ndiv = ndiv2 (default 30, the standard setting) for M <= 2. For M >= 3 a 30-division grid has 27,000 cells for
    an archive of 91, so nearly every occupied cell holds a single member and the crowding-based
    removal degenerates to removing at random: the archive then drifts to one corner of the front
    (DTLZ2 IGD 0.94 against 0.076 for MOEA/D). The resolution is therefore set so that the cells
    spanned by the (M-1)-dimensional front number about N_A / 3:
        ndiv = round((N_A / 3) ** (1 / (M - 1)))        -> 6 for M = 3, N_A = 91
    (the same rule gives 33 for M = 2, N_A = 100, i.e. close to the standard 30, which is kept).

The evaluation budget follows the convention used by the other algorithms in this package:
N evaluations for the initial swarm, then N per iteration while fe + N <= FE.
"""
import numpy as np
from problems import nd_mask


def _cells(F, ndiv):
    """Adaptive-grid cell index (one integer key per row of F)."""
    lo, hi = F.min(0), F.max(0)
    span = np.maximum(hi - lo, 1e-12)
    idx = np.minimum(((F - lo) / span * ndiv).astype(np.int64), ndiv - 1)
    return idx @ (ndiv ** np.arange(F.shape[1], dtype=np.int64))


def _update_archive(AX, AF, NA, ndiv, rng):
    """Keep the non-dominated, de-duplicated members; thin the most crowded cell down to NA."""
    m = nd_mask(AF)
    AX, AF = AX[m], AF[m]
    _, u = np.unique(np.round(AF, 12), axis=0, return_index=True)
    u = np.sort(u)
    AX, AF = AX[u], AF[u]
    while len(AF) > NA:
        _, inv, cnt = np.unique(_cells(AF, ndiv), return_inverse=True, return_counts=True)
        drop = rng.choice(np.where(inv == cnt.argmax())[0])
        keep = np.ones(len(AF), bool)
        keep[drop] = False
        AX, AF = AX[keep], AF[keep]
    return AX, AF


def _leaders(AX, AF, n, ndiv, rng):
    """Roulette selection of one archive member per particle, favouring sparsely populated cells."""
    _, inv, cnt = np.unique(_cells(AF, ndiv), return_inverse=True, return_counts=True)
    p = 10.0 / cnt
    cell = rng.choice(len(cnt), size=n, p=p / p.sum())
    order = np.argsort(inv, kind="stable")
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    pick = order[start[cell] + np.minimum((rng.random(n) * cnt[cell]).astype(int), cnt[cell] - 1)]
    return AX[pick]


def _poly_mutation(X, L, U, rng, pm, eta):
    """Deb's polynomial mutation, per-coordinate probability pm (X must already lie in the box)."""
    span = U - L
    mut = rng.random(X.shape) < pm
    u = rng.random(X.shape)
    d1, d2 = (X - L) / span, (U - X) / span
    mp = 1.0 / (eta + 1.0)
    dq = np.where(u < 0.5, (2*u + (1-2*u) * (1-d1) ** (eta+1)) ** mp - 1,
                  1 - (2*(1-u) + 2*(u-0.5) * (1-d2) ** (eta+1)) ** mp)
    return np.clip(np.where(mut, X + dq * span, X), L, U)


def mopso(prob, N, FE, rng, w=0.4, c1=1.0, c2=1.0, ndiv=None, pm0=0.5, mexp=5.0, NA=None, info=None,
          mutation="poly", eta_m=20.0, bounce=0.5, ndiv2=30):
    f, L, U, M = prob[:4]
    D = len(L)
    span = U - L
    NA = NA or N
    if ndiv is None:
        ndiv = ndiv2 if M <= 2 else max(2, int(round((NA / 3.0) ** (1.0 / (M - 1)))))

    X = L + rng.random((N, D)) * span
    V = np.zeros((N, D))
    F = f(X)
    fe = N
    PX, PF = X.copy(), F.copy()
    AX, AF = _update_archive(X, F, NA, ndiv, rng)
    T = max(FE // N, 1)
    t = 0

    while fe + N <= FE:
        lead = _leaders(AX, AF, N, ndiv, rng)
        r1, r2 = rng.random((N, D)), rng.random((N, D))
        V = w * V + c1 * r1 * (PX - X) + c2 * r2 * (lead - X)
        Xn = X + V
        out = (Xn < L) | (Xn > U)
        Xn = np.clip(Xn, L, U)
        V = np.where(out, -bounce * V, V)

        # mutation
        if mutation == "poly":
            Xn = _poly_mutation(Xn, L, U, rng, 1.0 / D, eta_m)
        else:
            p_mut = pm0 * (1.0 - t / T) ** mexp
            rows = np.where(rng.random(N) < p_mut)[0]
            if len(rows):
                cols = rng.integers(0, D, len(rows))
                Xn[rows, cols] = L[cols] + rng.random(len(rows)) * span[cols]

        X = Xn
        F = f(X)
        fe += N
        t += 1

        # personal best
        new_dom = (F <= PF).all(1) & (F < PF).any(1)
        old_dom = (PF <= F).all(1) & (PF < F).any(1)
        repl = new_dom | (~new_dom & ~old_dom & (rng.random(N) < 0.5))
        PX[repl], PF[repl] = X[repl], F[repl]

        AX, AF = _update_archive(np.r_[AX, X], np.r_[AF, F], NA, ndiv, rng)

    if info is not None:
        info["fe"], info["AX"] = fe, AX
    return AF


# Configurations reported in the study (see probe_mopso.py and HANDOVER.md).
# DEFAULT is the function's own defaults (w = 0.4, c1 = c2 = 1.0, 30 grid divisions for M = 2).
# TUNED  = best of the coarse probe grid on ZDT4 + ZDT6 (non-protocol seeds 5000-5004); polynomial mutation is NOT tuned.
# TUNED_EXT = best of the extended robustness grid (found after seeing results, outside the pre-declared grid): NOT a reported configuration, robustness note in the supplementary text only.
TUNED = dict(w=0.2, c1=2.0, c2=2.0, ndiv2=6)
TUNED_EXT = dict(w=0.05, c1=2.0, c2=2.0, ndiv2=6)
