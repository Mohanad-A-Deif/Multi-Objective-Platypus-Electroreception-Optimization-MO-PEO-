"""
GDE3 -- third evolution step of Generalized Differential Evolution
(after Kukkonen & Lampinen, CEC 2005), box-constrained, M >= 1 objectives.

Interface identical to the other baselines in algorithms.py:

    gde3(prob, N, FE, rng) -> (k, M) array of non-dominated objective vectors

    prob : (f, L, U, M[, D])  f maps an (n, D) array to an (n, M) array
    N    : population size
    FE   : evaluation budget (every call to f is counted; the loop never exceeds it)
    rng  : numpy Generator, seeded by the caller

Components
----------
* variation   DE/rand/1/bin with F = 0.5, CR = 0.9. For every target x_i three distinct indices
              r1, r2, r3 != i are drawn, v = x_r1 + F (x_r2 - x_r3), binomial crossover with one
              forced coordinate, and the trial is clipped to the box (a coordinate that leaves the
              box is set to the bound; nothing outside [L, U] is ever evaluated).
* selection   trial u_i against target x_i, objective space only (no constraints):
                 u_i weakly dominates x_i          -> u_i replaces x_i
                 x_i dominates u_i                 -> x_i is kept
                 neither dominates the other       -> both are kept (the population grows)
              All trials are built from the population of the previous generation (generational
              DE), so N evaluations are spent per generation.
* pruning     if the population exceeds N: non-dominated sorting, whole fronts are kept while they
              fit, and the members of the last front that does not fit are removed one at a time,
              always the one with the smallest crowding distance, recomputed after every removal
              (the pruning rule of the GDE3 paper).
* output      the non-dominated, de-duplicated members of the final population.

For M = 1 this is plain DE/rand/1/bin with greedy selection, which is what the sphere check exercises.
Budget convention as in the rest of the package: N evaluations for the initial population, then N per
generation while fe + N <= FE.
"""
import numpy as np
from problems import nd_mask, nd_sort, crowding


def _prune(X, F, N):
    """Reduce the population to N members: rank first, then repeated removal of the most crowded."""
    rank = nd_sort(F)
    idx = []
    for r in range(rank.max() + 1):
        fr = list(np.where(rank == r)[0])
        if len(idx) + len(fr) <= N:
            idx.extend(fr)
            continue
        while len(idx) + len(fr) > N:
            fr.pop(int(np.argmin(crowding(F[fr]))))
        idx.extend(fr)
        break
    return X[idx], F[idx]


def gde3(prob, N, FE, rng, CRate=0.9, Fscale=0.5, info=None):
    f, L, U, M = prob[:4]
    D = len(L)
    span = U - L
    eye2 = 2.0 * np.eye(N)
    ar = np.arange(N)

    X = L + rng.random((N, D)) * span
    F = f(X)
    fe = N

    while fe + N <= FE:
        # three distinct random indices, all different from i
        r = np.argsort(rng.random((N, N)) + eye2, axis=1)[:, :3]
        V = X[r[:, 0]] + Fscale * (X[r[:, 1]] - X[r[:, 2]])
        cross = rng.random((N, D)) < CRate
        cross[ar, rng.integers(0, D, N)] = True
        Ut = np.clip(np.where(cross, V, X), L, U)
        Fu = f(Ut)
        fe += N

        u_dom = (Fu <= F).all(1)                                  # trial weakly dominates target
        x_dom = (F <= Fu).all(1) & (F < Fu).any(1)                # target strictly dominates trial
        both = ~u_dom & ~x_dom                                    # mutually non-dominated
        Xn = np.where(u_dom[:, None], Ut, X)
        Fn = np.where(u_dom[:, None], Fu, F)
        if both.any():
            Xn, Fn = np.r_[Xn, Ut[both]], np.r_[Fn, Fu[both]]
        if len(Fn) > N:
            Xn, Fn = _prune(Xn, Fn, N)
        X, F = Xn, Fn

    m = nd_mask(F)
    X, F = X[m], F[m]
    _, u = np.unique(np.round(F, 12), axis=0, return_index=True)
    u = np.sort(u)
    X, F = X[u], F[u]
    if info is not None:
        info["fe"], info["AX"] = fe, X
    return F
