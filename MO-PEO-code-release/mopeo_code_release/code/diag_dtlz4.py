#!/usr/bin/env python3
"""
diag_dtlz4.py -- why does MO-PEO fail in about half of the DTLZ4 runs?  (observation only, no modification)

DTLZ4 maps the position variables by theta_i = x_i^100 * pi/2, so for ANY g the point on the front is fixed by
    elevation = asin(f3/|f|) = (pi/2) u1,  azimuth = atan2(f2, f1) = (pi/2) u2,   with u1 = x1^100, u2 = x2^100.
Almost all of [0,1] for x1 (x1 < 0.97) maps to u1 < 0.05, i.e. to the f3 = 0 edge; the informative window is a few hundredths wide.
This script records, for every iteration, where agents, archive members, leaders and offspring lie in the (u1, u2) square,
how the archive covers the interior of the front, which operator branch produces interior offspring, and what truncation removes.

Traced copies (measurements never touch the random generator; --verify checks them bit for bit against the originals on seeds 1000..1002):
    mopeo_traced4  = tau_diag.mopeo4_traced (final configuration, tau selectable) + DTLZ4 measurements
    gde3_traced, nsga2_traced = the baselines, measurements only.
Seeds: 7000+ (never the protocol seeds, and not the calibration seeds 6000..6039).
"""
import argparse, os, pickle, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import dtlz, dtlz_pf, nd_mask, nd_sort, crowding
from algorithms import weights, trunc_archive, nsga2, sbx_pm
from algorithms import select_by_rank_crowd
from gde3 import gde3, _prune
from mopeo4 import mopeo4
from metrics import igd

LO, HI = 0.05, 0.95                       # 'interior' = both u in (LO, HI); LO corresponds to x = 0.9705
P_REF = dtlz_pf("DTLZ4")
_el = np.arcsin(np.clip(P_REF[:, 2], 0, 1)) / (np.pi / 2)                 # u1 of every reference point
_az = np.arctan2(P_REF[:, 1], P_REF[:, 0]) / (np.pi / 2)                  # u2
INT_REF = (_el > LO) & (_el < HI) & (_az > LO) & (_az < HI)


def uv(X):
    return X[:, 0] ** 100, X[:, 1] ** 100


def interior(X):
    u1, u2 = uv(X); return (u1 > LO) & (u1 < HI) & (u2 > LO) & (u2 < HI)


def zone(X):
    """0 = interior, 1 = low elevation (f3 ~ 0), 2 = pole (u1 >= HI), 3 = interior elevation but azimuth at an edge"""
    u1, u2 = uv(X); z = np.zeros(len(X), int)
    z[(u1 <= LO)] = 1; z[(u1 >= HI)] = 2; z[(u1 > LO) & (u1 < HI) & ((u2 <= LO) | (u2 >= HI))] = 3
    return z


def front_state(X, F, prefix):
    """position of a set of solutions (decision vectors X with objective vectors F) relative to the front, and coverage of the front"""
    z = zone(X); d = np.sqrt(((P_REF[:, None] - F[None]) ** 2).sum(2)).min(1)
    return {prefix + "n": len(F), prefix + "int": int((z == 0).sum()), prefix + "lowel": int((z == 1).sum()), prefix + "pole": int((z == 2).sum()),
            prefix + "azedge": int((z == 3).sum()), prefix + "igd": float(d.mean()), prefix + "igd_int": float(d[INT_REF].mean()),
            prefix + "igd_edge": float(d[~INT_REF].mean()), prefix + "gap": float(d.max()),
            prefix + "u1_bins": int(len(np.unique(np.floor(np.clip(uv(X)[0], 0, 0.9999) * 10))))}


def joint(X, prefix):
    """joint distribution of (x1, x2) over the zones dead (x <= 0.9705), window (0.9705 < x < 1), bound (x = 1); keys j<zone of x1><zone of x2>"""
    zn = lambda x: np.where(x >= 1.0, 2, np.where(x > 0.9705, 1, 0))
    a, b = zn(X[:, 0]), zn(X[:, 1])
    return {prefix + "j%d%d" % (i, j): float(((a == i) & (b == j)).mean()) for i in range(3) for j in range(3)}


def pop_state(X, prefix="ag_"):
    """distribution of a set of decision vectors in the informative window"""
    x1, x2 = X[:, 0], X[:, 1]
    return {**joint(X, prefix), prefix + "int": float(interior(X).mean()), prefix + "x1_alive": float((x1 > 0.9705).mean()), prefix + "x1_clip": float((x1 >= 1.0).mean()),
            prefix + "x2_alive": float((x2 > 0.9705).mean()), prefix + "x1_med": float(np.median(x1)), prefix + "x1_p90": float(np.percentile(x1, 90))}


# ------------------------------------------------------------------ MO-PEO ----
def mopeo_traced4(prob, N, FE, rng, tau=0.4, alpha=1.5, beta=1.0, gamma=0.5, lam=0.8, zeta=0.9, eps=1e-6, theta=1e-3, NA=None, bs=0.1, zs=0.1):
    f, L, U, M = prob[:4]; D = len(L); span = U - L; NA = NA or N
    W = np.maximum(weights(M, N), 1e-6); N = len(W); b = beta * bs * span; z = zeta * zs * span
    X = L + rng.random((N, D)) * span; F = f(X); fe = N; AX, AF = trunc_archive(X, F, NA); Fp = None
    T = FE // N; t = 0
    init = dict(**pop_state(X, "x0_"), **front_state(AX, AF, "a0_"))
    tr = []
    while fe + N <= FE:
        zmin, zmax = AF.min(0), AF.max(0); r = np.maximum(zmax - zmin, 1e-12)
        nrm = lambda G: (G - zmin) / (r + eps); phi = lambda G, w: (w * nrm(G)).max(1)
        pc = phi(F, W); s = np.zeros(N, bool) if Fp is None else np.abs(pc - phi(Fp, W)) > theta
        pr = 0.0 if tau >= 1.0 else max(0.0, (t / T - tau) / (1 - tau))
        j = (np.arange(N) + rng.integers(1, N, N)) % N; dXn = (X[j] - X) / span; dP = phi(F[j], W) - pc
        E = -alpha * (dP / (np.linalg.norm(dXn, axis=1) + eps))[:, None] * dXn * span
        lead_idx = ((W[:, None, :] * nrm(AF)[None]).max(2)).argmin(1)
        lead = AX[lead_idx]
        K = np.zeros((N, D), bool); K[np.arange(N), rng.integers(0, D, N)] = True
        Mt = b * (2 * rng.random((N, D)) - 1) * K
        use = (rng.random(N) < pr)[:, None]
        Xi = np.where(use, np.abs(X - lead) * rng.standard_normal((N, D)), z * rng.standard_normal((N, D))) * K
        Xn = np.clip(np.where(s[:, None], X + lam * (lead - X) + Xi, X + E + Mt + gamma * (lead - X)), L, U)
        rec = dict(t=t + 1, s=float(s.mean()), pr=pr, rmin=float(r.min()), nrm_max=float(nrm(F).max()), phi_med=float(np.median(pc)))
        rec.update(pop_state(X, "ag_"))                                   # agents before the move
        rec["lead_int"] = float(interior(AX[lead_idx]).mean()); rec["lead_n"] = int(len(np.unique(lead_idx)))
        rec.update(joint(AX[lead_idx], "ld_")); rec["lead_pole"] = float((zone(AX[lead_idx]) == 2).mean()); rec["lead_low"] = float((zone(AX[lead_idx]) == 1).mean())
        grp = {"exp": ~s, "rel": s & use[:, 0], "abs": s & ~use[:, 0]}
        Fp = F; X = Xn; F = f(X); fe += N; t += 1
        offint = interior(X)
        UX, UF = np.r_[AX, X], np.r_[AF, F]
        m = nd_mask(UF); PX, PF = UX[m], UF[m]
        _, u = np.unique(np.round(PF, 12), axis=0, return_index=True); PX, PF = PX[u], PF[u]
        oldF = AF
        AX, AF = trunc_archive(UX, UF, NA)
        kept = set(map(tuple, np.round(AF, 12)))
        isk = np.array([tuple(np.round(row, 12)) in kept for row in PF])
        newk = np.array([tuple(np.round(row, 12)) in kept for row in F])
        pz = zone(PX)
        rec.update(pre=len(PF), cut=int((~isk).sum()), rem_int=int(((pz == 0) & ~isk).sum()), pre_int=int((pz == 0).sum()),
                   rem_low=int(((pz == 1) & ~isk).sum()), rem_pole=int(((pz == 2) & ~isk).sum()))
        rec["off_int"] = int(offint.sum()); rec["off_int_kept"] = int((offint & newk).sum()); rec["entered"] = int(newk.sum())
        for g, mk in grp.items():
            rec["off_int_" + g] = int((offint & mk).sum()); rec["n_" + g] = int(mk.sum())
        rec.update(front_state(AX, AF, "ar_")); rec.update(joint(AX, "am_"))
        tr.append(rec)
    return AF, tr, init


# -------------------------------------------------------------------- GDE3 ----
def gde3_traced(prob, N, FE, rng, CRate=0.9, Fscale=0.5):
    f, L, U, M = prob[:4]; D = len(L); span = U - L
    eye2 = 2.0 * np.eye(N); ar = np.arange(N)
    X = L + rng.random((N, D)) * span; F = f(X); fe = N
    tr = []; init = dict(**pop_state(X, "x0_"))
    while fe + N <= FE:
        r = np.argsort(rng.random((N, N)) + eye2, axis=1)[:, :3]
        V = X[r[:, 0]] + Fscale * (X[r[:, 1]] - X[r[:, 2]])
        cross = rng.random((N, D)) < CRate
        cross[ar, rng.integers(0, D, N)] = True
        Ut = np.clip(np.where(cross, V, X), L, U)
        Fu = f(Ut); fe += N
        u_dom = (Fu <= F).all(1); x_dom = (F <= Fu).all(1) & (F < Fu).any(1); both = ~u_dom & ~x_dom
        Xn = np.where(u_dom[:, None], Ut, X); Fn = np.where(u_dom[:, None], Fu, F)
        if both.any():
            Xn, Fn = np.r_[Xn, Ut[both]], np.r_[Fn, Fu[both]]
        if len(Fn) > N:
            Xn, Fn = _prune(Xn, Fn, N)
        X, F = Xn, Fn
        m = nd_mask(F); rec = dict(t=len(tr) + 1); rec.update(pop_state(X, "ag_")); rec.update(front_state(X[m], F[m], "ar_")); tr.append(rec)
    m = nd_mask(F); Xf, Ff = X[m], F[m]
    _, u = np.unique(np.round(Ff, 12), axis=0, return_index=True); u = np.sort(u)
    return Ff[u], tr, init


# ------------------------------------------------------------------ NSGA-II ---
def nsga2_traced(prob, N, FE, rng):
    f, L, U, M = prob[:4]; D = len(L)
    if M == 3: N = len(weights(M, N))
    X = L + rng.random((N, D)) * (U - L); F = f(X); fe = N
    rank = nd_sort(F); cd = np.zeros(N)
    for r in range(rank.max() + 1):
        idx = rank == r; cd[idx] = crowding(F[idx])
    tr = []; init = dict(**pop_state(X, "x0_"))
    while fe + N <= FE:
        a, bb = rng.integers(0, N, N), rng.integers(0, N, N)
        better = (rank[a] < rank[bb]) | ((rank[a] == rank[bb]) & (cd[a] > cd[bb]))
        par = X[np.where(better, a, bb)]
        C = sbx_pm(par, L, U, rng); FC = f(C); fe += N
        XX, FF = np.r_[X, C], np.r_[F, FC]
        idx, rk = select_by_rank_crowd(FF, N)
        X, F = XX[idx], FF[idx]; rank = nd_sort(F); cd = np.zeros(N)
        for r in range(rank.max() + 1):
            m = rank == r; cd[m] = crowding(F[m])
        m = nd_mask(F); rec = dict(t=len(tr) + 1); rec.update(pop_state(X, "ag_")); rec.update(front_state(X[m], F[m], "ar_")); tr.append(rec)
    return F[nd_mask(F)], tr, init


ALG = {"MO-PEO": mopeo_traced4, "GDE3": gde3_traced, "NSGA-II": nsga2_traced}
ORIG = {"MO-PEO": lambda pr, N, FE, r: mopeo4(pr, N, FE, r, sparse=True, mode="delay", rspeed=0.4), "GDE3": gde3, "NSGA-II": nsga2}


def verify():
    pr = dtlz("DTLZ4"); ok = True
    for name in ALG:
        for s in (1000, 1001, 1002):
            A0 = ORIG[name](pr, 91, 30000, np.random.default_rng(s)); A1 = ALG[name](pr, 91, 30000, np.random.default_rng(s))[0]
            A0 = A0[np.lexsort(A0.T)]; A1 = A1[np.lexsort(A1.T)]
            same = A0.shape == A1.shape and np.array_equal(A0, A1); ok &= same
            print(name, s, "IDENTICAL" if same else "DIFFERENT", "IGD %.4f" % igd(A0, P_REF), flush=True)
    print("VERIFY", "OK" if ok else "FAILED"); return ok


def one(a):
    name, seed = a
    AF, tr, init = ALG[name](dtlz("DTLZ4"), 91, 30000, np.random.default_rng(seed))
    return (name, seed), dict(tr=tr, init=init, final_igd=float(igd(AF, P_REF)), size=len(AF))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=7000); ap.add_argument("--algs", nargs="+", default=list(ALG)); ap.add_argument("--out", default="../results/diag_dtlz4.pkl")
    ap.add_argument("--procs", type=int, default=2); a = ap.parse_args()
    if a.verify:
        sys.exit(0 if verify() else 1)
    assert a.seed0 >= 5000
    from multiprocessing import Pool
    jobs = [(n, a.seed0 + i) for i in range(a.runs) for n in a.algs]; out = {}; t0 = time.time()
    with Pool(a.procs) as p:
        for k, (key, v) in enumerate(p.imap_unordered(one, jobs)):
            out[key] = v
            if (k + 1) % 10 == 0: print("%d/%d %.0fs" % (k + 1, len(jobs), time.time() - t0), flush=True)
    pickle.dump(out, open(a.out, "wb")); print("saved", a.out, len(out))


if __name__ == "__main__":
    main()
