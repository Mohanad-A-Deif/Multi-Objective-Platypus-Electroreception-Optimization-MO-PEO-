import numpy as np
from itertools import combinations

# ---------------- ZDT ----------------
def zdt(name, D):
    L = np.zeros(D); U = np.ones(D)
    if name == "ZDT4":
        L = np.r_[0.0, -5*np.ones(D-1)]; U = np.r_[1.0, 5*np.ones(D-1)]
    def f(X):
        x1 = X[:, 0]
        if name in ("ZDT1", "ZDT2", "ZDT3"):
            g = 1 + 9*X[:, 1:].sum(1)/(D-1)
        elif name == "ZDT4":
            r = X[:, 1:]
            g = 1 + 10*(D-1) + (r**2 - 10*np.cos(4*np.pi*r)).sum(1)
        elif name == "ZDT6":
            x1v = x1
            f1 = 1 - np.exp(-4*x1v)*np.sin(6*np.pi*x1v)**6
            g = 1 + 9*(X[:, 1:].sum(1)/(D-1))**0.25
            f2 = g*(1-(f1/g)**2)
            return np.c_[f1, f2]
        f1 = x1
        if name in ("ZDT1", "ZDT4"):
            h = 1 - np.sqrt(f1/g)
        elif name == "ZDT2":
            h = 1 - (f1/g)**2
        elif name == "ZDT3":
            h = 1 - np.sqrt(f1/g) - (f1/g)*np.sin(10*np.pi*f1)
        return np.c_[f1, g*h]
    return f, L, U, 2

def zdt_pf(name, n=10000):
    if name == "ZDT6":
        f1 = np.linspace(0.2807753191, 1, n); return np.c_[f1, 1-f1**2]
    f1 = np.linspace(0, 1, n)
    if name in ("ZDT1", "ZDT4"): return np.c_[f1, 1-np.sqrt(f1)]
    if name == "ZDT2": return np.c_[f1, 1-f1**2]
    if name == "ZDT3":
        P = np.c_[f1, 1-np.sqrt(f1)-f1*np.sin(10*np.pi*f1)]
        return P[nd_mask(P)]

# ---------------- DTLZ (M=3) ----------------
def dtlz(name, M=3):
    k = {"DTLZ1": 5, "DTLZ7": 20}.get(name, 10)
    D = M - 1 + k
    L = np.zeros(D); U = np.ones(D)
    def f(X):
        xm = X[:, M-1:]
        if name in ("DTLZ1", "DTLZ3"):
            g = 100*(k + ((xm-0.5)**2 - np.cos(20*np.pi*(xm-0.5))).sum(1))
        elif name in ("DTLZ2", "DTLZ4", "DTLZ5"):
            g = ((xm-0.5)**2).sum(1)
        elif name == "DTLZ6":
            g = (xm**0.1).sum(1)
        elif name == "DTLZ7":
            g = 1 + 9*xm.sum(1)/k
        if name == "DTLZ1":
            F = np.empty((X.shape[0], M))
            for m in range(M):
                v = 0.5*(1+g)*np.prod(X[:, :M-1-m], axis=1)
                if m > 0: v = v*(1-X[:, M-1-m])
                F[:, m] = v
            return F
        if name == "DTLZ7":
            F = np.empty((X.shape[0], M)); F[:, :M-1] = X[:, :M-1]
            h = M - (F[:, :M-1]/(1+g[:, None])*(1+np.sin(3*np.pi*F[:, :M-1]))).sum(1)
            F[:, M-1] = (1+g)*h; return F
        th = X[:, :M-1].copy()
        if name == "DTLZ4": th = th**100
        if name in ("DTLZ5", "DTLZ6"):
            t = (1+2*g[:, None]*th)/(2*(1+g[:, None]))
            t[:, 0] = th[:, 0]; th = t
        th = th*np.pi/2
        F = np.empty((X.shape[0], M))
        for m in range(M):
            v = (1+g)*np.prod(np.cos(th[:, :M-1-m]), axis=1)
            if m > 0: v = v*np.sin(th[:, M-1-m])
            F[:, m] = v
        return F
    return f, L, U, M, D

def dtlz_pf(name, M=3):
    if name in ("DTLZ5", "DTLZ6"):
        t = np.linspace(0, np.pi/2, 5000)
        return np.c_[np.cos(t)*np.cos(np.pi/4), np.cos(t)*np.sin(np.pi/4), np.sin(t)]
    if name == "DTLZ7":
        a = np.linspace(0, 1, 200); A, B = np.meshgrid(a, a)
        x = np.c_[A.ravel(), B.ravel()]
        h = 3 - (x/2*(1+np.sin(3*np.pi*x))).sum(1)
        P = np.c_[x, 2*h]; return P[nd_mask(P)]
    W = das_dennis(M, 99)
    if name == "DTLZ1": return 0.5*W
    return W/np.linalg.norm(W, axis=1, keepdims=True)

# ---------------- utilities ----------------
def das_dennis(M, H):
    out = []
    for c in combinations(range(H+M-1), M-1):
        prev = -1; v = []
        for ci in c: v.append(ci-prev-1); prev = ci
        v.append(H+M-1-prev-1); out.append(v)
    return np.array(out, float)/H

def dom_matrix(F):
    le = (F[:, None, :] <= F[None, :, :]).all(2)
    lt = (F[:, None, :] < F[None, :, :]).any(2)
    return le & lt   # [i,j] True if i dominates j

def nd_mask(F):
    if len(F) > 3000:  # chunked
        keep = np.ones(len(F), bool)
        for s in range(0, len(F), 1000):
            B = F[s:s+1000]
            d = ((F[:, None, :] <= B[None]).all(2) & (F[:, None, :] < B[None]).any(2)).any(0)
            keep[s:s+1000] = ~d
        return keep
    return ~dom_matrix(F).any(0)

def nd_sort(F):
    Dm = dom_matrix(F); cnt = Dm.sum(0); rank = np.full(len(F), -1)
    r = 0; cur = np.where(cnt == 0)[0]
    while len(cur):
        rank[cur] = r
        cnt = cnt - Dm[cur].sum(0); cnt[rank >= 0] = -1
        cur = np.where(cnt == 0)[0]; r += 1
    return rank

def crowding(F):
    n, M = F.shape; cd = np.zeros(n)
    if n <= 2: return np.full(n, np.inf)
    for m in range(M):
        o = np.argsort(F[:, m]); span = F[o[-1], m]-F[o[0], m]
        cd[o[0]] = cd[o[-1]] = np.inf
        if span > 0:
            cd[o[1:-1]] += (F[o[2:], m]-F[o[:-2], m])/span
    return cd

def select_by_rank_crowd(F, n):
    rank = nd_sort(F); idx = []
    for r in range(rank.max()+1):
        fr = np.where(rank == r)[0]
        if len(idx)+len(fr) <= n: idx.extend(fr)
        else:
            cd = crowding(F[fr]); idx.extend(fr[np.argsort(-cd)[:n-len(idx)]]); break
    return np.array(idx), rank
