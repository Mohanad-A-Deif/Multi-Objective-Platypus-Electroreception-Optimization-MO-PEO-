import numpy as np
from problems import nd_sort, crowding, select_by_rank_crowd, nd_mask, das_dennis

def weights(M, N):
    if M == 2:
        a = np.linspace(0, 1, N); return np.c_[a, 1-a]
    H = 1
    while len(das_dennis(M, H+1)) <= N: H += 1
    return das_dennis(M, H)

def trunc_archive(X, F, NA):
    m = nd_mask(F); X, F = X[m], F[m]
    _, u = np.unique(np.round(F, 12), axis=0, return_index=True); X, F = X[u], F[u]
    if len(F) > NA:
        cd = crowding(F); keep = np.argsort(-cd)[:NA]; X, F = X[keep], F[keep]
    return X, F

# ---------------- MO-PEO (design spec D1-D4) ----------------
def mopeo(prob, N, FE, rng, alpha=1.5, beta=1.0, gamma=0.5, lam=0.8, zeta=0.9,
          eps=1e-6, theta=1e-3, NA=None, bscale=1.0, zscale=1.0, normx=False, stats=None):
    f, L, U, M = prob[:4]; D = len(L); NA = NA or N; span = U-L
    W = np.maximum(weights(M, N), 1e-6); N = len(W)
    b = beta*bscale*span; z = zeta*zscale*span
    X = L + rng.random((N, D))*span; F = f(X); fe = N
    AX, AF = trunc_archive(X, F, NA)
    Fprev = None
    while fe + N <= FE:
        zmin, zmax = AF.min(0), AF.max(0); r = zmax-zmin
        r = np.where(r < 1e-12, np.maximum(F.max(0)-F.min(0), 1e-12), r)
        nrm = lambda G: (G-zmin)/(r+eps)
        phi = lambda G, w: (w*nrm(G)).max(1)
        phic = phi(F, W)
        s = np.zeros(N, bool) if Fprev is None else np.abs(phic-phi(Fprev, W)) > theta  # D3
        j = (np.arange(N) + rng.integers(1, N, N)) % N                                  # j != i
        dX = X[j]-X; dPhi = phi(F[j], W)-phic                                             # D1
        dXn = dX/span if normx else dX
        E = -alpha*(dPhi/(np.linalg.norm(dXn, axis=1)+eps))[:, None]*dXn*(span if normx else 1)
        Mt = b*(2*rng.random((N, D))-1)                                                   # Eq.7
        lead = AX[((W[:, None, :]*nrm(AF)[None]).max(2)).argmin(1)]                       # D2
        Xexp = X+E+Mt+gamma*(lead-X)
        Xepl = X+lam*(lead-X)+z*rng.standard_normal((N, D))
        Xn = np.clip(np.where(s[:, None], Xepl, Xexp), L, U)                              # Pi_Omega
        if stats is not None: stats.append(s.mean())
        Fprev = F; X = Xn; F = f(X); fe += N
        AX, AF = trunc_archive(np.r_[AX, X], np.r_[AF, F], NA)                            # D4
    return AF

# ---------------- variation operators ----------------
def sbx_pm(P, L, U, rng, pc=0.9, ec=5, em=20):
    n, D = P.shape; C = P.copy()
    for i in range(0, n-1, 2):
        if rng.random() > pc: continue
        a, bb = P[i], P[i+1]; u = rng.random(D)
        beta = np.where(u <= .5, (2*u)**(1/(ec+1)), (1/(2*(1-u)))**(1/(ec+1)))
        mask = rng.random(D) < 0.5
        c1 = .5*((1+beta)*a+(1-beta)*bb); c2 = .5*((1-beta)*a+(1+beta)*bb)
        C[i] = np.where(mask, c1, a); C[i+1] = np.where(mask, c2, bb)
    C = np.clip(C, L, U)   # bound children BEFORE polynomial mutation (else (1-d)**k -> NaN)
    pm = 1/D; mut = rng.random((n, D)) < pm
    u = rng.random((n, D)); d1 = (C-L)/(U-L); d2 = (U-C)/(U-L)
    mp = 1/(em+1)
    dq = np.where(u < .5, (2*u+(1-2*u)*(1-d1)**(em+1))**mp-1, 1-(2*(1-u)+2*(u-.5)*(1-d2)**(em+1))**mp)
    C = np.where(mut, C+dq*(U-L), C)
    return np.clip(C, L, U)

def nsga2(prob, N, FE, rng):
    f, L, U, M = prob[:4]; D = len(L)
    if M == 3: N = len(weights(M, N))
    X = L+rng.random((N, D))*(U-L); F = f(X); fe = N
    rank = nd_sort(F); cd = np.zeros(N)
    for r in range(rank.max()+1):
        idx = rank == r; cd[idx] = crowding(F[idx])
    while fe+N <= FE:
        a, bb = rng.integers(0, N, N), rng.integers(0, N, N)
        better = (rank[a] < rank[bb]) | ((rank[a] == rank[bb]) & (cd[a] > cd[bb]))
        par = X[np.where(better, a, bb)]
        C = sbx_pm(par, L, U, rng); FC = f(C); fe += N
        XX, FF = np.r_[X, C], np.r_[F, FC]
        idx, rk = select_by_rank_crowd(FF, N)
        X, F = XX[idx], FF[idx]; rank = nd_sort(F); cd = np.zeros(N)
        for r in range(rank.max()+1):
            m = rank == r; cd[m] = crowding(F[m])
    return F[nd_mask(F)]

def moead(prob, N, FE, rng, T=20, delta=0.9, nr=2):
    f, L, U, M = prob[:4]; D = len(L)
    W = np.maximum(weights(M, N), 1e-6); N = len(W)
    B = np.argsort(((W[:, None]-W[None])**2).sum(2), 1)[:, :T]
    X = L+rng.random((N, D))*(U-L); F = f(X); fe = N; zs = F.min(0)
    while fe+N <= FE:
        for i in rng.permutation(N):
            pool = B[i] if rng.random() < delta else np.arange(N)
            p = rng.choice(pool, 2, replace=False)
            c = sbx_pm(X[p], L, U, rng)[:1]; fc = f(c)[0]; fe += 1
            zs = np.minimum(zs, fc)
            g_old = (W[pool]*np.abs(F[pool]-zs)).max(1); g_new = (W[pool]*np.abs(fc-zs)).max(1)
            upd = pool[rng.permutation(np.where(g_new <= g_old)[0])][:nr]
            X[upd] = c[0]; F[upd] = fc
    return F[nd_mask(F)]
