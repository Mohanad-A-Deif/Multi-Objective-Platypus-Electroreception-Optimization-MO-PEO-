import numpy as np
from problems import nd_mask
def igd(A, P):
    d = np.sqrt(((P[:, None, :]-A[None, :, :])**2).sum(2)).min(1); return d.mean()
def hv2(F, ref):
    F = F[(F < ref).all(1)]
    if len(F) == 0: return 0.0
    F = F[nd_mask(F)]; F = F[np.argsort(F[:, 0])]
    h = 0.0; py = ref[1]
    for x, y in F: h += (ref[0]-x)*(py-y); py = y
    return h
def hv3(F, ref):
    F = F[(F < ref).all(1)]
    if len(F) == 0: return 0.0
    F = F[nd_mask(F)]; F = F[np.argsort(F[:, 2])]
    h = 0.0
    for i in range(len(F)):
        z_next = F[i+1, 2] if i+1 < len(F) else ref[2]
        h += hv2(F[:i+1, :2], ref[:2])*(z_next-F[i, 2])
    return h
def hv_norm(A, P):
    lo, hi = P.min(0), P.max(0); rng = np.where(hi-lo > 0, hi-lo, 1)
    An = (A-lo)/rng; ref = np.full(A.shape[1], 1.1)
    return hv2(An, ref) if A.shape[1] == 2 else hv3(An, ref)
