import numpy as np
from problems import nd_mask
from algorithms import weights, trunc_archive
def mopeo4(prob,N,FE,rng,alpha=1.5,beta=1.0,gamma=0.5,lam=0.8,zeta=0.9,eps=1e-6,theta=1e-3,NA=None,
           bs=0.1,zs=0.1,mode="fixed",sparse=True,prel=0.7,floor=0.01,rspeed=1.0,up=0.05,dn=0.05,lo=0.3,hi=0.9,
           density=None):
    # density (added 2026-09-23, DTLZ4 dose-response, results/dtlz4_dose_protocol.md): fraction of the D
    # coordinates perturbed per agent, k = max(1, round(density*D)); density=None (default) leaves the
    # original sparse/dense choice below untouched -- byte-for-byte the same code path, same rng draws --
    # so every previously stored key still reproduces. density=1/D reproduces the reported (sparse=True)
    # rate in expectation (not bit-identically: it costs one rng.random((N,D)) call instead of rng.integers).
    f,L,U,M=prob[:4]; D=len(L); span=U-L; NA=NA or N
    W=np.maximum(weights(M,N),1e-6); N=len(W); b=beta*bs*span; z=zeta*zs*span
    X=L+rng.random((N,D))*span; F=f(X); fe=N; AX,AF=trunc_archive(X,F,NA); Fp=None
    T=FE//N; t=0; pr_i=np.full(N,prel); succ=np.zeros(N)
    while fe+N<=FE:
        zmin,zmax=AF.min(0),AF.max(0); r=np.maximum(zmax-zmin,1e-12)
        nrm=lambda G:(G-zmin)/(r+eps); phi=lambda G,w:(w*nrm(G)).max(1)
        pc=phi(F,W); s=np.zeros(N,bool) if Fp is None else np.abs(pc-phi(Fp,W))>theta
        if mode=="ramp": pr_i[:]=min(1.0,t/(T*rspeed))
        elif mode=="pow": pr_i[:]=(t/T)**rspeed
        elif mode=="delay": pr_i[:]=max(0.0,(t/T-rspeed)/(1-rspeed))
        elif mode=="split": pr_i=(np.arange(N)%2).astype(float)
        j=(np.arange(N)+rng.integers(1,N,N))%N; dXn=(X[j]-X)/span; dP=phi(F[j],W)-pc
        E=-alpha*(dP/(np.linalg.norm(dXn,axis=1)+eps))[:,None]*dXn*span
        lead=AX[((W[:,None,:]*nrm(AF)[None]).max(2)).argmin(1)]
        if density is not None:
            k=max(1,min(D,round(density*D)))
            if k>=D:
                K=np.ones((N,D),bool)
            else:
                idx=np.argpartition(rng.random((N,D)),k,axis=1)[:,:k]
                K=np.zeros((N,D),bool); K[np.repeat(np.arange(N),k),idx.reshape(-1)]=True
        elif sparse:
            K=np.zeros((N,D),bool); K[np.arange(N),rng.integers(0,D,N)]=True
        else:
            K=np.ones((N,D),bool)
        Mt=b*(2*rng.random((N,D))-1)*K
        use=(rng.random(N)<pr_i)[:,None]
        if mode=="floor":
            Xi=(np.abs(X-lead)+floor*span)*rng.standard_normal((N,D))*K
        else:
            Xi=np.where(use,np.abs(X-lead)*rng.standard_normal((N,D)),z*rng.standard_normal((N,D)))*K
        Xn=np.clip(np.where(s[:,None],X+lam*(lead-X)+Xi,X+E+Mt+gamma*(lead-X)),L,U)
        Fp=F; X=Xn; F=f(X); fe+=N; t+=1
        ok=nd_mask(np.r_[AF,F])[len(AF):]
        if mode=="succ_b":
            pr_i=np.clip(np.where(ok,pr_i+up,pr_i-dn),lo,hi)
        elif mode=="succ":   # succeed -> refine (relative); fail -> explore (absolute)
            pr_i=np.clip(np.where(ok,pr_i+0.1,pr_i-0.1),0.0,1.0)
        elif mode=="succ_inv":
            pr_i=np.clip(np.where(ok,pr_i-0.1,pr_i+0.1),0.0,1.0)
        AX,AF=trunc_archive(np.r_[AX,X],np.r_[AF,F],NA)
    return AF
