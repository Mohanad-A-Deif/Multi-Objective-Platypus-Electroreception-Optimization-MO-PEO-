import numpy as np
from algorithms import weights, trunc_archive
def mopeo5(prob,N,FE,rng,alpha=1.5,beta=1.0,gamma=0.5,lam=0.8,zeta=0.9,eps=1e-6,theta=1e-3,NA=None,
           bs=0.1,zs=0.1,tau=0.4,kmode="sparse",dfrac=0.5,pmode="time"):
    f,L,U,M=prob[:4]; D=len(L); span=U-L; NA=NA or N
    W=np.maximum(weights(M,N),1e-6); N=len(W); b=beta*bs*span; z=zeta*zs*span
    X=L+rng.random((N,D))*span; F=f(X); fe=N; AX,AF=trunc_archive(X,F,NA); Fp=None
    T=FE//N; t=0
    prel_i=(np.arange(N)/(N-1))   # spatial heterogeneity: each agent keeps its own refine/explore bias
    while fe+N<=FE:
        p=max(0.0,(t/T-tau)/(1-tau))                 # deferred refinement schedule
        zmin,zmax=AF.min(0),AF.max(0); r=np.maximum(zmax-zmin,1e-12)
        nrm=lambda G:(G-zmin)/(r+eps); phi=lambda G,w:(w*nrm(G)).max(1)
        pc=phi(F,W); s=np.zeros(N,bool) if Fp is None else np.abs(pc-phi(Fp,W))>theta
        j=(np.arange(N)+rng.integers(1,N,N))%N; dXn=(X[j]-X)/span; dP=phi(F[j],W)-pc
        E=-alpha*(dP/(np.linalg.norm(dXn,axis=1)+eps))[:,None]*dXn*span
        lead=AX[((W[:,None,:]*nrm(AF)[None]).max(2)).argmin(1)]
        if kmode=="sparse": rate=0.0
        elif kmode=="dense": rate=1.0
        elif kmode=="ramp":  rate=p                      # sparse early -> dense late
        elif kmode=="ramp2": rate=p**2
        elif kmode=="switch": rate=1.0 if p>0 else 0.0
        if kmode=="split":
            rate=np.where(rng.random(N)<dfrac,(prel_i if pmode=="spatial" else p),0.0)[:,None]
        elif kmode=="cap":
            rate=min(p,0.5)
        K=rng.random((N,D))<rate; K[np.arange(N),rng.integers(0,D,N)]=True
        Mt=b*(2*rng.random((N,D))-1)*K
        pv=prel_i if pmode=="spatial" else p
        use=(rng.random(N)<pv)[:,None]
        Xi=np.where(use,np.abs(X-lead)*rng.standard_normal((N,D)),z*rng.standard_normal((N,D)))*K
        Xn=np.clip(np.where(s[:,None],X+lam*(lead-X)+Xi,X+E+Mt+gamma*(lead-X)),L,U)
        Fp=F; X=Xn; F=f(X); fe+=N; t+=1
        AX,AF=trunc_archive(np.r_[AX,X],np.r_[AF,F],NA)
    return AF
