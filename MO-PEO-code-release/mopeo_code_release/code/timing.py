import numpy as np, time, json
from problems import *; from algorithms import nsga2, moead
from mopeo4 import mopeo4
FIN=dict(sparse=True,mode="delay",rspeed=0.1)
A={"MO-PEO":lambda p,N,FE,r: mopeo4(p,N,FE,r,**FIN),"NSGA-II":nsga2,"MOEA/D":moead}
out={}
for n,(pr,N,FE) in {"ZDT1":(zdt("ZDT1",30),100,25000),"DTLZ2":(dtlz("DTLZ2"),91,30000)}.items():
    for a,fn in A.items():
        ts=[]
        for s in range(5):
            t=time.perf_counter(); fn(pr,N,FE,np.random.default_rng(s)); ts.append(time.perf_counter()-t)
        out[n+"|"+a]=[float(np.mean(ts)),float(np.std(ts))]
        print(n,a,"%.2f +- %.2f s"%(np.mean(ts),np.std(ts)),flush=True)
json.dump(out,open("timing.json","w"))
