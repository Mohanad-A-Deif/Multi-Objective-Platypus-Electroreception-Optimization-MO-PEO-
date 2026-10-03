import json, numpy as np
import os
def _p(f):
    return os.path.join("results", f) if os.path.exists(os.path.join("results", f)) else f

from scipy.stats import ranksums
r=json.load(open(_p("results_final.json")))
names=["ZDT1","ZDT2","ZDT3","ZDT4","ZDT6","DTLZ1","DTLZ2","DTLZ3","DTLZ4","DTLZ5","DTLZ6","DTLZ7"]
def fmt(m,s): return "%.3e (%.1e)"%(m,s)
def block(algs,metric,base="MO-PEO-final",better_is_low=True):
    L=[]; wl={a:[0,0,0] for a in algs if a!=base}; ranks=[]
    for n in names:
        b=r[n+"|"+base][metric]; cells=[]; means=[]
        for a in algs:
            v=r[n+"|"+a][metric]; means.append(np.mean(v)); c=fmt(np.mean(v),np.std(v))
            sg=""
            if a!=base:
                p=ranksums(b,v).pvalue
                worse=(np.mean(v)>np.mean(b)) if better_is_low else (np.mean(v)<np.mean(b))
                sg="$+$" if (p<0.05 and worse) else ("$-$" if p<0.05 else "$\\approx$")
                wl[a][0 if sg=="$+$" else (1 if sg=="$-$" else 2)]+=1
            cells.append((c,sg))
        best=int(np.argmin(means) if better_is_low else np.argmax(means))
        cells=[("\\textbf{%s}"%c if i==best else c)+((" "+sg) if sg else "") for i,(c,sg) in enumerate(cells)]
        ranks.append(np.argsort(np.argsort(means if better_is_low else [-m for m in means]))+1)
        L.append(n+" & "+" & ".join(cells)+" \\\\")
    ranks=np.array(ranks)
    L.append("\\midrule")
    L.append("Mean rank & "+" & ".join("%.2f"%ranks[:,i].mean() for i in range(len(algs)))+" \\\\")
    L.append("$+/-/\\approx$ & --- & "+" & ".join("%d/%d/%d"%tuple(wl[a]) for a in algs if a!="MO-PEO-final")+" \\\\")
    return "\n".join(L)

main=["MO-PEO-final","NSGA-II","MOEA/D"]
abl=["MO-PEO-final","A1-sparse","B2-delayOnly","A0-noSparse-noUnion"]
T1="""\\begin{table}[htbp]
\\centering
\\caption{Mean IGD (standard deviation in parentheses) over 20 independent runs. Best value per problem in bold. Symbols denote Wilcoxon rank-sum outcomes against MO-PEO at $\\alpha=0.05$: $+$ MO-PEO significantly better, $-$ significantly worse, $\\approx$ no significant difference.}
\\label{tab:igd}
\\begin{tabular}{lccc}
\\toprule
Problem & MO-PEO & NSGA-II & MOEA/D \\\\
\\midrule
%s
\\bottomrule
\\end{tabular}
\\end{table}"""%block(main,"IGD")
T2="""\\begin{table}[htbp]
\\centering
\\caption{Mean hypervolume (standard deviation in parentheses) over 20 independent runs, computed after normalisation with reference point $\\mathbf{1.1}$. Higher is better.}
\\label{tab:hv}
\\begin{tabular}{lccc}
\\toprule
Problem & MO-PEO & NSGA-II & MOEA/D \\\\
\\midrule
%s
\\bottomrule
\\end{tabular}
\\end{table}"""%block(main,"HV",better_is_low=False)
T3="""\\begin{table}[htbp]
\\centering
\\caption{Ablation study (mean IGD over 20 runs). A1 retains only the localised perturbation, B2 only the deferred-refinement schedule, and A0 neither.}
\\label{tab:ablation}
\\begin{tabular}{lcccc}
\\toprule
Problem & MO-PEO & A1 (sparse only) & B2 (schedule only) & A0 (neither) \\\\
\\midrule
%s
\\bottomrule
\\end{tabular}
\\end{table}"""%block(abl,"IGD")
tau=json.load(open(_p("tau.json"))); taus=[0.0,0.2,0.4,0.6,0.8,1.0]
rows=[]
for n in ["ZDT1","ZDT4","DTLZ1","DTLZ6"]:
    vals=[tau["%s|%.1f"%(n,t)][0] for t in taus]; b=int(np.argmin(vals))
    cs=["%.4f"%v for v in vals]; cs[b]="\\textbf{%s}"%cs[b]
    rows.append(n+" & "+" & ".join(cs)+" \\\\")
T4="""\\begin{table}[htbp]
\\centering
\\caption{Sensitivity of MO-PEO to the deferred-refinement threshold $\\tau$ (mean IGD over 8 runs). $\\tau=1$ disables the schedule.}
\\label{tab:tau}
\\begin{tabular}{lcccccc}
\\toprule
Problem & $\\tau=0.0$ & $0.2$ & $0.4$ & $0.6$ & $0.8$ & $1.0$ \\\\
\\midrule
%s
\\bottomrule
\\end{tabular}
\\end{table}"""%"\n".join(rows)
tm=json.load(open(_p("timing.json")))
rows=["%s & %.2f (%.2f) & %.2f (%.2f) & %.2f (%.2f) \\\\"%((n,)+tuple(x for a in ["MO-PEO","NSGA-II","MOEA/D"] for x in tm[n+"|"+a])) for n in ["ZDT1","DTLZ2"]]
T5="""\\begin{table}[htbp]
\\centering
\\caption{Mean wall-clock time in seconds (standard deviation) over 5 runs at the budgets of Section~\\ref{sec:setup}, single-threaded NumPy on one CPU core. Implementation effects dominate these figures; they are indicative only.}
\\label{tab:time}
\\begin{tabular}{lccc}
\\toprule
Problem & MO-PEO & NSGA-II & MOEA/D \\\\
\\midrule
%s
\\bottomrule
\\end{tabular}
\\end{table}"""%"\n".join(rows)
open("tables.tex","w").write("\n\n".join([T1,T2,T3,T4,T5]))
print(open("tables.tex").read()[:1200])
