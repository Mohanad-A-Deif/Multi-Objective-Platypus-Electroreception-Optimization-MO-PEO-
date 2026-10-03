"""Calibration record for the two dose-volume constraint thresholds in rt_problem.py (HANDOVER 7.2).
Traces an approximate Pareto front by weighted-sum L-BFGS-B (bounds [0, U0], k = OAR objective
weight) and reports where each candidate OAR dose-volume threshold would bind along it. This is
choosing the DEFINITION of the problem's constraints (like choosing a benchmark function's own
parameters) before any MOEA is run on it, not tuning the algorithm; done once here, then fixed.
Usage: python rt_calibration.py  ->  results/rt_calibration.md, results/rt_calibration.png"""
import numpy as np
from scipy.optimize import minimize
from rt_problem import RTProblem, D95_TARGET_GY, DMAX_GY, OAR_VOL_LIMIT, RX_GY

rt = RTProblem()
Dp, Do = rt.D[rt.ptv_idx], rt.D[rt.oar_idx]
n_ptv, n_oar = Dp.shape[0], Do.shape[0]


def obj(w, k):
    dp, do = Dp @ w * RX_GY, Do @ w * RX_GY
    under = np.maximum(D95_TARGET_GY - dp, 0.0)
    f1, f2 = (under ** 2).mean(), (do ** 2).mean()
    g1 = (-2 * under / n_ptv)[None, :] @ Dp * RX_GY
    g2 = (2 * do / n_oar)[None, :] @ Do * RX_GY
    return f1 + k * f2, (g1 + k * g2).ravel()


if __name__ == "__main__":
    bounds = [(0, rt.U0)] * rt.n_beamlets
    w0 = np.full(rt.n_beamlets, rt.w0)
    ks = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
    rows = []
    for k in ks:
        res = minimize(obj, w0, args=(k,), jac=True, bounds=bounds, method="L-BFGS-B", options=dict(maxiter=400))
        w = res.x
        m = rt.metrics(w)
        dose = rt.D @ w * RX_GY
        v = (dose[rt.oar_idx] >= DMAX_GY).mean()
        d95_ok = m["D95_PTV"] >= D95_TARGET_GY
        oar_ok = v <= OAR_VOL_LIMIT
        rows.append(dict(k=k, D95=m["D95_PTV"], Dmean_OAR=m["Dmean_OAR"], Vtol=v,
                          d95_ok=d95_ok, oar_ok=oar_ok, feasible=d95_ok and oar_ok))

    md = ["# Calibration of the two dose-volume constraint thresholds (`rt_problem.py`)\n",
          "Approximate Pareto front by weighted-sum L-BFGS-B (`f1 + k*f2`, `k` = OAR objective weight); "
          "not the MOEA, just a fast way to see the achievable range before fixing the constraint "
          "thresholds. D95 target = %.1f Gy, OAR dose-volume threshold = %.1f Gy, OAR volume limit = %.2f.\n"
          % (D95_TARGET_GY, DMAX_GY, OAR_VOL_LIMIT),
          "| k (OAR weight) | D95(PTV) Gy | PTV feasible | Dmean(OAR) Gy | V_tol(OAR) | OAR feasible | BOTH feasible |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| %.2f | %.1f | %s | %.1f | %.3f | %s | %s |" %
                   (r["k"], r["D95"], r["d95_ok"], r["Dmean_OAR"], r["Vtol"], r["oar_ok"], r["feasible"]))
    both = [r["k"] for r in rows if r["feasible"]]
    md.append("\n**Feasible corridor: k in %s.** Neither extreme (k=0, pure coverage; k>=1, pure sparing) is "
               "feasible: k=0 violates the OAR constraint (V_tol=%.2f > %.2f), k>=1 violates the PTV constraint "
               "(D95=%.1f < %.1f Gy). This is the intended narrow trade-off, not a degenerate one where either "
               "objective alone already satisfies both constraints." %
               (str(both) if both else "NONE FOUND", rows[0]["Vtol"], OAR_VOL_LIMIT,
                rows[-1]["D95"], D95_TARGET_GY))
    open("../results/rt_calibration.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                          "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
                          "figure.dpi": 600, "savefig.dpi": 600})
    fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
    feas = [r["feasible"] for r in rows]
    D95s = [r["D95"] for r in rows]
    Vs = [r["Vtol"] for r in rows]
    ax.plot(D95s, Vs, color="0.4", lw=0.8, zorder=1)
    ax.scatter([d for d, f in zip(D95s, feas) if f], [v for v, f in zip(Vs, feas) if f],
               marker="o", facecolor="k", edgecolor="k", s=18, zorder=2, label="feasible")
    ax.scatter([d for d, f in zip(D95s, feas) if not f], [v for v, f in zip(Vs, feas) if not f],
               marker="x", color="0.5", s=18, zorder=2, label="infeasible")
    ax.axvline(D95_TARGET_GY, color="0.7", lw=0.6, ls="--")
    ax.axhline(OAR_VOL_LIMIT, color="0.7", lw=0.6, ls="--")
    ax.set_xlabel("PTV D95 (Gy)"); ax.set_ylabel("OAR V$_{tol}$"); ax.legend(loc="upper right", fontsize=6)
    fig.savefig("../results/rt_calibration.png", dpi=600)
    print("wrote results/rt_calibration.md, results/rt_calibration.png")
