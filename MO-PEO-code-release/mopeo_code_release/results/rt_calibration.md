# Calibration of the two dose-volume constraint thresholds (`rt_problem.py`)

Approximate Pareto front by weighted-sum L-BFGS-B (`f1 + k*f2`, `k` = OAR objective weight); not the MOEA, just a fast way to see the achievable range before fixing the constraint thresholds. D95 target = 70.3 Gy, OAR dose-volume threshold = 22.0 Gy, OAR volume limit = 0.15.

| k (OAR weight) | D95(PTV) Gy | PTV feasible | Dmean(OAR) Gy | V_tol(OAR) | OAR feasible | BOTH feasible |
|---|---|---|---|---|---|---|
| 0.00 | 242.3 | True | 247.6 | 1.000 | False | False |
| 0.01 | 88.7 | True | 13.2 | 0.189 | False | False |
| 0.02 | 84.3 | True | 12.6 | 0.180 | False | False |
| 0.03 | 76.9 | True | 12.0 | 0.167 | False | False |
| 0.05 | 75.7 | True | 11.0 | 0.162 | False | False |
| 0.07 | 76.5 | True | 10.1 | 0.153 | False | False |
| 0.10 | 76.1 | True | 9.1 | 0.135 | True | True |
| 0.15 | 73.9 | True | 8.0 | 0.122 | True | True |
| 0.20 | 70.8 | True | 7.2 | 0.081 | True | True |
| 0.30 | 68.1 | False | 6.2 | 0.063 | True | False |
| 0.50 | 65.7 | False | 4.9 | 0.036 | True | False |
| 1.00 | 60.5 | False | 3.4 | 0.000 | True | False |
| 2.00 | 56.9 | False | 2.2 | 0.000 | True | False |
| 5.00 | 52.2 | False | 1.1 | 0.000 | True | False |

**Feasible corridor: k in [0.1, 0.15, 0.2].** Neither extreme (k=0, pure coverage; k>=1, pure sparing) is feasible: k=0 violates the OAR constraint (V_tol=1.00 > 0.15), k>=1 violates the PTV constraint (D95=52.2 < 70.3 Gy). This is the intended narrow trade-off, not a degenerate one where either objective alone already satisfies both constraints.
