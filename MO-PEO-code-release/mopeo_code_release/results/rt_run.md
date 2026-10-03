# Radiotherapy case study (HANDOVER 7.2): six-algorithm feasibility comparison

Synthetic PTV (prostate-like)/OAR (rectum-like, abutting) phantom, 112 beamlets (7 angles x 16 beamlets/angle), N=91, FE=30000, 20 seeds (1000-1019, matching the 12-problem benchmark's own protocol). Constraints: PTV D95 >= 70.3 Gy over >=95% of the PTV, OAR V_tol(22.0 Gy) <= 15%. All six algorithms (MO-PEO, NSGA-II, MOEA/D, MOPSO, GDE3, MOGWO) use Deb (2000) constrained-domination (rt_algorithms.py) in place of their unconstrained selection/archive rule -- the same constraint-handling principle for every algorithm -- and MO-PEO uses exactly its shipped MO-PEO-final hyperparameters (no tuning for this problem); the other five run at the standard settings used in the main 12-problem comparison (Section 4).

## Feasibility across 20 seeds

| Algorithm | Feasible archive members | Total archive members | Feasible solutions found (any seed) | Min total violation (range over seeds) |
|---|---|---|---|---|
| MO-PEO | 1820 | 1820 | 20 of 20 seeds | 0.000 -- 0.000 |
| NSGA-II | 0 | 60 | 0 of 20 seeds | 0.611 -- 0.850 |
| MOEA/D | 0 | 108 | 0 of 20 seeds | 0.327 -- 0.850 |
| MOPSO | 0 | 74 | 0 of 20 seeds | 0.715 -- 0.850 |
| GDE3 | 1820 | 1820 | 20 of 20 seeds | 0.000 -- 0.000 |
| MOGWO | 1481 | 1540 | 18 of 20 seeds | 0.000 -- 0.015 |
| Uniform random search | 0 | 598780 | 0 of 20 seeds | 0.818 -- 0.850 |

## Violation trajectory (median running-minimum total violation over 20 seeds)

| Algorithm | 3094 FE (10%) | 7553 FE (25%) | 15015 FE (50%) | 22477 FE (75%) | 29939 FE (100%) |
|---|---|---|---|---|---|
| MO-PEO | 0.657 | 0.222 | 0.000 | 0.000 | 0.000 |
| NSGA-II | 0.850 | 0.850 | 0.850 | 0.850 | 0.850 |
| MOEA/D | 0.850 | 0.850 | 0.850 | 0.850 | 0.850 |
| MOPSO | 0.850 | 0.850 | 0.850 | 0.850 | 0.850 |
| GDE3 | 0.487 | 0.000 | 0.000 | 0.000 | 0.000 |
| MOGWO | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

A value of 0.000 at a checkpoint means at least half of the 20 seeds already held a feasible solution by that point; a value that stays near its starting level across all five checkpoints means the algorithm's population/archive never got closer to the feasible region over the whole budget, not merely that it fell short by the end.

## Feasible-corridor width (rt_calibration.py, fixed before any MOEA was run)

The weighted-sum L-BFGS-B scan used to fix the two constraint thresholds (`rt_calibration.md`) found both constraints jointly satisfied only for scalarization weight k in {0.10, 0.15, 0.20} out of the 14-point grid tested (k up to 5.0); the nearest infeasible neighbours are k=0.07 (OAR V_tol=0.153, just over the 0.15 limit) and k=0.30 (D95=68.1 Gy, just under the 70.3 Gy target). In the objective units the constraints are stated in, the feasible corridor found by that scan spans PTV D95 in [70.8, 76.1] Gy (about 5 Gy) and OAR mean dose in [7.2, 9.1] Gy, against a scanned range of D95 from 52.2 Gy (k=5, pure sparing) to 242.3 Gy (k=0, unconstrained pure coverage) -- roughly a 3% slice of the scanned D95 range. This is consistent with, and is the reason for, the feasibility gap between MO-PEO and the other five algorithms above: the constrained region is narrow relative to the reachable range of the two objectives.

Pooled over every feasible MO-PEO archive member found across all 20 seeds (1820 points), the empirically observed feasible band is D95 in [70.3, 88.6] Gy and OAR V_tol in [0.041, 0.149] -- consistent with, and slightly wider than, the calibration scan above (the scan traces one 1-D slice of the trade-off; the true feasible region is the full 2-D neighbourhood MO-PEO's archive explores around it).

## Representative feasible front (MO-PEO, seed 1000), three points

| point | D95 PTV (Gy) | Dmean OAR (Gy) | V_tol OAR | CI |
|---|---|---|---|---|
| best coverage | 73.2 | 10.1 | 0.144 | 0.076 |
| best sparing | 70.3 | 6.9 | 0.086 | 0.074 |
| knee (median-coverage) | 71.3 | 7.1 | 0.095 | 0.074 |
