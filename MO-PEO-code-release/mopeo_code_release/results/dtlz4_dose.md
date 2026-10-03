# DTLZ4 dose-response: perturbation density (fraction of D coordinates perturbed per agent)

Pre-declared protocol and acceptance rule: `results/dtlz4_dose_protocol.md`. Shipped configuration = density 1/D (reported). 40 calibration seeds (6000-6039). D per problem: DTLZ4=12, ZDT4=10, DTLZ1=7, DTLZ3=12, DTLZ6=12.

## DTLZ4 failure rate (IGD > 0.2) by density

| density | failed / 40 | median IGD |
|---|---|---|
| 1/D | 17 | 0.09473 |
| 0.1 | 17 | 0.09473 |
| 0.25 | 18 | 0.08676 |
| 0.5 | 14 | 0.08679 |
| 1.0 | 10 | 0.09265 |

## Median IGD on the four problems already known/suspected to be harmed by denser noise, and the ratio to density = 1/D

| problem | 1/D | 0.1 | 0.25 | 0.5 | 1.0 |
|---|---|---|---|---|---|
| ZDT4 | 0.02117 (1.00x) | 0.02117 (1.00x) | 0.8937 (42.22x) | 3.693 (174.45x) | 5.244 (247.71x) |
| DTLZ1 | 0.03819 (1.00x) | 0.03819 (1.00x) | 0.3755 (9.83x) | 2.277 (59.63x) | 5.195 (136.04x) |
| DTLZ3 | 4.449 (1.00x) | 4.449 (1.00x) | 43.99 (9.89x) | 132.4 (29.76x) | 156.1 (35.09x) |
| DTLZ6 | 0.009186 (1.00x) | 0.009186 (1.00x) | 0.01287 (1.40x) | 0.006162 (0.67x) | 0.005403 (0.59x) |

## Acceptance rule (pre-declared; see protocol file) applied to each density

| density | DTLZ4 failed < 10/40 | ZDT4 <= 1.5x | DTLZ1 <= 1.5x | DTLZ6 <= 1.5x | ADOPT |
|---|---|---|---|---|---|
| 1/D | False | True | True | True | no |
| 0.1 | False | True | True | True | no |
| 0.25 | False | False | False | True | no |
| 0.5 | False | False | False | True | no |
| 1.0 | False | False | False | True | no |


## Caveat on the grid

For every problem tested here D is between 7 and 12, so 0.1 * D is between 0.7 and 1.2 and rounds to k = 1
coordinate under `k = max(1, round(density * D))` -- the same k as density = 1/D. The "1/D" and "0.1" columns
above are therefore the same configuration for all five problems (identical results, as the table shows), not
two independent points. The grid as declared does not resolve anything between k = 1 and k = 3 (density 0.25 on
D = 12 gives k = 3; on D = 7, k = 2). This was not caught before running because the rule was declared in terms
of density, not of k, and D was not looked at first; it is reported here rather than corrected, since amending
the grid after seeing results would break the pre-declared protocol. The finding stands regardless: already at
k = 3 (density 0.25, the next point above the baseline for every problem here), ZDT4, DTLZ1 and DTLZ3 degrade by
10-42x while DTLZ4 does not improve (18 vs 17 failed); the acceptance rule would have failed here even with a
grid point at k = 2.

## Verdict

**No density on the grid satisfies all four conditions. Per the pre-declared protocol, none is adopted.** The DTLZ4 discovery failure is reported as a characterised limitation (mechanism analysis in `results/diag_dtlz4_mech.md`); MO-PEO-final (density = 1/D) remains shipped.
