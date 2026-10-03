# DTLZ4: where the interior of the front is lost in the update rule of MO-PEO (observation only)

MO-PEO as reported (sparse, delayed refinement, tau = 0.1), 40 runs on seeds 7000..7039, 328 iterations of 91 evaluations. Interior = x1, x2 in (0.9705, 0.99949) (both coordinates). Uniform random sampling puts a candidate in the interior with probability 8.410e-04 (25.1 interior candidates per run of 29848 evaluations).

Failure (final IGD > 0.2): 28 of 40 runs. Runs with at least one interior offspring during the run: 14 (failed among them: 2).

## 1. Are interior candidates generated and then projected away?

Candidates over all runs: 1193920. Interior before the projection to the box: 72408; interior after it: 72408; candidates made interior by the projection: 0; candidates made non-interior by the projection: 0. The clipping acts coordinate by coordinate and is the identity on the window (which lies inside the box), so it can neither remove nor create an interior candidate: 72408 of them exist before and after. The projection does act elsewhere: x1 leaves the box in 30.4% of candidates and x2 in 29.2%.

## 2. Realised interior candidates, by branch

| runs | offspring, exploration (s = 0) | interior | offspring, exploitation (s = 1) | interior |
|---|---|---|---|---|
| failed (n=28) | 433847 | 3 | 401897 | 28 |
| not failed (n=12) | 176120 | 4917 | 182056 | 67460 |

## 3. Probability that a candidate is interior, from the states visited (counterfactual draws, same state, separate random stream)

Rates per candidate, pooled over the iterations before the first realised interior offspring (failed runs: the whole run), with 95% bootstrap intervals over runs. Uniform sampling: 8.41e-04. Each row changes ONE term of the move and keeps the others; nothing here is a proposal.

| move | all branches: failed runs | ... rate / uniform | all branches: not-failed runs, before discovery | exploration branch, failed | exploitation branch, failed |
|---|---|---|---|---|---|
| reported move | 5.17e-06 [1.3e-06, 1.0e-05] | 0.01 | 5.00e-05 [2.2e-05, 1.1e-04] | 7.19e-06 | 2.99e-06 |
| no leader attraction (gamma = lambda = 0) | 7.33e-06 [2.5e-06, 1.4e-05] | 0.01 | 3.18e-05 [1.2e-05, 6.8e-05] | 8.47e-06 | 6.09e-06 |
| no directional-sensing term | 1.72e-06 [5.7e-07, 3.1e-06] | 0.00 | 3.91e-05 [1.4e-05, 9.0e-05] | 0.00e+00 | 3.58e-06 |
| no noise term | 3.13e-06 [4.5e-07, 6.8e-06] | 0.00 | 7.27e-06 [0.0e+00, 2.0e-05] | 6.01e-06 | 0.00e+00 |
| noise on every coordinate | 1.20e-05 [6.3e-06, 1.9e-05] | 0.01 | 1.26e-04 [6.4e-05, 2.6e-04] | 6.35e-06 | 1.82e-05 |
| random archive member as leader | 9.07e-06 [5.1e-06, 1.4e-05] | 0.01 | 3.72e-05 [1.2e-05, 8.4e-05] | 5.71e-06 | 1.27e-05 |

**Marginal against joint (reported move, failed runs).** Probability that a candidate has x1 in the window, x2 in the window, both, and the product of the marginals:

| branch | P(x1 in window) | P(x2 in window) | product | P(both) | joint / product |
|---|---|---|---|---|---|
| all | 0.234 | 0.179 | 4.19e-02 | 5.17e-06 | 0.000 |
| exploration | 0.055 | 0.036 | 1.99e-03 | 7.19e-06 | 0.004 |
| exploitation | 0.428 | 0.334 | 1.43e-01 | 2.99e-06 | 0.000 |
(For uniform sampling each marginal is 0.0290 and the joint equals the product.)

## 4. Expected number of interior candidates over a run (integrated hazard)

H = expected number of interior candidates before the first realised one (whole run if none), summed over iterations from the counterfactual draws; a well-calibrated hazard makes H at the first event about 1 (exponential with mean 1).

| move | median H, discovered runs (n=14) | median H, runs without discovery (n=26) | mean of exp(-H) over all runs (predicted fraction without discovery) |
|---|---|---|---|
| reported move | 0.17 | 0.02 | 0.88 |
| no leader attraction (gamma = lambda = 0) | 0.08 | 0.04 | 0.88 |
| no directional-sensing term | 0.11 | 0.00 | 0.93 |
| no noise term | 0.00 | 0.00 | 0.95 |
| noise on every coordinate | 0.40 | 0.15 | 0.75 |
| random archive member as leader | 0.07 | 0.12 | 0.83 |

Observed fraction of runs without any interior offspring: 0.65 (26 of 40). Uniform sampling would give H = 25.1 over a whole run.

## 5. The state the moves start from (mean over runs of the fraction of agents and leaders whose coordinate is in the window)

| run group | budget | agents x1 | agents x2 | leaders x1 | leaders x2 | agents both |
|---|---|---|---|---|---|---|
| failed | 2% | 0.119 | 0.149 | 0.128 | 0.156 | 0.000 |
| failed | 20% | 0.237 | 0.200 | 0.305 | 0.245 | 0.000 |
| failed | 100% | 0.263 | 0.177 | 0.317 | 0.229 | 0.000 |
| not failed | 2% | 0.091 | 0.248 | 0.114 | 0.287 | 0.021 |
| not failed | 20% | 0.317 | 0.311 | 0.396 | 0.344 | 0.124 |
| not failed | 100% | 0.356 | 0.370 | 0.432 | 0.406 | 0.263 |
