# DTLZ4 diagnosis: instrumented runs, 40 seeds (7000..7039), 328 iterations of N = 91 evaluations

Position on the front is fixed by u1 = x1^100 (elevation) and u2 = x2^100 (azimuth); 'interior' = 0.05 < u1, u2 < 0.95, i.e. x1, x2 > 0.9705 and not at the bound. Failure = final IGD > 0.2 (the final IGD is bimodal: see below).

## Failure rates and the bimodality of the final IGD

| algorithm | failed runs | median IGD of failed runs | median IGD of the others | largest IGD of the others | smallest IGD of the failed |
|---|---|---|---|---|---|
| MO-PEO | 26/40 | 0.2577 | 0.0851 | 0.0952 | 0.2194 |
| GDE3 | 0/40 | - | 0.0827 | 0.0929 | - |
| NSGA-II | 12/40 | 0.2566 | 0.0825 | 0.0887 | 0.2565 |

Fisher exact test, failure counts MO-PEO vs NSGA-II: p = 0.003; MO-PEO vs GDE3: p = 0.0000.

## Composition of the final archive (medians over runs; MO-PEO, failed vs not failed; the baselines' non-dominated set)

| set | n runs | size | interior members | low-elevation (f3 ~ 0) | pole | interior elevation, azimuth at an edge | IGD | IGD on interior part of the front | IGD on the edge part |
|---|---|---|---|---|---|---|---|---|---|
| MO-PEO, failed | 26 | 91 | 0 | 56 | 3 | 32 | 0.2577 | 0.3057 | 0.1296 |
| MO-PEO, not failed | 14 | 91 | 65 | 10 | 5 | 10 | 0.0851 | 0.0833 | 0.0874 |
| GDE3, not failed | 40 | 91 | 70 | 9 | 3 | 8 | 0.0827 | 0.0800 | 0.0891 |
| NSGA-II, failed | 12 | 91 | 0 | 48 | 3 | 42 | 0.2566 | 0.3048 | 0.1281 |
| NSGA-II, not failed | 28 | 91 | 69 | 9 | 3 | 9 | 0.0825 | 0.0817 | 0.0848 |

## Archive / non-dominated set over the run (medians over runs)

**Interior members of the archive (of 91)**

| run group | 2% | 5% | 10% | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|---|---|---|
| MO-PEO, failed (n=26) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| MO-PEO, not failed (n=14) | 0 | 0 | 0 | 6 | 58 | 62 | 64 | 65 |
| GDE3 (n=40) | 0 | 0 | 2 | 71 | 70 | 72 | 71 | 70 |
| NSGA-II, failed (n=12) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| NSGA-II, not failed (n=28) | 0 | 0 | 2 | 56 | 68 | 68 | 69 | 69 |

**IGD restricted to the interior part of the front (0% of the reference points)**

| run group | 2% | 5% | 10% | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|---|---|---|
| MO-PEO, failed (n=26) | 0.637 | 0.521 | 0.334 | 0.310 | 0.307 | 0.306 | 0.306 | 0.306 |
| MO-PEO, not failed (n=14) | 0.640 | 0.459 | 0.304 | 0.259 | 0.085 | 0.084 | 0.084 | 0.083 |
| GDE3 (n=40) | 0.613 | 0.454 | 0.238 | 0.092 | 0.083 | 0.080 | 0.080 | 0.080 |
| NSGA-II, failed (n=12) | 0.641 | 0.522 | 0.382 | 0.318 | 0.305 | 0.305 | 0.305 | 0.305 |
| NSGA-II, not failed (n=28) | 0.676 | 0.518 | 0.336 | 0.145 | 0.089 | 0.083 | 0.081 | 0.082 |

**Archive size**

| run group | 2% | 5% | 10% | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|---|---|---|
| MO-PEO, failed (n=26) | 91 | 91 | 91 | 91 | 91 | 91 | 91 | 91 |
| MO-PEO, not failed (n=14) | 88 | 91 | 91 | 91 | 91 | 91 | 91 | 91 |
| GDE3 (n=40) | 16 | 17 | 58 | 91 | 91 | 91 | 91 | 91 |
| NSGA-II, failed (n=12) | 44 | 62 | 91 | 91 | 91 | 91 | 91 | 91 |
| NSGA-II, not failed (n=28) | 58 | 70 | 91 | 91 | 91 | 91 | 91 | 91 |

## The population: where do the agents stand in the informative window? (agents before the move; medians over runs)

**Fraction of agents with x1 > 0.9705 (elevation above 4.5 degrees)**

| run group | 2% | 5% | 10% | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|---|---|---|
| MO-PEO, failed (n=26) | 0.247 | 0.308 | 0.379 | 0.401 | 0.445 | 0.473 | 0.478 | 0.473 |
| MO-PEO, not failed (n=14) | 0.269 | 0.308 | 0.451 | 0.560 | 0.599 | 0.604 | 0.599 | 0.621 |
| GDE3 (n=40) | 0.236 | 0.269 | 0.505 | 0.901 | 0.901 | 0.901 | 0.907 | 0.901 |
| NSGA-II, failed (n=12) | 0.137 | 0.280 | 0.473 | 0.505 | 0.484 | 0.467 | 0.462 | 0.478 |
| NSGA-II, not failed (n=28) | 0.170 | 0.335 | 0.648 | 0.824 | 0.901 | 0.901 | 0.901 | 0.901 |

**Fraction of agents at x1 = 1 (bound)**

| run group | 2% | 5% | 10% | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|---|---|---|
| MO-PEO, failed (n=26) | 0.049 | 0.143 | 0.154 | 0.170 | 0.176 | 0.187 | 0.176 | 0.192 |
| MO-PEO, not failed (n=14) | 0.027 | 0.137 | 0.170 | 0.187 | 0.220 | 0.220 | 0.203 | 0.214 |
| GDE3 (n=40) | 0.209 | 0.209 | 0.143 | 0.044 | 0.033 | 0.033 | 0.033 | 0.033 |
| NSGA-II, failed (n=12) | 0.082 | 0.060 | 0.022 | 0.011 | 0.011 | 0.016 | 0.011 | 0.011 |
| NSGA-II, not failed (n=28) | 0.082 | 0.088 | 0.049 | 0.044 | 0.033 | 0.033 | 0.033 | 0.033 |

**Fraction of agents in the interior of the front**

| run group | 2% | 5% | 10% | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|---|---|---|
| MO-PEO, failed (n=26) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| MO-PEO, not failed (n=14) | 0.000 | 0.000 | 0.000 | 0.060 | 0.236 | 0.269 | 0.264 | 0.269 |
| GDE3 (n=40) | 0.000 | 0.000 | 0.033 | 0.780 | 0.769 | 0.791 | 0.780 | 0.775 |
| NSGA-II, failed (n=12) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| NSGA-II, not failed (n=28) | 0.000 | 0.000 | 0.022 | 0.621 | 0.747 | 0.747 | 0.758 | 0.758 |

## Joint occupancy of (x1, x2): the interior is the cell 'window, window'
Interior of the front = both x1 and x2 in the window (0.9705, 1). A point at the bound (x = 1) is on an edge of the octant (elevation or azimuth pi/2); a point in the dead zone (x <= 0.9705) has u = x^100 < 0.05, i.e. sits within 4.5 degrees of the corresponding edge.

**Agents (before the move)** (mean over runs of the fraction of members in each (x1 zone, x2 zone) cell; cells with less than 0.5% everywhere are omitted)

| run group | budget | x1 dead, x2 dead | x1 dead, x2 win | x1 dead, x2 bound | x1 win, x2 dead | x1 win, x2 win | x1 win, x2 bound | x1 bound, x2 dead | x1 bound, x2 win | x1 bound, x2 bound |
|---|---|---|---|---|---|---|---|---|---|---|
| MO-PEO, failed (n=26) | 2% | 0.489 | 0.232 | 0.044 | 0.189 | 0.000 | 0.000 | 0.045 | 0.000 | 0.001 |
| MO-PEO, failed (n=26) | 20% | 0.216 | 0.206 | 0.157 | 0.256 | 0.000 | 0.000 | 0.164 | 0.000 | 0.000 |
| MO-PEO, failed (n=26) | 100% | 0.178 | 0.181 | 0.160 | 0.292 | 0.000 | 0.000 | 0.189 | 0.000 | 0.000 |
| MO-PEO, not failed (n=14) | 2% | 0.432 | 0.301 | 0.051 | 0.147 | 0.027 | 0.002 | 0.032 | 0.003 | 0.004 |
| MO-PEO, not failed (n=14) | 20% | 0.165 | 0.158 | 0.125 | 0.208 | 0.119 | 0.027 | 0.105 | 0.033 | 0.060 |
| MO-PEO, not failed (n=14) | 100% | 0.156 | 0.089 | 0.134 | 0.060 | 0.301 | 0.030 | 0.088 | 0.028 | 0.113 |
| GDE3 (n=40) | 2% | 0.555 | 0.032 | 0.159 | 0.032 | 0.001 | 0.004 | 0.170 | 0.005 | 0.043 |
| GDE3 (n=40) | 20% | 0.016 | 0.082 | 0.012 | 0.058 | 0.776 | 0.017 | 0.011 | 0.019 | 0.011 |
| GDE3 (n=40) | 100% | 0.011 | 0.076 | 0.011 | 0.052 | 0.808 | 0.004 | 0.008 | 0.018 | 0.011 |
| NSGA-II, failed (n=12) | 2% | 0.607 | 0.127 | 0.096 | 0.095 | 0.000 | 0.000 | 0.074 | 0.000 | 0.000 |
| NSGA-II, failed (n=12) | 20% | 0.042 | 0.367 | 0.016 | 0.559 | 0.000 | 0.000 | 0.016 | 0.000 | 0.000 |
| NSGA-II, failed (n=12) | 100% | 0.054 | 0.445 | 0.014 | 0.473 | 0.000 | 0.000 | 0.015 | 0.000 | 0.000 |
| NSGA-II, not failed (n=28) | 2% | 0.568 | 0.155 | 0.077 | 0.091 | 0.005 | 0.008 | 0.079 | 0.009 | 0.008 |
| NSGA-II, not failed (n=28) | 20% | 0.023 | 0.247 | 0.016 | 0.163 | 0.492 | 0.020 | 0.014 | 0.016 | 0.009 |
| NSGA-II, not failed (n=28) | 100% | 0.013 | 0.076 | 0.012 | 0.063 | 0.798 | 0.004 | 0.013 | 0.011 | 0.011 |

**Leaders assigned to the agents (MO-PEO only)** (mean over runs of the fraction of members in each (x1 zone, x2 zone) cell; cells with less than 0.5% everywhere are omitted)

| run group | budget | x1 dead, x2 dead | x1 dead, x2 win | x1 dead, x2 bound | x1 win, x2 dead | x1 win, x2 win | x1 win, x2 bound | x1 bound, x2 dead | x1 bound, x2 win | x1 bound, x2 bound |
|---|---|---|---|---|---|---|---|---|---|---|
| MO-PEO, failed (n=26) | 2% | 0.421 | 0.192 | 0.115 | 0.155 | 0.000 | 0.000 | 0.117 | 0.000 | 0.000 |
| MO-PEO, failed (n=26) | 20% | 0.174 | 0.247 | 0.133 | 0.315 | 0.000 | 0.000 | 0.131 | 0.000 | 0.000 |
| MO-PEO, failed (n=26) | 100% | 0.156 | 0.235 | 0.119 | 0.358 | 0.000 | 0.000 | 0.132 | 0.000 | 0.000 |
| MO-PEO, not failed (n=14) | 2% | 0.377 | 0.270 | 0.119 | 0.119 | 0.024 | 0.009 | 0.074 | 0.009 | 0.000 |
| MO-PEO, not failed (n=14) | 20% | 0.130 | 0.166 | 0.116 | 0.243 | 0.159 | 0.016 | 0.073 | 0.000 | 0.096 |
| MO-PEO, not failed (n=14) | 100% | 0.141 | 0.069 | 0.144 | 0.057 | 0.363 | 0.010 | 0.074 | 0.000 | 0.141 |

**Archive members (MO-PEO only)** (mean over runs of the fraction of members in each (x1 zone, x2 zone) cell; cells with less than 0.5% everywhere are omitted)

| run group | budget | x1 dead, x2 dead | x1 dead, x2 win | x1 dead, x2 bound | x1 win, x2 dead | x1 win, x2 win | x1 win, x2 bound | x1 bound, x2 dead | x1 bound, x2 win | x1 bound, x2 bound |
|---|---|---|---|---|---|---|---|---|---|---|
| MO-PEO, failed (n=26) | 2% | 0.453 | 0.252 | 0.054 | 0.195 | 0.000 | 0.000 | 0.044 | 0.000 | 0.002 |
| MO-PEO, failed (n=26) | 20% | 0.190 | 0.399 | 0.019 | 0.375 | 0.000 | 0.000 | 0.011 | 0.000 | 0.005 |
| MO-PEO, failed (n=26) | 100% | 0.131 | 0.492 | 0.011 | 0.344 | 0.000 | 0.000 | 0.012 | 0.001 | 0.008 |
| MO-PEO, not failed (n=14) | 2% | 0.402 | 0.336 | 0.059 | 0.126 | 0.022 | 0.006 | 0.040 | 0.006 | 0.003 |
| MO-PEO, not failed (n=14) | 20% | 0.072 | 0.308 | 0.013 | 0.256 | 0.296 | 0.007 | 0.034 | 0.004 | 0.010 |
| MO-PEO, not failed (n=14) | 100% | 0.025 | 0.086 | 0.011 | 0.046 | 0.766 | 0.011 | 0.031 | 0.013 | 0.011 |

## MO-PEO mechanics: leaders, normalisation, phase switching, truncation (medians over runs)

**Fraction of agents whose leader is an interior member**

| run group | 2% | 5% | 10% | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|---|---|---|
| MO-PEO, failed (n=26) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| MO-PEO, not failed (n=14) | 0.000 | 0.000 | 0.000 | 0.033 | 0.319 | 0.313 | 0.313 | 0.319 |

**Fraction of agents whose leader has low elevation (f3 ~ 0)**

| run group | 2% | 5% | 10% | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|---|---|---|
| MO-PEO, failed (n=26) | 0.714 | 0.681 | 0.626 | 0.555 | 0.527 | 0.516 | 0.516 | 0.522 |
| MO-PEO, not failed (n=14) | 0.742 | 0.659 | 0.500 | 0.412 | 0.346 | 0.374 | 0.368 | 0.357 |

**Distinct leaders (of 91 agents)**

| run group | 2% | 5% | 10% | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|---|---|---|
| MO-PEO, failed (n=26) | 15 | 14 | 30 | 34 | 34 | 34 | 36 | 34 |
| MO-PEO, not failed (n=14) | 16 | 20 | 30 | 36 | 36 | 37 | 38 | 37 |

**Exploitation fraction s**

| run group | 2% | 5% | 10% | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|---|---|---|
| MO-PEO, failed (n=26) | 0.643 | 0.489 | 0.500 | 0.511 | 0.505 | 0.495 | 0.462 | 0.456 |
| MO-PEO, not failed (n=14) | 0.615 | 0.484 | 0.516 | 0.500 | 0.505 | 0.505 | 0.500 | 0.473 |

**Smallest objective range of the archive (normalisation denominator)**

| run group | 2% | 5% | 10% | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|---|---|---|
| MO-PEO, failed (n=26) | 1.43 | 1.38 | 1.06 | 1.01 | 1 | 1 | 1 | 1 |
| MO-PEO, not failed (n=14) | 1.19 | 1.26 | 1.06 | 1.03 | 1.02 | 1.01 | 1.01 | 1 |

**Largest normalised objective value of any agent**

| run group | 2% | 5% | 10% | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|---|---|---|
| MO-PEO, failed (n=26) | 1.1 | 1.53 | 2.21 | 2.63 | 2.31 | 2.33 | 2.43 | 2.67 |
| MO-PEO, not failed (n=14) | 1.03 | 1.41 | 2.16 | 2.56 | 2.61 | 2.77 | 2.52 | 2.78 |

**Members removed by truncation per iteration**

| run group | 2% | 5% | 10% | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|---|---|---|
| MO-PEO, failed (n=26) | 10.5 | 13.5 | 23.5 | 24.0 | 24.0 | 26.0 | 27.0 | 30.5 |
| MO-PEO, not failed (n=14) | 0.0 | 11.5 | 27.0 | 28.0 | 26.5 | 28.0 | 32.5 | 30.5 |

**Offspring per iteration that land in the interior of the front**

| run group | 2% | 5% | 10% | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|---|---|---|
| MO-PEO, failed (n=26) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| MO-PEO, not failed (n=14) | 0.0 | 0.0 | 0.0 | 5.5 | 18.0 | 21.5 | 22.0 | 23.0 |

**... of which enter the archive**

| run group | 2% | 5% | 10% | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|---|---|---|
| MO-PEO, failed (n=26) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| MO-PEO, not failed (n=14) | 0.0 | 0.0 | 0.0 | 1.5 | 7.5 | 8.0 | 11.0 | 12.0 |

## Which branch produces interior offspring? (MO-PEO, all runs, all iterations)

| branch | offspring | interior offspring | rate per 1000 offspring | share of interior offspring |
|---|---|---|---|---|
| exploration (E + M + R) | 608178 | 4581 | 7.53 | 6.0% |
| exploitation, absolute noise | 418904 | 41764 | 99.70 | 54.7% |
| exploitation, relative noise | 166838 | 29957 | 179.56 | 39.3% |

Interior offspring over a whole run (median [min, max]): failed runs 0 [0, 170] ; not failed runs 5411 [2413, 7764].
Of those, entering the archive: failed 0 ; not failed 2164.

## Does crowding truncation (D4) remove interior members preferentially? (MO-PEO, all iterations, all runs)

Probability that a non-dominated member is removed by truncation: interior members 0.183 (47655 of 260693), all members 0.223 (335193 of 1504037).

## The discovery event (MO-PEO): first iteration at which any offspring lands in the interior of the front

Runs with at least one interior offspring: failed 3 of 26, not failed 14 of 14.
First interior offspring in the runs that succeed: median iteration 38 (0.12 of the budget), range 4-171 (0.01-0.52 of the budget).
Failed runs that had a first interior offspring: iterations [np.int64(7), np.int64(57), np.int64(66)] (they never built an archive with more than 16 interior members).
At the discovery iteration, the branches that produced interior offspring (a run can count in several): exploration 4, exploitation with absolute noise 13, exploitation with relative noise 1 (of 17 runs with a discovery).

Last iteration in which an interior offspring was produced, in the failed runs (0 = never): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 57, 102]

## Time to success

First iteration with at least 30 interior archive members (of 328 iterations), by algorithm:

| algorithm | reached (of 40) | median iteration among those | fraction of budget | 10th / 90th percentile |
|---|---|---|---|---|
| MO-PEO | 14 | 95 | 0.29 | 29 / 141 |
| GDE3 | 40 | 49 | 0.15 | 44 / 55 |
| NSGA-II | 28 | 52 | 0.16 | 26 / 131 |

## Is failure decided at initialisation? (MO-PEO)

| initial-population feature | failed runs (median) | not failed (median) | Mann-Whitney p | AUC for failure |
|---|---|---|---|---|
| fraction of initial agents in the interior | 0 | 0 | 0.311 | 0.46 |
| fraction of initial agents with x1 > 0.9705 | 0.0275 | 0.0275 | 0.784 | 0.47 |
| fraction with x2 > 0.9705 | 0.033 | 0.033 | 0.840 | 0.52 |
| interior members of the initial archive | 0 | 0 | 1.000 | 0.50 |
| pole members of the initial archive | 0 | 0 | 0.680 | 0.52 |
| IGD of the initial archive | 1.06 | 0.932 | 0.152 | 0.36 |

Interior archive members at 10% of the budget: failed median 0, not failed median 0 (Mann-Whitney p = 0.000405).

Interior archive members at 30% of the budget: failed median 0, not failed median 34 (Mann-Whitney p = 1.69e-07).