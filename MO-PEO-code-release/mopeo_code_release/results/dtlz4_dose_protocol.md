# Pre-declared protocol: dose-response of the perturbation density on DTLZ4 (written before running)

Written 2026-09-23, before any run of this experiment, per the user's instruction after reading the DTLZ4
mechanism report (`results/diag_dtlz4_mech.md`, HANDOVER 7.1g). That report implicated the single-coordinate
perturbation (k = 1 of D coordinates per agent) as a candidate cause of the DTLZ4 discovery failure, without a
dose-response run, and noted that dense noise (k = D) is already known to hurt ZDT4 and DTLZ1. This experiment
tests a declared grid between those two extremes and commits to an acceptance rule before seeing any result, so
it cannot become a tuning exercise.

## Design

* **Algorithm.** The shipped configuration, MO-PEO-final (`mode="delay", rspeed=TAU=0.1`), with `sparse` replaced
  by the new `density` parameter of `mopeo4` (added 2026-09-23; `density=None`, i.e. every previously stored key,
  is byte-for-byte unaffected -- verified). Nothing else changes.
* **Density grid** (fraction of the D coordinates perturbed per agent, k = max(1, round(density * D))):
  1/D (= the reported, currently shipped setting; k = 1 regardless of D), 0.1, 0.25, 0.5, 1.0 (= dense, all
  coordinates -- identical code path to the existing A0/B2 dense-noise ablations, verified bit-identical at
  density=1.0).
* **Problems.** DTLZ4 (D = 12, the failing problem), plus ZDT4 (D = 10), DTLZ1 (D = 7), DTLZ3 (D = 12) and DTLZ6
  (D = 12) as the four problems already known or suspected to be harmed by denser perturbation (7.1f: A0/B2 dense
  ablations; option (b) comparison). DTLZ3 is reported but excluded from the acceptance rule (every algorithm in
  the six-algorithm comparison already fails DTLZ3 by the >5x-best-median criterion; its numbers are too noisy at
  n = 40 to gate a decision on, per the existing convention of excluding it from the failure-rate tables).
* **Seeds.** The 40 calibration seeds, 6000-6039 (`calibrate_tau.py`) -- NOT the protocol seeds (1000-1019) and
  NOT the DTLZ4-diagnosis seeds (7000-7039), so this is neither the seed pool tau was tuned on for the final
  report nor the seed pool the mechanism was diagnosed on. Population size and evaluation budget as in the main
  protocol (`reproduce.problem()`: ZDT N=100/FE=25000, DTLZ N=91/FE=30000).
* **Outcome measures.** For DTLZ4: failure rate (IGD > 0.2, the criterion already used throughout 7.1e/7.1g) at
  each density. For ZDT4, DTLZ1, DTLZ3, DTLZ6: median IGD at each density, reported as the ratio to the density
  = 1/D (reported) baseline.

## Acceptance rule (fixed before running; nothing on this list may change afterward)

A density value on the grid is **adopted** as a proposed change to MO-PEO only if, at that density, ALL of the
following hold simultaneously:

1. DTLZ4 failure rate (of 40) is below 10/40 (i.e. < 25%, against the reported ~28/40 = 70% on the diagnosis
   seeds and the expected similar rate here); AND
2. median IGD on ZDT4 does not degrade by more than 1.5x relative to density = 1/D; AND
3. median IGD on DTLZ1 does not degrade by more than 1.5x relative to density = 1/D; AND
4. median IGD on DTLZ6 does not degrade by more than 1.5x relative to density = 1/D.

If more than one density on the grid satisfies all four conditions, the smallest such density is preferred (it
is closest to the shipped configuration and changes the least). If no density on the grid satisfies all four
conditions, none is adopted: the DTLZ4 discovery failure is reported as a characterised limitation (the
mechanism analysis in 7.1g already stands on its own), and MO-PEO-final (density = 1/D) remains shipped as
decided (HANDOVER 7.1f). No other rule, threshold, or reading of "improvement" is used to justify a change; a
density that helps DTLZ4 but fails 2-4 is not adopted regardless of how large the DTLZ4 gain is.

This is a diagnostic/exploratory run on calibration seeds, not a second confirmatory protocol run: if a density
is adopted, it would still need the full 12-problem/20-run protocol and the standing reproducibility audit
(`reproduce.py --stage audit`) before it could replace the shipped configuration in the manuscript.
