# DTLZ4: dynamic ablations of one term of the update (probe; 40 seeds, 7000-7039, tau = 0.1)

| configuration | failed runs (IGD > 0.2) | median IGD of the others | median IGD overall |
|---|---|---|---|
| reported | 28/40 | 0.0874 | 0.257 |
| no leader attraction (gamma = lambda = 0) | 40/40 | - | 0.533 |
| no directional-sensing term (alpha = 0) | 28/40 | 0.0775 | 0.257 |
| noise on every coordinate | 5/40 | 0.0892 | 0.0894 |
| noise only (alpha = gamma = lambda = 0) | 40/40 | - | 0.49 |
