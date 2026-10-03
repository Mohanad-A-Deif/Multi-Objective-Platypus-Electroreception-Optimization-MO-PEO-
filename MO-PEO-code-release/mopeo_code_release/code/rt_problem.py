"""MO-PEO radiotherapy case study: problem definition (HANDOVER 7.2).

Extends Eq. (43) of the original PEO paper (main.tex Sec. 7.2), which minimizes the WEIGHTED SUM
    f(x) = sum_{i in PTV} w_i (D_i - D_presc)^2  +  sum_{j in OAR} v_j max(0, D_j - Dmax)^2,
into two separate objectives -- f1 the PTV term, f2 the OAR term, same functional form, no w_i/v_j
weights to fix in advance -- so the trade-off the original collapsed into one scalar is now a
genuine Pareto front. Two dose-volume constraints (PTV D95, OAR V_Dmax) are added on top, handled by
constrained-domination (Deb 2000) in the archive rather than a penalty; the original paper had no
such constraint (HANDOVER 7.2). Decision vector w = beamlet weights, w >= 0.

Doses are reported in Gy, at a prescription RX_GY = 74 Gy (a standard conventional-fractionation
prostate EBRT prescription, and in the range of the ~70-71 Gy mean PTV dose the original paper's
single-objective PEO run reports). Internally the beamlet weights are still calibrated in the
dose_model.py normalized units (mean PTV dose = 1.0 for a uniform plan) and then scaled by RX_GY;
this is a constant rescaling of both objectives together, so it changes none of the Pareto-front or
feasibility structure, only the units the numbers are reported in.

The two dose-volume constraint levels below are chosen (code/rt_calibration.md) so that neither
extreme of the PTV/OAR trade-off is feasible on its own -- maximal PTV coverage violates the OAR
constraint and maximal OAR sparing violates the PTV constraint -- leaving a narrow feasible corridor
in between for the optimizer to find, rather than a constraint that is trivially satisfied or never
satisfiable. This is choosing the problem's own definition, done once before any MOEA is run on it,
not tuning MO-PEO.
"""
import numpy as np
from phantom import make_phantom
from dose_model import build_dose_matrix, calibrate_prescription

RX_GY = 74.0                  # prescription dose, Gy
DMAX_GY = 22.0                 # OAR (rectum) dose-volume constraint threshold, Gy
D95_TARGET_GY = 0.95 * RX_GY  # PTV D95 (dose covering 95% of the PTV) must be >= this; also f1's hinge
D95_FRAC_LIMIT = 0.05          # equivalently: at most 5% of PTV volume may fall below D95_TARGET_GY
OAR_VOL_LIMIT = 0.15           # at most 15% of OAR volume may receive >= DMAX_GY


class RTProblem:
    def __init__(self):
        self.ph = make_phantom()
        self.D, self.rows, self.cols, self.meta = build_dose_matrix(self.ph)
        self.ptv_idx = self.ph["ptv"][self.rows, self.cols]
        self.oar_idx = self.ph["oar"][self.rows, self.cols]
        self.healthy_idx = self.ph["healthy"][self.rows, self.cols]
        self.w0, self.U0 = calibrate_prescription(self.D, self.ptv_idx)
        self.n_beamlets = self.D.shape[1]
        self.L = np.zeros(self.n_beamlets)
        self.U = np.full(self.n_beamlets, self.U0)

    def dose(self, W):
        """W: (n, n_beamlets) weight vectors -> (n, n_voxels) doses, in Gy."""
        return (W @ self.D.T) * RX_GY

    def objectives(self, W):
        """Eq. (43) of the original PEO paper, split into its two terms (f1 = PTV, f2 = OAR) instead
        of a weighted sum, with two deliberate departures from its exact functional form, both
        documented rather than silently made:
        (1) f1 is the one-sided underdose penalty max(D_target - D_i, 0)^2, not Eq. (43)'s two-sided
        (D_i - D_presc)^2. A two-sided penalty tolerates being under OR over D_target equally, so it
        does not by itself drive PTV coverage (D95) without the per-voxel weights w_i Eq. (43) needed
        hand-tuned -- exactly the manual weighting this two-objective version removes. One-sided
        "underdose-only" objectives are also the standard choice in practical inverse planning.
        (2) f2 is the plain mean squared OAR dose, not Eq. (43)'s one-sided max(0, D_j-Dmax)^2. The
        hinge form makes the OAR term free below Dmax with nothing to push weight down further,
        which left the coverage-only corner of the trade-off numerically degenerate (arbitrarily
        large, unconstrained beamlet weights with zero PTV or OAR cost) and produced a wide range of
        k with no OAR constraint bound at all (code/rt_calibration.md, superseded version) rather than
        the intended narrow corridor. "Minimize mean OAR dose" is itself a standard clinical planning
        objective, if anything more common than a single hard Dmax cutoff."""
        dose = self.dose(W)
        d_ptv = dose[:, self.ptv_idx]
        d_oar = dose[:, self.oar_idx]
        f1 = np.mean(np.maximum(D95_TARGET_GY - d_ptv, 0.0) ** 2, axis=1)   # PTV coverage
        f2 = np.mean(d_oar ** 2, axis=1)                                     # OAR sparing
        return np.c_[f1, f2]

    def violation(self, W):
        """Two dose-volume constraints, absent from the original paper's penalty-only formulation
        (HANDOVER 7.2), handled by constrained-domination rather than folded into the objectives."""
        dose = self.dose(W)
        d_ptv = dose[:, self.ptv_idx]
        d_oar = dose[:, self.oar_idx]
        frac_under = np.mean(d_ptv < D95_TARGET_GY, axis=1)
        v1 = np.maximum(frac_under - D95_FRAC_LIMIT, 0.0)
        frac_over = np.mean(d_oar >= DMAX_GY, axis=1)
        v2 = np.maximum(frac_over - OAR_VOL_LIMIT, 0.0)
        return v1 + v2, v1, v2

    def metrics(self, w):
        dose = self.D @ w * RX_GY
        d_ptv, d_oar = dose[self.ptv_idx], dose[self.oar_idx]
        D95 = np.percentile(d_ptv, 5)              # dose covering 95% of the PTV volume, Gy
        Dmean_oar = d_oar.mean()
        Vtol_oar = np.mean(d_oar >= DMAX_GY)
        ref = dose >= D95_TARGET_GY                  # reference isodose = 95% Rx level, whole body
        v_ptv_ref = (self.ptv_idx & ref).sum()
        v_ptv = self.ptv_idx.sum()
        v_ref = ref.sum()
        CI = (v_ptv_ref ** 2) / (v_ptv * v_ref) if v_ptv > 0 and v_ref > 0 else 0.0
        return dict(D95_PTV=float(D95), Dmean_OAR=float(Dmean_oar), Vtol_OAR=float(Vtol_oar), CI=float(CI))


if __name__ == "__main__":
    rt = RTProblem()
    rng = np.random.default_rng(0)
    print("beamlets: %d, U0 = %.4g" % (rt.n_beamlets, rt.U0))

    W = np.r_[rt.w0 * np.ones((1, rt.n_beamlets)),                       # uniform / open field
              rng.random((2000, rt.n_beamlets)) * rt.U0]                 # random plans
    F = rt.objectives(W)
    V, V1, V2 = rt.violation(W)
    print("uniform plan: f1(PTV)=%.4g Gy^2 f2(OAR)=%.4g Gy^2  v1=%.3f v2=%.3f  feasible=%s" %
          (F[0, 0], F[0, 1], V1[0], V2[0], V[0] <= 0))
    print("random plans: feasible fraction %.3f (v1>0: %.3f, v2>0: %.3f)" %
          ((V[1:] <= 0).mean(), (V1[1:] > 0).mean(), (V2[1:] > 0).mean()))
    print(rt.metrics(W[0]))

    # sanity: weighting away from the OAR side should lower f2 and raise v2's slack
    w_spare = rt.w0 * np.ones(rt.n_beamlets)
    angles = rt.meta["angles"]
    beam_dir = np.repeat(np.c_[np.cos(angles), np.sin(angles)], rt.meta["n_beamlets"], axis=0)
    toward_oar = beam_dir[:, 1] > 0.3   # beams entering roughly from the OAR side of the phantom
    w_spare[toward_oar] *= 0.2
    f_spare = rt.objectives(w_spare[None])[0]
    v_spare = rt.violation(w_spare[None])[0][0]
    print("down-weighting OAR-side beams: f1=%.4g (was %.4g) f2=%.4g (was %.4g) v=%.3f (was %.3f)" %
          (f_spare[0], F[0, 0], f_spare[1], F[0, 1], v_spare, V[0]))
