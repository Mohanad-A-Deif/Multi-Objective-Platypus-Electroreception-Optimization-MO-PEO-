"""Beamlet dose-influence matrix for the synthetic phantom (HANDOVER 7.2).

Simplified parallel-beam pencil-beam model: for each of `n_angles` coplanar gantry angles, the
beam is split into `n_beamlets` parallel beamlets across the body diameter. Each beamlet's dose to
a voxel is the product of an exponential attenuation along the beamlet's own axis from the body
surface (depth of the voxel's own ray, evaluated at the beamlet's lateral centre -- see note below)
and a Gaussian lateral penumbra centred on the beamlet. This is NOT a clinical dose calculation: it
omits beam divergence, scatter, heterogeneity and inverse-square falloff, and mu / sigma_pen are
representative orders of magnitude for a 6 MV beam in water-equivalent tissue at this resolution,
not measured. It is standard practice for a synthetic "digital phantom" beamlet-optimization test
problem (the kind used to test an optimizer, not to plan a real patient), and is documented here so
its limits are explicit before any clinical-sounding number is quoted from it.

dose(voxel) = D[voxel, :] @ w,  w >= 0 the beamlet weights (the decision vector optimized).
"""
import numpy as np

MU = 0.010        # attenuation per voxel (~ 6 MV in water-equivalent tissue at 2 mm/voxel)
SIGMA_PEN = 1.5    # Gaussian penumbra sigma, voxels (~3 mm)
N_ANGLES = 7
N_BEAMLETS = 16


def build_dose_matrix(ph, n_angles=N_ANGLES, n_beamlets=N_BEAMLETS, mu=MU, sigma_pen=SIGMA_PEN, angle0=0.0):
    R = ph["R_body"]
    body = ph["body"]
    rows, cols = np.where(body)
    X = ph["X"][rows, cols]
    Y = ph["Y"][rows, cols]
    n_vox = len(rows)
    angles = angle0 + np.arange(n_angles) * (2 * np.pi / n_angles)
    u_centers = np.linspace(-R + R / n_beamlets, R - R / n_beamlets, n_beamlets)
    beam_width = 2 * R / n_beamlets

    D = np.empty((n_vox, n_angles * n_beamlets), dtype=np.float64)
    col = 0
    for a in angles:
        d0, d1 = np.cos(a), np.sin(a)
        p0, p1 = -np.sin(a), np.cos(a)
        u_vox = X * p0 + Y * p1
        s_vox = X * d0 + Y * d1
        for uj in u_centers:
            entry_s = -np.sqrt(max(R * R - uj * uj, 0.0))
            depth = np.clip(s_vox - entry_s, 0.0, None)
            atten = np.exp(-mu * depth)
            pen = np.exp(-0.5 * ((u_vox - uj) / sigma_pen) ** 2)
            D[:, col] = atten * pen
            col += 1
    meta = dict(angles=angles, u_centers=u_centers, beam_width=beam_width, n_angles=n_angles,
                n_beamlets=n_beamlets, mu=mu, sigma_pen=sigma_pen)
    return D, rows, cols, meta


def calibrate_prescription(D, ptv_idx):
    """Weight w0 such that a uniform (open, unweighted) plan delivers mean PTV dose = 1.0 (the
    prescription dose Rx, in these normalized units). Returns (w0, U) with U a generous upper
    bound (4x w0) for the per-beamlet weight used as the optimization variable bound."""
    dose_uniform = D.sum(1)
    m = dose_uniform[ptv_idx].mean()
    w0 = 1.0 / m
    return w0, 4.0 * w0


if __name__ == "__main__":
    from phantom import make_phantom
    ph = make_phantom()
    D, rows, cols, meta = build_dose_matrix(ph)
    ptv_idx = ph["ptv"][rows, cols]
    oar_idx = ph["oar"][rows, cols]
    w0, U = calibrate_prescription(D, ptv_idx)
    print("D shape:", D.shape, "beamlets:", meta["n_angles"], "x", meta["n_beamlets"])
    print("w0 = %.5g (uniform weight giving mean PTV dose = 1.0 Rx), U = %.5g" % (w0, U))
    dose = D @ (w0 * np.ones(D.shape[1]))
    print("uniform plan: mean PTV dose %.3f, mean OAR dose %.3f, PTV D95 %.3f, max dose in body %.3f" %
          (dose[ptv_idx].mean(), dose[oar_idx].mean(), np.percentile(dose[ptv_idx], 5), dose.max()))
