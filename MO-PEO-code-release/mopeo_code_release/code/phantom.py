"""Synthetic 2D radiotherapy phantom for the MO-PEO case study (HANDOVER 7.2).

This extends the same clinical scenario as the original (single-objective) PEO paper's radiotherapy
case study (main.tex Sec. 7.2 / Eq. 43: PTV = prostate, OARs = bladder and rectum), rather than an
unrelated toy problem, so the two papers describe the same application. For a genuinely two-objective
version (HANDOVER 7.2) the two OARs of the original are represented as a single "OAR" here, labelled
prostate/rectum since it is the organ that directly abuts the PTV in this geometry (the harder and
more clinically discussed constraint of the two); bladder-type sparing is not separately modelled.

A circular body cross-section with a central PTV (the prostate) and an adjacent OAR (the rectum)
whose boundaries overlap before the priority rule below is applied, so that sparing one costs
coverage of the other wherever they abut -- this is what makes the trade-off unavoidable rather than
resolvable by beam-angle choice alone (see rt_calibration.md). This is a synthetic digital phantom
for testing the optimizer, not a real patient anatomy or a real DICOM structure set; the geometry is
fixed and documented here so every downstream script uses the same one.
"""
import numpy as np

N = 96                 # grid size (n x n voxels)
PX_MM = 2.0             # mm per voxel edge
R_BODY = 45             # body radius, voxels (45 * 2mm = 9 cm radius, 18 cm diameter cross-section)
PTV_CENTER = (0.0, 0.0)  # voxel offset from isocenter (grid centre)
PTV_R = 12               # 2.4 cm radius PTV
OAR_CENTER = (0.0, 17.0)  # centre-to-centre distance < ptv_r + oar_r: PTV and OAR boundaries overlap
OAR_R = 9                # 1.8 cm radius OAR; the overlap region is assigned to PTV (tumour abutting
                          # the organ wall, e.g. a locally-advanced tumour invading an adjacent organ),
                          # leaving a crescent of OAR immediately against the PTV boundary on that side.
                          # This is what makes the trade-off unavoidable: covering the abutting part of
                          # the PTV cannot be done without dosing the adjacent OAR crescent, whichever
                          # beam angles are used, unlike a PTV/OAR pair separated by a healthy-tissue gap.


def make_phantom(n=N, px_mm=PX_MM, R_body=R_BODY, ptv_center=PTV_CENTER, ptv_r=PTV_R,
                  oar_center=OAR_CENTER, oar_r=OAR_R):
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    X = (xx - c).astype(float)
    Y = (yy - c).astype(float)
    body = (X ** 2 + Y ** 2) <= R_body ** 2
    ptv = ((X - ptv_center[0]) ** 2 + (Y - ptv_center[1]) ** 2) <= ptv_r ** 2
    oar = ((X - oar_center[0]) ** 2 + (Y - oar_center[1]) ** 2) <= oar_r ** 2
    ptv = ptv & body
    oar = oar & body & ~ptv          # PTV takes priority over OAR in any geometric overlap
    healthy = body & ~ptv & ~oar
    gap_vox = float(np.hypot(*np.subtract(oar_center, ptv_center)) - ptv_r - oar_r)
    return dict(n=n, px_mm=px_mm, R_body=R_body, X=X, Y=Y, body=body, ptv=ptv, oar=oar, healthy=healthy,
                ptv_center=ptv_center, ptv_r=ptv_r, oar_center=oar_center, oar_r=oar_r,
                ptv_oar_gap_vox=gap_vox, ptv_oar_gap_mm=gap_vox * px_mm)


if __name__ == "__main__":
    ph = make_phantom()
    print("voxels: body %d, PTV %d, OAR %d, healthy %d" % (ph["body"].sum(), ph["ptv"].sum(), ph["oar"].sum(), ph["healthy"].sum()))
    print("PTV-OAR gap: %.1f voxels (%.1f mm)" % (ph["ptv_oar_gap_vox"], ph["ptv_oar_gap_mm"]))
    print("body radius %d vox = %.1f cm; PTV radius %d vox = %.1f cm; OAR radius %d vox = %.1f cm" %
          (ph["R_body"], ph["R_body"] * ph["px_mm"] / 10, ph["ptv_r"], ph["ptv_r"] * ph["px_mm"] / 10,
           ph["oar_r"], ph["oar_r"] * ph["px_mm"] / 10))
