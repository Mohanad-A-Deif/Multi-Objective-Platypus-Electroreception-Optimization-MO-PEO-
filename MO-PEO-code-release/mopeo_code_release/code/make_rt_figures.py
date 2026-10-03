#!/usr/bin/env python3
"""Figures for the radiotherapy case study (HANDOVER 7.2): the feasible Pareto front and the dose
maps of three representative plans on it. Nature/IEEE grayscale style throughout (project style
guide): white background, serif font, thin inward ticks, 600 dpi, structures distinguished by line
style/markers and contour, never colour.
Usage: python make_rt_figures.py  ->  results/rt_front.png, results/rt_dose_maps.png"""
import json
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rt_problem import RTProblem, D95_TARGET_GY, DMAX_GY, OAR_VOL_LIMIT

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.edgecolor": "black", "axes.linewidth": 0.8, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7, "legend.frameon": False,
    "figure.dpi": 600, "savefig.dpi": 600, "figure.facecolor": "white", "axes.facecolor": "white",
    "mathtext.fontset": "dejavuserif",
})

rt = RTProblem()
data = json.load(open("../results/rt_run.json"))
res0 = data["results"]["mopeo"][0]
AF = np.array(res0["F"]); AV = np.array(res0["V"]); AX = np.array(res0["X"])
feas = AV <= 0
F_feas, X_feas = AF[feas], AX[feas]

# ---------------------------------------------------------------- Pareto front figure --
# y-axis is OAR V_tol (the constrained quantity, dimensionless fraction of OAR volume >= DMAX_GY),
# not OAR mean dose: this makes the plotted constraint boundary (OAR_VOL_LIMIT) and the calibration
# figure (rt_calibration.png) use the same axes, and avoids putting a dose-threshold line (DMAX_GY)
# on an axis that is not itself a dose.
fig, ax = plt.subplots(figsize=(3.6, 2.9), constrained_layout=True)
D95s = np.array([rt.metrics(w)["D95_PTV"] for w in X_feas])
Vtols = np.array([rt.metrics(w)["Vtol_OAR"] for w in X_feas])
order = np.argsort(D95s)

# Non-dominated subset only (maximize D95, minimize Vtol): the raw feasible archive includes many
# dominated points, and connecting all of them by D95-sorted order zig-zags in Vtol and misrepresents
# an unordered archive as a monotonic sweep. The connecting line is drawn through the actual Pareto-
# efficient points only; all feasible archive members are still shown as the scatter underneath.
nd = np.ones(len(D95s), dtype=bool)
for i in range(len(D95s)):
    dominated = (D95s >= D95s[i]) & (Vtols <= Vtols[i]) & ((D95s > D95s[i]) | (Vtols < Vtols[i]))
    if dominated.any():
        nd[i] = False
nd_order = np.argsort(D95s[nd])
ax.plot(D95s[nd][nd_order], Vtols[nd][nd_order], color="0.55", lw=0.9, zorder=1)
line = ax.scatter(D95s, Vtols, facecolor="none", edgecolor="black", marker="o", s=14, lw=0.7, zorder=2,
                   label="feasible plans")

i_cov = np.argmax(D95s); i_spa = np.argmin(Vtols)
i_knee = order[len(order) // 2]
labelled = [(i_cov, "best\ncoverage", "^", (6, 6), "left", "bottom"),
            (i_knee, "knee", "D", (-7, -8), "right", "top"),
            (i_spa, "best\nsparing", "v", (6, -6), "left", "top")]
for i, lab, mk, off, ha, va in labelled:
    ax.scatter([D95s[i]], [Vtols[i]], facecolor="black", edgecolor="black", marker=mk, s=28, zorder=3)
    ax.annotate(lab, (D95s[i], Vtols[i]), textcoords="offset points", xytext=off, fontsize=6.3, ha=ha, va=va)

ax.axvline(D95_TARGET_GY, color="0.7", lw=0.6, ls="--")
ax.axhline(OAR_VOL_LIMIT, color="0.7", lw=0.6, ls="--")
ax.set_xlabel("PTV D95 (Gy)"); ax.set_ylabel("OAR $V_{tol}$ (fraction $\\geq$ %.0f Gy)" % DMAX_GY)
ax.margins(x=0.12, y=0.15)
# legend outside the axes, per the project's Nature-style guide ("legend outside, never hiding data")
ax.legend(handles=[line], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1,
          borderaxespad=0.2, handletextpad=0.4)
fig.savefig("../results/rt_front.png")
print("wrote results/rt_front.png")

# ---------------------------------------------------------------- dose map figure --
# Two fixes to the earlier version of this figure:
# (1) the simplified attenuation model (dose_model.py) has no surface-buildup region, so every beam
#     is hottest at skin entry; the resulting peripheral rim (mean ~144 Gy, checked numerically) has
#     nothing to do with the PTV/OAR trade-off and dominated a whole-body view. Fixed by cropping the
#     display window to a margin around the PTV/OAR (as the original PEO paper's own dose-map figure
#     is itself a bounded view, not whole-body), not by hiding or renormalizing the rim's actual value.
# (2) a fixed vmax=90 Gy saturated (and made indistinguishable) both the PTV core, which legitimately
#     runs well above the D95 constraint since nothing in the problem penalizes PTV hot spots, and the
#     entry rim. vmax is now set from the data itself (99th percentile dose inside the crop window,
#     across all three plans) so the true range is visible; isodose contours at the two dose levels
#     that actually define the constraints (D95_TARGET_GY, DMAX_GY) are overlaid for clinical reading.
ph = rt.ph
weights = data["representative_weights"]
names = ["best coverage", "knee (median-coverage)", "best sparing"]

roi = ph["ptv"] | ph["oar"]
rr, cc = np.where(roi)
margin = 18  # voxels (~3.6 cm) around the PTV/OAR union
r0, r1 = max(rr.min() - margin, 0), min(rr.max() + margin, ph["n"] - 1)
c0, c1 = max(cc.min() - margin, 0), min(cc.max() + margin, ph["n"] - 1)

def display_smooth(a, sigma=0.8):
    """Gaussian-smooth a 2D field for DISPLAY ONLY, correctly handling the NaN border (outside the
    body) by normalizing against a smoothed validity mask rather than letting NaN/zero-padding bleed
    dark edges in. With only 7 gantry angles x 16 beamlets and a narrow penumbra (sigma_pen=1.5 vox,
    dose_model.py), the raw pencil-beam field has a fine speckled/streaked texture wherever beams from
    few angles overlap (mainly the periphery); this is a genuine feature of that simplified model, not
    a bug, but it obscures the PTV/OAR trade-off the figure is meant to show. All quantitative metrics
    (D95, Dmean, Vtol, CI in rt_problem.py) are computed from the unsmoothed dose field and are
    unaffected by this display choice."""
    valid = ~np.isnan(a)
    filled = np.where(valid, a, 0.0)
    num = gaussian_filter(filled, sigma=sigma)
    den = gaussian_filter(valid.astype(float), sigma=sigma)
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
    out[~valid] = np.nan
    return out


ZOOM = 4  # upsampling factor for structure-outline display only (mask stays circular; this just
           # avoids the faceted/staircase look marching-squares gives a 12-voxel-radius circle)


def hi_res_mask(mask2d):
    hi = zoom(mask2d.astype(float), ZOOM, order=1)
    h, w = mask2d.shape
    xs = np.linspace(0, w - 1, hi.shape[1])
    ys = np.linspace(0, h - 1, hi.shape[0])
    return xs, ys, hi


dosemaps = {}
for name in names:
    w = np.array(weights[name])
    dm = np.full((ph["n"], ph["n"]), np.nan)
    dm[rt.rows, rt.cols] = rt.D @ w * 74.0
    dm[~ph["body"]] = np.nan
    dosemaps[name] = display_smooth(dm[r0:r1 + 1, c0:c1 + 1])
vmax = float(np.ceil(np.nanpercentile(np.stack(list(dosemaps.values())), 99) / 10.0) * 10.0)

ptv_xs, ptv_ys, ptv_hi = hi_res_mask(ph["ptv"][r0:r1 + 1, c0:c1 + 1])
oar_xs, oar_ys, oar_hi = hi_res_mask(ph["oar"][r0:r1 + 1, c0:c1 + 1])

fig2, axs = plt.subplots(1, 3, figsize=(7.0, 2.8), constrained_layout=True)
for ax, name in zip(axs, names):
    dm = dosemaps[name]
    im = ax.imshow(dm, cmap="Greys", vmin=0, vmax=vmax, origin="lower")
    cs = ax.contour(dm, levels=[DMAX_GY, D95_TARGET_GY], colors="0.3", linewidths=0.7,
                     linestyles=["--", "-"])
    ax.clabel(cs, inline=True, fontsize=5, fmt={DMAX_GY: "%d Gy" % DMAX_GY,
                                                 D95_TARGET_GY: "%d Gy" % D95_TARGET_GY})
    ax.contour(ptv_xs, ptv_ys, ptv_hi, levels=[0.5], colors="black", linewidths=1.1, linestyles="-")
    ax.contour(oar_xs, oar_ys, oar_hi, levels=[0.5], colors="black", linewidths=1.1, linestyles=":")
    ax.set_title(name, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
cbar = fig2.colorbar(im, ax=axs, shrink=0.75, pad=0.02)
cbar.set_label("dose (Gy)", fontsize=8)
axs[0].plot([], [], color="black", lw=1.1, ls="-", label="PTV")
axs[0].plot([], [], color="black", lw=1.1, ls=":", label="OAR")
axs[0].plot([], [], color="0.3", lw=0.7, ls="-", label="%d Gy isodose (D95 target)" % D95_TARGET_GY)
axs[0].plot([], [], color="0.3", lw=0.7, ls="--", label="%d Gy isodose ($V_{tol}$ threshold)" % DMAX_GY)
fig2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=6.3, frameon=False,
            handletextpad=0.5, columnspacing=1.2)
fig2.savefig("../results/rt_dose_maps.png", bbox_inches="tight")
print("wrote results/rt_dose_maps.png (crop rows %d:%d cols %d:%d, vmax=%.0f Gy)" % (r0, r1, c0, c1, vmax))
