#!/usr/bin/env python3
"""
make_figures.py — regenerate every figure of the MO-PEO study from stored results.

Usage
-----
    python make_figures.py              # all figures
    python make_figures.py 2 4          # only figures 2 and 4

Everything you are likely to change lives in the CONFIG block below:
page widths, font sizes, dpi, output format, the grey level / marker / line style
of each algorithm, which problems appear in each figure, and the panel layout.
No other part of the file needs editing for a restyle.

Inputs (paths set in CONFIG.data):
    results_final.json  IGD and HV per run, per problem, per algorithm variant
    fronts_med.pkl      objective vectors of the median run   -> figures 1 and 3
    conv.pkl            IGD against budget                    -> figure 4
    tau.json            tau sensitivity                       -> figure 5
    ref_fronts.pkl      sampled true Pareto fronts            -> figures 1 and 3
                        (if problems.py is importable it is used instead)
"""
import json, pickle, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==================================================================== CONFIG ==
CONFIG = dict(

    data=dict(                      # where the stored results live
        dir="results",              # folder holding the files below
        results="results_final.json",
        fronts="fronts_med.pkl",
        conv="conv.pkl",
        tau="tau.json",
        ref="ref_fronts.pkl",
    ),

    out=dict(dir="figs", fmt="png", dpi=600),   # fmt: png / pdf / svg / eps

    rc={                            # matplotlib rcParams (Nature/IEEE greyscale)
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 11,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "grid.color": "black",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "lines.linewidth": 1.0,
        "mathtext.fontset": "dejavuserif",
    },

    # display name -> key in results_final.json / fronts_med.pkl
    algorithms={"MO-PEO": "MO-PEO-final", "NSGA-II": "NSGA-II", "MOEA/D": "MOEA/D",
                "MOPSO": "MOPSO", "GDE3": "GDE3", "MOGWO": "MOGWO"},

    # greyscale only: algorithms are separated by marker and line style, never colour
    style={
        "MO-PEO":  dict(color="black", marker="o", ls="-",  ms=3.0, mfc="none", mew=0.7),
        "NSGA-II": dict(color="0.45",  marker="s", ls="--", ms=3.0, mfc="none", mew=0.7),
        "MOEA/D":  dict(color="0.65",  marker="^", ls=":",  ms=3.0, mfc="none", mew=0.7),
        "MOPSO":   dict(color="0.30",  marker="v", ls=(0, (3, 1, 1, 1, 1, 1)), ms=3.0, mfc="none", mew=0.7),
        "GDE3":    dict(color="0.55",  marker="D", ls="-.", ms=2.6, mfc="none", mew=0.7),
        "MOGWO":   dict(color="0.78",  marker="<", ls=(0, (5, 1)), ms=3.0, mfc="none", mew=0.7),
    },
    legend_ncol=4,                  # columns of the legend above figures 1 and 4 (seven entries in figure 1)

    truth=dict(color="0.55", lw=1.2, ls="-", label="True PF"),   # reference front, 2-D
    truth3d=dict(color="0.82", s=0.8),                           # reference front, 3-D

    boxstyle=dict(facecolor="0.92", edgecolor="black", linewidth=0.8,
                  median_lw=1.2, flier_marker="+", flier_ms=3, width=0.55),

    tag=dict(fc="white", ec="none", alpha=0.75, pad=2.0),        # panel-label box

    fig1=dict(problems=["ZDT1", "ZDT3", "ZDT4", "ZDT6"], size=(7.0, 9.0),
              groups=[["MO-PEO", "NSGA-II", "MOEA/D"], ["MO-PEO", "MOPSO", "GDE3", "MOGWO"]],
              ylim={"ZDT4": (-0.05, 2.0)}, xlim={"ZDT6": (0.2, 1.05)}),
    fig2=dict(problems=["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6",
                        "DTLZ1", "DTLZ2", "DTLZ4", "DTLZ6", "DTLZ7"],
              ncols=2, size=(7.0, 11.0), logy=True, label_rotation=35, label_fontsize=7.5),
    fig3=dict(problem="DTLZ2", ncols=3, size=(7.0, 5.0), elev=26, azim=40, ms=9),
    fig4=dict(problems=["ZDT1", "ZDT4", "DTLZ1", "DTLZ6"], size=(3.4, 9.6), logy=True),
    fig5=dict(unimodal=["ZDT1"], multimodal=["ZDT4", "DTLZ1", "DTLZ6"],
              taus=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=(3.4, 5.6),
              greys=["black", "0.35", "0.55", "0.7"],
              markers=[dict(marker="o", ls="-"), dict(marker="s", ls="--"),
                       dict(marker="^", ls=":"), dict(marker="d", ls="-.")],
              vline=0.4),
)
# ============================================================== END OF CONFIG ==

C = CONFIG
plt.rcParams.update(C["rc"])
plt.rcParams["figure.dpi"] = C["out"]["dpi"]
plt.rcParams["savefig.dpi"] = C["out"]["dpi"]
TAGS = "abcdefghij"


def path(key):
    return os.path.join(C["data"]["dir"], C["data"][key])


def load(key, binary=False):
    p = path(key)
    if not os.path.exists(p):                      # also accept files next to the script
        p = C["data"][key]
    with open(p, "rb" if binary else "r") as fh:
        return pickle.load(fh) if binary else json.load(fh)


def reference_front(name):
    """True Pareto front: from problems.py when importable, else from the stored pickle."""
    try:
        from problems import zdt_pf, dtlz_pf
        return zdt_pf(name) if name.startswith("ZDT") else dtlz_pf(name)
    except Exception:
        return load("ref", binary=True)[name]


def tag(ax, i, text, loc="lower left"):
    x, y, ha, va = dict(
        **{"lower left": (0.02, 0.04, "left", "bottom"),
           "upper left": (0.03, 0.93, "left", "top"),
           "upper right": (0.97, 0.93, "right", "top")})[loc]
    ax.text(x, y, "(%s) %s" % (TAGS[i], text), transform=ax.transAxes,
            ha=ha, va=va, bbox=dict(**C["tag"]))


def save(fig, stem, **kw):
    os.makedirs(C["out"]["dir"], exist_ok=True)
    f = os.path.join(C["out"]["dir"], "%s.%s" % (stem, C["out"]["fmt"]))
    fig.savefig(f, dpi=C["out"]["dpi"], bbox_inches="tight", **kw)
    plt.close(fig)
    print("wrote", f)


# ------------------------------------------------------------------ figure 1 --
def figure1():
    """Non-dominated fronts of the median run on the bi-objective problems.
    Two columns, same axes in every row (shared): the three original algorithms on the left, MO-PEO against the three
    added baselines on the right; six overlaid marker sets in one panel are unreadable."""
    cfg = C["fig1"]
    fronts = load("fronts", binary=True)
    groups = cfg["groups"]
    fig, axes = plt.subplots(len(cfg["problems"]), len(groups), figsize=cfg["size"], sharex="row", sharey="row", squeeze=False)
    for i, name in enumerate(cfg["problems"]):
        P = reference_front(name)
        for g, labs in enumerate(groups):
            ax = axes[i, g]
            ax.plot(P[:, 0], P[:, 1], label=C["truth"]["label"] if i == 0 else None,
                    zorder=1, **{k: v for k, v in C["truth"].items() if k != "label"})
            for lab in labs:
                A = fronts.get((name, C["algorithms"][lab]))
                if A is None:
                    continue
                s = dict(C["style"][lab]); s.pop("ls")
                ax.plot(A[:, 0], A[:, 1], ls="none", color=s.pop("color"),
                        label=lab if i == 0 else None, zorder=3, **s)
            ax.grid(True)
            if g == 0: ax.set_ylabel(r"$f_2$")
            tag(ax, 2 * i + g, name, loc="lower left")
            if name in cfg["ylim"]: ax.set_ylim(*cfg["ylim"][name])
            if name in cfg["xlim"]: ax.set_xlim(*cfg["xlim"][name])
    for g in range(len(groups)):
        axes[-1, g].set_xlabel(r"$f_1$")
        h, l = axes[0, g].get_legend_handles_labels()
        axes[0, g].legend(h, l, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, borderaxespad=0.0)
    fig.tight_layout()
    save(fig, "fig1_fronts_zdt")


# ------------------------------------------------------------------ figure 2 --
def figure2():
    """Boxplots of IGD over all runs."""
    cfg = C["fig2"]; bs = C["boxstyle"]
    res = load("results")
    labs = list(C["algorithms"])
    nrows = int(np.ceil(len(cfg["problems"]) / cfg["ncols"]))
    fig, axes = plt.subplots(nrows, cfg["ncols"], figsize=cfg["size"])
    for i, (name, ax) in enumerate(zip(cfg["problems"], np.ravel(axes))):
        data = [res["%s|%s" % (name, C["algorithms"][l])]["IGD"] for l in labs]
        bp = ax.boxplot(data, widths=bs["width"], patch_artist=True, showfliers=True)
        for b in bp["boxes"]:
            b.set(facecolor=bs["facecolor"], edgecolor=bs["edgecolor"], linewidth=bs["linewidth"])
        for part in ("whiskers", "caps"):
            for w in bp[part]:
                w.set(color=bs["edgecolor"], linewidth=bs["linewidth"])
        for m in bp["medians"]:
            m.set(color="black", linewidth=bs["median_lw"])
        for f in bp["fliers"]:
            f.set(marker=bs["flier_marker"], markersize=bs["flier_ms"],
                  markeredgecolor="0.4", markeredgewidth=0.6)
        if cfg["logy"]: ax.set_yscale("log")
        y0, y1 = ax.get_ylim()                        # head-room so that the panel tag never covers a box
        ax.set_ylim(y0, y1 * (y1 / y0) ** 0.22 if cfg["logy"] else y1 + 0.22 * (y1 - y0))
        ax.grid(True, axis="y")
        ax.set_xticklabels(labs, fontsize=cfg["label_fontsize"], rotation=cfg["label_rotation"], ha="right", rotation_mode="anchor")
        tag(ax, i, name, loc="upper left")
        if i % cfg["ncols"] == 0: ax.set_ylabel("IGD")
    fig.tight_layout()
    save(fig, "fig2_boxplots")


# ------------------------------------------------------------------ figure 3 --
def figure3():
    """Three-objective fronts, one panel per algorithm."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the projection)
    cfg = C["fig3"]
    fronts = load("fronts", binary=True)
    P = reference_front(cfg["problem"])
    fig = plt.figure(figsize=cfg["size"])
    nrows = int(np.ceil(len(C["algorithms"]) / cfg["ncols"]))
    for i, lab in enumerate(C["algorithms"]):
        ax = fig.add_subplot(nrows, cfg["ncols"], i + 1, projection="3d")
        A = fronts[(cfg["problem"], C["algorithms"][lab])]
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=C["truth3d"]["color"], s=C["truth3d"]["s"],
                   marker=".", linewidths=0, depthshade=False)
        ax.scatter(A[:, 0], A[:, 1], A[:, 2], s=cfg["ms"], facecolors="none",
                   edgecolors="black", linewidths=0.6,
                   marker=C["style"][lab]["marker"], depthshade=False)
        ax.view_init(elev=cfg["elev"], azim=cfg["azim"])
        ax.set_xlabel(r"$f_1$", labelpad=-6, fontsize=9)
        ax.set_ylabel(r"$f_2$", labelpad=-6, fontsize=9)
        # the z label is drawn by hand: set_zlabel follows the rotating axis and
        # ends up either inside the data or clipped at the figure edge
        ax.text2D(-0.02, 0.52, r"$f_3$", transform=ax.transAxes, fontsize=9,
                  rotation=90, va="center", ha="right")
        ax.tick_params(pad=-2, labelsize=7)
        ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0]); ax.set_zticks([0, 0.5, 1.0])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor("white"); axis.pane.set_edgecolor("0.7")
            axis._axinfo["grid"].update(dict(color="0.85", linestyle=":", linewidth=0.5))
        ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05); ax.set_zlim(0, 1.05)
        ax.set_title("(%s) %s" % (TAGS[i], lab), fontsize=10, pad=6)
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.02, top=0.94, wspace=0.06, hspace=0.12)
    save(fig, "fig3_%s_3d" % cfg["problem"].lower(), pad_inches=0.08)


# ------------------------------------------------------------------ figure 4 --
def figure4():
    """IGD against the evaluation budget."""
    cfg = C["fig4"]
    conv = load("conv", binary=True)
    fig, axes = plt.subplots(len(cfg["problems"]), 1, figsize=cfg["size"])
    for i, name in enumerate(cfg["problems"]):
        ax = axes[i]
        for lab, key in C["algorithms"].items():
            pts = sorted(v for (n, a, _), v in conv.items() if n == name and a == key)
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    label=lab if i == 0 else None, **C["style"][lab])
        if cfg["logy"]: ax.set_yscale("log")
        ax.grid(True); ax.set_ylabel("IGD")
        tag(ax, i, name, loc="upper right")
    axes[-1].set_xlabel("Function evaluations")
    fig.legend(loc="upper center", bbox_to_anchor=(0.55, 1.005), ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    save(fig, "fig4_convergence")


# ------------------------------------------------------------------ figure 5 --
def figure5():
    """Sensitivity to the deferred-refinement threshold tau: the calibration heat map of the regret on all twelve
    problems (analyse_tau_calib.py writes results/tau_calib.png; the figure is copied under its manuscript name)."""
    import shutil
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "tau_calib.png")
    os.makedirs(C["out"]["dir"], exist_ok=True)
    dst = os.path.join(C["out"]["dir"], "fig5_tau_sensitivity.png")
    shutil.copyfile(src, dst); print("wrote", dst)


FIGURES = {1: figure1, 2: figure2, 3: figure3, 4: figure4, 5: figure5}

if __name__ == "__main__":
    wanted = [int(a) for a in sys.argv[1:]] or sorted(FIGURES)
    for n in wanted:
        FIGURES[n]()
