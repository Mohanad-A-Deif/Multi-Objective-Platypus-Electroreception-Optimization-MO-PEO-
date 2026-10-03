import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
 "font.family":"serif","font.serif":["DejaVu Serif"],"font.size":11,
 "axes.edgecolor":"black","axes.linewidth":0.8,"axes.labelsize":11,"axes.titlesize":11,
 "xtick.direction":"in","ytick.direction":"in","xtick.major.size":4,"ytick.major.size":4,
 "xtick.labelsize":9,"ytick.labelsize":9,"legend.fontsize":9,"legend.frameon":False,
 "grid.alpha":0.25,"grid.linestyle":":","grid.linewidth":0.6,"grid.color":"black",
 "figure.dpi":600,"savefig.dpi":600,"figure.facecolor":"white","axes.facecolor":"white",
 "lines.linewidth":1.0,"mathtext.fontset":"dejavuserif"})
# grayscale, distinguished by marker/linestyle only
STY={"MO-PEO":dict(color="black",marker="o",ls="-",ms=3.0,mfc="none",mew=0.7),
     "NSGA-II":dict(color="0.45",marker="s",ls="--",ms=3.0,mfc="none",mew=0.7),
     "MOEA/D":dict(color="0.65",marker="^",ls=":",ms=3.0,mfc="none",mew=0.7)}
KEY={"MO-PEO":"MO-PEO-final","NSGA-II":"NSGA-II","MOEA/D":"MOEA/D"}
