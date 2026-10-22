#!/usr/bin/env python3
"""Generate vector diagrams and plots from the supplied numerical tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
R=Path(__file__).resolve().parents[1];O=R/"figures";O.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.labelsize":9,
 "legend.fontsize":7.5,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none",
 "axes.spines.top":False,"axes.spines.right":False,"savefig.facecolor":"white"})
blue="#28699A";red="#B64A48";green="#337E62";purple="#715598";orange="#B67725"
def save(fig,name):
    for ext in ["pdf","png"]:fig.savefig(O/(name+"."+ext),dpi=200,bbox_inches="tight")
    plt.close(fig)
def panel(ax,s):ax.text(-.12,1.04,s,transform=ax.transAxes,fontweight="bold")
fig,ax=plt.subplots(figsize=(7.2,2.7));ax.set(xlim=(0,10),ylim=(0,4));ax.axis("off")
cards=[(.05,blue,"Internal splitting",r"$\hat{P}\hat{\mu}\hat{P}$","At exact degeneracy",r"$\kappa=(d_{\max}-d_{\min})/2$","Nonzero width: linear cusp"),
       (3.38,purple,"Subspace deformation",r"$\hat{X}=\hat{Q}(\partial_F\hat{P})\hat{P}$","For an isolated cluster",r"$\hat{H}_Q\hat{X}-\hat{X}\hat{H}_P=\hat{B}$","Metric: "+r"$g=\mathrm{Tr}(\hat{X}^\dagger\hat{X})$"),
       (6.71,green,"Numerical continuity",r"$S_{0F}=\Psi_0^\dagger\Psi_F$","For the computed branch","Singular values\n+ field refinement","Separate model and solver effects")]
for x,c,title,math,cond,eq,out in cards:
    ax.add_patch(FancyBboxPatch((x,.35),3.18,3.3,boxstyle="round,pad=0.02,rounding_size=0.08",fc="#FAFBFC",ec=c,lw=1.2))
    ax.text(x+1.59,3.30,title,ha="center",va="center",color=c,fontweight="bold",fontsize=10)
    ax.text(x+1.59,2.65,math,ha="center",fontsize=12)
    ax.text(x+1.59,2.02,cond,ha="center",fontsize=8)
    ax.text(x+1.59,1.40,eq,ha="center",va="center",fontsize=8.5)
    ax.text(x+1.59,.72,out,ha="center",fontsize=7.4)
save(fig,"scheme")
if "--scheme-only" in __import__("sys").argv:
    print("Updated operator notation in scheme.")
    raise SystemExit(0)
b=pd.read_csv(R/"evidence/B2/B2_stage2_combined_runs/stage2c_runs/B2_final_near_crossing_response.csv")
b=b.sort_values("R_Angstrom");x=b.R_Angstrom.to_numpy();rc=1.9441854605348607
fig,axs=plt.subplots(2,2,figsize=(7.1,5.3),layout="constrained")
a=axs[0,0];a.plot(x,1000*b.stage1b_X_minus_A_gap_Eh,"o-",color=blue,ms=3);a.axhline(0,color=".6",lw=.7);a.set_ylabel(r"$E_X-E_A$ (m$E_{\rm h}$)");panel(a,"(a)")
a=axs[0,1];a.plot(x,b.mu_direct_perpendicular_au,"o-",color=blue,ms=3,label="Direct dipole")
a.plot(x,b.mu_relaxed_transport_perpendicular_au,"s-",color=orange,ms=3,label="Aligned SA3 derivative")
a.set_ylabel("Magnitude (a.u.)");a.legend(loc="best");panel(a,"(b)")
for a,kind,lab,p in [(axs[1,0],"alpha",r"$\alpha_{\rm hyb}$ (a.u.)","(c)"),(axs[1,1],"metric",r"$g_{\rm hyb}$ (a.u.)","(d)")]:
    for suffix,name,c in [("x","Transverse",blue),("z","Longitudinal",orange)]:
        y=pd.to_numeric(b[f"{kind}_{suffix}_final_au"],errors="coerce").to_numpy()
        for k,sel in enumerate([x<rc-1e-8,x>rc+1e-8]):
            a.plot(x[sel],y[sel],"o-",color=c,ms=3,label=name if k==0 else None)
    a.set(yscale="log",ylabel=lab);a.legend();panel(a,p)
for a in axs.ravel():
    a.axvline(rc,color=purple,ls="--",lw=.8);a.set_xlabel(r"$R$ ($\AA$)");a.tick_params(direction="in")
save(fig,"B2")
s=pd.read_csv(R/"evidence/HO2/manifold_summary.csv").sort_values("source_index")
def reaction(early,name,controls=False):
    x=early.source_index.to_numpy()/57
    y=early.plateau_reconstructed_ground_alpha_median_au.to_numpy()
    lo=early.plateau_reconstructed_ground_alpha_lower_au.to_numpy()
    hi=early.plateau_reconstructed_ground_alpha_upper_au.to_numpy()
    fig,axs=plt.subplots(1,2 if not controls else 1,figsize=(7.1,2.85) if not controls else (5.8,2.7),layout="constrained")
    axs=np.atleast_1d(axs);a=axs[0]
    a.fill_between(x,lo,hi,color=purple,alpha=.18,label="Four-field range")
    a.plot(x,y,"o-",color=purple,ms=3,label="SA4 rank-two reconstruction")
    if not controls:
        for key,label,c in [("sa2_ground_alpha_au","SA2 root","#777777"),("ss_ground_alpha_au","State-specific",green)]:
            z=early[key].to_numpy()
            a.plot(x,z,"s-",color=c,ms=3,label=label)
    a.set(xlabel="Normalized source index",ylabel=r"$\alpha_{\parallel}$ (a.u.)")
    a.legend(loc="best",fontsize=7)
    if not controls:
        panel(a,"(a)");a=axs[1]
        a.plot(x,early.plateau_external_ground_alpha_median_au,"o-",color=blue,ms=3,label="Aligned background")
        a.plot(x,early.plateau_internal_manifold_ground_alpha_median_au,"s-",color=orange,ms=3,label="Internal mixing")
        a.plot(x,y,"-",color=purple,lw=1,label="Total median")
        a.set(xlabel="Normalized source index",ylabel="Response contribution (a.u.)");a.legend(loc="center right",fontsize=7);panel(a,"(b)")
        for a in axs:
            for j,l in [(5,"I"),(9,"II")]:
                xx=j/57;a.axvline(xx,color=".4",ls="--",lw=.7)
                a.text(xx,.98,l,transform=a.get_xaxis_transform(),ha="center",va="top",fontsize=8)
    else:
        a.annotate("Index 18: range test fails",xy=(18/57,y[-1]),xytext=(16/57,33.3),arrowprops={"arrowstyle":"->","lw":.7},fontsize=8)
        a.set_ylim(26.8,34)
    save(fig,name)
reaction(s[s.source_index.between(2,11)],"HO2")
reaction(s[s.source_index.between(15,18)],"HO2_controls",True)
print("Wrote scheme, B2, HO2 and HO2_controls.")

