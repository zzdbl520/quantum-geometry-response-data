#!/usr/bin/env python3
"""Generate vector diagrams and plots from the supplied numerical tables."""
from pathlib import Path
from io import BytesIO
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
    for ext in ["pdf","png"]:
        target = O/(name+"."+ext)
        temporary = target.with_name(target.name+".tmp")
        buffer = BytesIO()
        fig.savefig(buffer,format=ext,dpi=200,bbox_inches="tight")
        data = buffer.getvalue()
        if ext == "pdf" and not data.rstrip().endswith(b"%%EOF"):
            raise RuntimeError(f"Incomplete PDF output: {target}")
        temporary.write_bytes(data)
        temporary.replace(target)
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

# Match the author's original publication figures: closed black axes,
# inward ticks, open markers, and the original component colours.
plt.rcParams.update({
    "font.family":"sans-serif", "font.sans-serif":["Arial","DejaVu Sans"],
    "mathtext.fontset":"stixsans", "font.size":8.6, "axes.labelsize":9.5,
    "axes.linewidth":1.0, "axes.edgecolor":"black", "text.color":"black",
    "axes.spines.top":True, "axes.spines.right":True,
    "xtick.direction":"in", "ytick.direction":"in",
    "xtick.major.size":4.2, "ytick.major.size":4.2,
    "xtick.major.width":.9, "ytick.major.width":.9,
    "legend.fontsize":7.5, "legend.frameon":False})
black="#171717";blue="#3976BA";red="#C74440";teal="#2A9186"
orange="#D58A17";purple="#7B61A8"
def style(ax,label=None,right=True):
    for spine in ax.spines.values():
        spine.set_visible(True);spine.set_color("black");spine.set_linewidth(1.)
    ax.tick_params(which="major",top=True,right=right,direction="in",colors="black")
    ax.minorticks_off()
    if label:ax.text(-.08,1.025,label,transform=ax.transAxes,ha="left",va="bottom",
                     fontsize=10.2,fontweight="bold",clip_on=False)

b=pd.read_csv(R/"evidence/B2/B2_stage2_combined_runs/stage2c_runs/B2_final_near_crossing_response.csv")
b=b.sort_values("R_Angstrom");x=b.R_Angstrom.to_numpy();rc=1.9441854605348607
fig,axs=plt.subplots(2,2,figsize=(7.35,6.05),layout="constrained")
a=axs[0,0]
a.plot(x,1000*b.stage1b_X_minus_A_gap_Eh,"o-",color=black,ms=3.8,mfc="white",lw=1.5)
a.axhline(0,color=".65",lw=.7)
a.set_ylabel(r"$E_X-E_A$ (m$E_{\rm h}$)")
a.text(.49,.08,r"$R_c\simeq1.944$ $\AA$",transform=a.transAxes,fontsize=8)
a=axs[0,1]
a.plot(x,b.mu_direct_perpendicular_au,"o-",color=black,ms=3.8,lw=1.5,label="Direct dipole")
a.plot(x,b.mu_relaxed_transport_perpendicular_au,"s-",color=teal,mfc="white",ms=3.8,lw=1.5,label="Aligned SA3 derivative")
a.set_ylabel("Transition magnitude (a.u.)");a.legend(loc="best",handlelength=2.)
for a,kind,lab in [(axs[1,0],"alpha",r"Hybrid polarizability, $\alpha_{\rm hyb}$ (a.u.)"),
                   (axs[1,1],"metric",r"Hybrid metric, $g_{\rm hyb}$ (a.u.)")]:
    for suffix,name,c,marker,ls,mfc in [("x",r"$\alpha_\perp$" if kind=="alpha" else r"$g_\perp$",blue,"o","-",blue),
                                     ("z",r"$\alpha_\parallel$" if kind=="alpha" else r"$g_\parallel$",orange,"s","--","white")]:
        y=pd.to_numeric(b[f"{kind}_{suffix}_final_au"],errors="coerce").to_numpy()
        for k,sel in enumerate([x<rc-1e-8,x>rc+1e-8]):
            a.plot(x[sel],y[sel],color=c,marker=marker,ls=ls,mfc=mfc,ms=3.8,lw=1.5,label=name if k==0 else None)
    a.set_ylabel(lab);a.legend(loc="upper left",handlelength=2.)
    if kind=="metric":a.set_yscale("log")
    else:a.set_ylim(bottom=0)
for i,a in enumerate(axs.ravel()):
    a.axvline(rc,color=purple,ls=(0,(3.,2.3)),lw=1.1,zorder=0)
    style(a,f"({chr(97+i)})")
    if i<2:a.tick_params(labelbottom=False)
    else:a.set_xlabel(r"B--B distance, $R$ ($\AA$)")
save(fig,"B2")

s=pd.read_csv(R/"evidence/HO2/manifold_summary.csv").sort_values("source_index")
def reaction(early,name,controls=False):
    x=early.source_index.to_numpy()/57
    y=early.plateau_reconstructed_ground_alpha_median_au.to_numpy()
    lo=early.plateau_reconstructed_ground_alpha_lower_au.to_numpy()
    hi=early.plateau_reconstructed_ground_alpha_upper_au.to_numpy()
    pcol="#6A3D9A";bcol="#1F78B4";ocol="#D95F02";gcol="#1B9E77"
    fig,axs=plt.subplots(1,1 if controls else 2,figsize=(5.8,2.8) if controls else (7.35,3.15),layout="constrained")
    axs=np.atleast_1d(axs);a=axs[0]
    a.fill_between(x,lo,hi,color=pcol,alpha=.15,lw=0)
    a.plot(x,y,"o-",color=pcol,mfc="white",mew=1.05,ms=4.2,lw=1.7,label="SA4 rank-two target")
    a.set(xlabel=r"Normalized source index, $\xi$",ylabel="Polarizability (a.u.)")
    if not controls:
        a.plot(x,early.sa2_ground_alpha_au,"--",color="#606060",lw=1.35,label="SA2 root")
        z=early.ss_ground_alpha_au.to_numpy()
        a.plot(x,z,"s-",color=gcol,mfc="white",mew=1.0,ms=4.0,lw=1.25,label="State-specific")
        for j,lab in [(5,"I"),(9,"II")]:
            k=np.flatnonzero(early.source_index.to_numpy()==j)[0]
            a.vlines(x[k],y[k],33.2,color=black,ls=(0,(3,2)),lw=.8,zorder=0)
            a.plot(x[k],y[k],"v",mfc="white",mec=black,ms=5.5,zorder=6)
            a.text(x[k]+.004,y[k]+1.0,"Peak "+lab,fontsize=8.1)
        a.legend(loc="lower right",fontsize=7.7,handlelength=2.)
        style(a,"(a)")
        a=axs[1]
        a.plot(x,early.plateau_external_ground_alpha_median_au,"s-",color=bcol,mfc="white",mew=1.05,ms=4.2,lw=1.7,label="Aligned background")
        a.plot(x,y,"--",color=pcol,lw=1.2,label="Total median")
        a.set(xlabel=r"Normalized source index, $\xi$",ylabel="Response (a.u.)")
        twin=a.twinx()
        twin.plot(x,early.plateau_internal_manifold_ground_alpha_median_au,"^-",color=ocol,mfc="white",mew=1.05,ms=4.4,lw=1.7,label=r"Internal mixing $I$")
        twin.set_ylabel("Internal mixing (a.u.)",color=ocol)
        style(a,"(b)",right=False);twin.minorticks_off()
        twin.tick_params(axis="y",direction="in",right=True,colors=ocol,width=.9,length=4.2)
        for side in ['left','top','bottom']:twin.spines[side].set_visible(False)
        twin.spines['right'].set_color('black');twin.spines['right'].set_linewidth(1.)
        handles=a.get_lines()+twin.get_lines()
        a.legend(handles,[h.get_label() for h in handles],loc="lower center",bbox_to_anchor=(.52,1.01),
                 ncol=2,fontsize=7.3,handlelength=1.6,handletextpad=.4,columnspacing=.8,borderaxespad=0.)
        for j in [5,9]:a.axvline(j/57,color=black,ls=(0,(3,2)),lw=.8,zorder=0)
    else:
        a.legend(loc="lower right",fontsize=8)
        a.annotate("Index 18: range test fails",xy=(18/57,y[-1]),xytext=(16.35/57,33.3),
                   arrowprops={"arrowstyle":"-","lw":.8,"color":black},fontsize=8.3,ha="center")
        a.set_ylim(26.8,34);style(a)
    save(fig,name)
reaction(s[s.source_index.between(2,11)],"HO2")
reaction(s[s.source_index.between(15,18)],"HO2_controls",True)
print("Wrote original-style B2, HO2, and HO2_controls from the current tables.")
