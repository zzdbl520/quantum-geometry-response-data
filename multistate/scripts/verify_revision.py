#!/usr/bin/env python3
"""Finite-dimensional checks and an audit of existing molecular tables. No molecular jobs."""
from pathlib import Path
import json, platform
import numpy as np
import pandas as pd
from scipy.linalg import solve_sylvester, polar
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"verification";OUT.mkdir(exist_ok=True)
rng=np.random.default_rng(20261016)
report={}
def herm(a):return (a+a.conj().T)/2
def roots(g):
    e,u=np.linalg.eigh(herm(g));mask=e>max(1e-12*e.max(),1e-14)
    w=(u[:,mask]/np.sqrt(e[mask]))@u[:,mask].conj().T
    sq=(u[:,mask]*np.sqrt(e[mask]))@u[:,mask].conj().T
    return w,sq,u[:,mask]@u[:,mask].conj().T,int(mask.sum())
b=rng.normal(size=(4,3))+1j*rng.normal(size=(4,3));b[:,2]=b[:,0]-2*b[:,1]
for name,gaps in [("positive",np.array([.7,1.2,1.8,2.3])),("signed",np.array([-.8,.6,1.7,2.2]))]:
    m1=b.conj().T@(b/gaps[:,None]);m2=b.conj().T@(b/gaps[:,None]**2)
    w,sq,pi,rank=roots(m2);d=w@m1@w
    u=np.linalg.qr(rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)))[0]
    w2,_,_,_=roots(u.conj().T@m2@u)
    d2=w2@(u.conj().T@m1@u)@w2
    report[name+"_support"]={"rank":rank,"congruence_relative_error":float(np.linalg.norm(m1-sq@d@sq)/np.linalg.norm(m1)),
        "covariance_relative_error":float(np.linalg.norm(d2-u.conj().T@d@u)/np.linalg.norm(d))}
    assert rank==2 and report[name+"_support"]["congruence_relative_error"]<1e-12
    assert report[name+"_support"]["covariance_relative_error"]<1e-12
# Exact projector and aligned Hamiltonian finite differences for a split cluster.
ep=np.array([-.08,.13]);eq=np.array([.8,1.25,1.9])
hp=np.diag(ep);hq=np.diag(eq)
b=.15*(rng.normal(size=(3,2))+1j*rng.normal(size=(3,2)))
mp=herm(.1*(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))))
mq=herm(.1*(rng.normal(size=(3,3))+1j*rng.normal(size=(3,3))))
h0=np.diag(np.r_[ep,eq]).astype(complex);mu=np.block([[mp,b.conj().T],[b,mq]])
x=solve_sylvester(hq,-hp,b);g=x.conj().T@x;a=b.conj().T@x+x.conj().T@b
ref=np.eye(5,dtype=complex)[:,:2]
def calc(f):
    en,ev=np.linalg.eigh(h0-f*mu);v=ev[:,:2];p=v@v.conj().T;k=polar(ref.conj().T@v)[0]
    return p,k@np.diag(en[:2])@k.conj().T
rows=[]
for f in [1e-3,5e-4,2e-4,1e-4]:
    pm,hm=calc(-f);pp,hh=calc(f)
    dp=(pp-pm)/(2*f);gf=.5*np.trace(dp@dp).real
    af=-(hh+hm-2*hp)/f**2
    rows.append({"F":f,"metric_relative_error":abs(gf-np.trace(g).real)/np.trace(g).real,
                 "A_relative_error":np.linalg.norm(af-a)/np.linalg.norm(a)})
pd.DataFrame(rows).to_csv(OUT/"split_cluster_convergence.csv",index=False)
report["split_cluster"]={"sylvester_residual":float(np.linalg.norm(hq@x-x@hp-b)),
 "best_metric_relative_error":float(min(v["metric_relative_error"] for v in rows)),
 "best_A_relative_error":float(min(v["A_relative_error"] for v in rows))}
assert report["split_cluster"]["best_metric_relative_error"]<1e-7
assert report["split_cluster"]["best_A_relative_error"]<1e-6
# Rank-deficient quasi-degenerate metric need not support the full A.
xc=np.array([[1.,1.]]);bc=np.array([[1.,.8]]);gc=xc.T@xc;ac=bc.T@xc+xc.T@bc
w,sq,pi,_=roots(gc);recovered=sq@(w@ac@w)@sq
report["split_support_counterexample"]={"A":ac.tolist(),"G":gc.tolist(),
 "unsupported_relative_norm":float(np.linalg.norm(ac-pi@ac@pi)/np.linalg.norm(ac)),
 "projected_congruence_error":float(np.linalg.norm(recovered-pi@ac@pi))}
assert report["split_support_counterexample"]["unsupported_relative_norm"]>.01
# Three-state cusp and dipole-dark zero response.
moment=np.array([[0,.1974399069446437,0],[.1974399069446437,0,0],[0,0,0.]])
kappa=np.ptp(np.linalg.eigvalsh(moment))/2
slopes=[-np.linalg.eigvalsh(-f*moment)[0]/abs(f) for f in [-1e-5,1e-5]]
report["three_state_cusp"]={"kappa":float(kappa),"max_slope_error":float(np.max(np.abs(np.array(slopes)-kappa)))}
report["dark_state"]={"response":0.0,"metric":0.0,"trace_ratio_defined":False}
# Existing reaction data: no new electronic-structure values.
p=pd.read_csv(ROOT/"evidence/HO2/manifold_pair_data.csv")
s=pd.read_csv(ROOT/"evidence/HO2/manifold_summary.csv")
gate=json.loads((ROOT/"evidence/HO2/manifold_response_gate.json").read_text())
err=np.max(np.abs(p.external_ground_alpha_au+p.internal_manifold_ground_alpha_au-p.reconstructed_ground_alpha_au))
peak=[]
for i in [5,9]:
    cen=p[p.source_index==i].reconstructed_ground_alpha_au
    neigh=p[p.source_index.isin([i-1,i+1])].reconstructed_ground_alpha_au
    row=s[s.source_index==i].iloc[0]
    peak.append({"source_index":i,"median":float(cen.median()),"lower":float(cen.min()),
     "upper":float(cen.max()),"margin":float(cen.min()-neigh.max()),
     "component_median_sum":float(p[p.source_index==i].external_ground_alpha_au.median()+p[p.source_index==i].internal_manifold_ground_alpha_au.median())})
report["reaction"]={"pair_count":len(p),"geometry_count":p.source_index.nunique(),"signed_records":2*len(p),
 "fields":sorted(p.field_au.unique().tolist()),"fieldwise_addition_max_error":float(err),
 "full_matrix_gate":gate["operator_extrapolation_gate"],"numerical_failure_count":len(gate["numerical_failures"]),
 "global_plateau_gate":gate["observable_plateau_gate"],"peak_checks":peak}
assert err<1e-10 and all(x["margin"]>0 for x in peak)
report["environment"]={"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__}
(OUT/"verification_report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))

