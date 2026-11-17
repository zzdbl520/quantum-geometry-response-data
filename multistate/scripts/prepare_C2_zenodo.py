#!/usr/bin/env python3
"""Validate and archive C2 SA6 production checkpoints, read-only."""
from pathlib import Path
from collections import defaultdict
from decimal import Decimal
import argparse,csv,hashlib,json,tarfile
import h5py,numpy as np

def hashfile(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""): h.update(b)
 return h.hexdigest()

def run():
 a=argparse.ArgumentParser();a.add_argument("--root",type=Path,default=Path("."));a.add_argument("--out",type=Path,default=Path("C2_release"));a.add_argument("--include-controls",action="store_true");v=a.parse_args()
 root=v.root.resolve();out=v.out.resolve()
 zero=root/"C2_SA6_global_transport_results_v1";field=root/"C2_SA6_global_field_work_v2"
 zs=sorted((zero/"points").glob("R_*/point.h5"));fs=sorted((field/"fields").glob("R_*/*/*.h5"))
 if len(zs)!=119 or len(fs)!=888: raise RuntimeError(f"Count mismatch: {len(zs)} zero, {len(fs)} field")
 records=[];pairs=defaultdict(dict);zerors=set();fieldrs=set();selected=set()
 for typ,files in [("zero",zs),("field",fs)]:
  for p in files:
   if p.is_symlink():raise RuntimeError(f"Symlink {p}")
   with h5py.File(p,"r") as h:
    if h["ci_vectors"].shape!=(6,70,70) or h["mo_coeff"].shape!=(92,92) or h["e_states_Eh"].shape!=(6,): raise RuntimeError(f"Shape {p}")
    if not all(np.isfinite(h[k][()]).all() for k in ("ci_vectors","mo_coeff","e_states_Eh")):raise RuntimeError(f"Nonfinite {p}")
    if not bool(h.attrs.get("converged",False)):raise RuntimeError(f"Not converged {p}")
    r=str(Decimal(str(h.attrs["R_angstrom"])).normalize());axis=mag=sign=""
    if typ=="zero":
     if r in zerors:raise RuntimeError(f"Duplicate zero R {r}")
     zerors.add(r)
    else:
     axis=str(h.attrs["axis"]).lower();mag=str(Decimal(str(h.attrs["field_magnitude_au"])).normalize());sgn=int(h.attrs["field_sign"])
     if sgn not in (-1,1) or axis not in ("x","y","z"):raise RuntimeError(f"Invalid field {p}")
     sign="plus" if sgn==1 else "minus";key=(r,axis,mag)
     if sign in pairs[key]:raise RuntimeError(f"Duplicate {key} {sign}")
     pairs[key][sign]=p.relative_to(root).as_posix();fieldrs.add(r)
   rel=p.relative_to(root).as_posix();selected.add(rel)
   records.append(dict(kind=typ,R_Angstrom=r,axis=axis,field_magnitude_au=mag,sign=sign,path=rel,bytes=p.stat().st_size,sha256=hashfile(p)))
 if len(pairs)!=444 or len(fieldrs)!=54 or any(set(d)!={"plus","minus"} for d in pairs.values()) or fieldrs-zerors:raise RuntimeError("Missing or invalid pairs / geometry anchors")
 if len(set(x["sha256"] for x in records))!=1007:raise RuntimeError("Duplicate checkpoint hashes")
 for d in (zero,field):
  for p in d.rglob("*"):
   if p.is_file() and not p.is_symlink() and p.suffix.lower() in (".csv",".json",".txt",".py",".md") and not any(x in p.parts for x in ("logs","task_logs")):selected.add(p.relative_to(root).as_posix())
 for p in root.iterdir():
  if p.is_file() and p.suffix.lower() in (".py",".json",".md",".yml",".yaml"):selected.add(p.relative_to(root).as_posix())
 controls={}
 if v.include_controls:
  for label in ("C2_SA6_local_transport_results_v1","C2_SA6_crossing_field_work","C2_SA6_z_closure_R_2p9","C2_SA6_z_closure_R_3p2"):
   f=[p for p in (root/label).rglob("*") if p.is_file() and not p.is_symlink()];controls[label]=len(f);selected.update(p.relative_to(root).as_posix() for p in f)
 out.mkdir(parents=True,exist_ok=True)
 with (out/"wavefunction_manifest.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 with (out/"pair_mapping.csv").open("w",newline="") as f:
  w=csv.writer(f);w.writerow(["R_Angstrom","axis","field_magnitude_au","plus_checkpoint","minus_checkpoint"])
  for k,d in sorted(pairs.items(),key=lambda x:(Decimal(x[0][0]),x[0][1],Decimal(x[0][2]))):w.writerow([*k,d["plus"],d["minus"]])
 (out/"SHA256SUMS").write_text("".join(f"{hashfile(root/p)}  {p}\n" for p in sorted(selected)))
 report=dict(zero_checkpoints=119,field_checkpoints=888,complete_pairs=444,field_geometries=54,source_files=len(selected),controls=controls,scope="Checkpoint inventory and integrity audit; not a full independent overlap or CASSCF rerun")
 (out/"audit_report.json").write_text(json.dumps(report,indent=2)+"\n")
 (out/"README_ARCHIVE.md").write_text("C2 SA6-CASSCF: 119 zero and 888 field wavefunction files; 444 complete positive/negative pairs. Verify after extraction with sha256sum -c SHA256SUMS. The archive does not establish independent wavefunction overlap recomputation or clean-environment CASSCF reruns.\n")
 name="C2_SA6_wavefunctions_with_controls_v1.tar.gz" if v.include_controls else "C2_SA6_wavefunctions_v1.tar.gz";archive=out/name
 with tarfile.open(archive,"w:gz") as t:
  for rel in sorted(selected):t.add(root/rel,arcname=rel,recursive=False)
  for n in ("SHA256SUMS","wavefunction_manifest.csv","pair_mapping.csv","audit_report.json","README_ARCHIVE.md"):t.add(out/n,arcname=n,recursive=False)
 report.update(archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=hashfile(archive))
 (out/"release_report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":run()
