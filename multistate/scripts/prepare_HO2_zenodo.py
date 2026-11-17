#!/usr/bin/env python3
"""Prepare a read-only HO2 SA2/SA4/SS HDF5 wavefunction archive."""
from pathlib import Path
from collections import defaultdict
from decimal import Decimal
import argparse, csv, hashlib, json, tarfile
import h5py
import numpy as np

PROTOCOLS={
 "SA2":("HO2_qgeom_pilot_work","fields",2,21,252,126),
 "SA4":("HO2_qgeom_protocol_validation_work/protocol_validation/sa4","manifold_fields",4,21,148,74),
 "SS":("HO2_qgeom_protocol_validation_work/protocol_validation/ss","fields",1,21,136,68),
}
def sha256(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--root",type=Path,default=Path("/scr/zdb"))
 ap.add_argument("--out",type=Path,default=Path("/scr/zdb/HO2_release"))
 args=ap.parse_args()
 root,out=args.root.resolve(),args.out.resolve()
 selected=set(); records=[]; pairs=[]; stats={}
 for protocol,(sub,fieldname,nroots,nzero,nfield,npairs) in PROTOCOLS.items():
  base=root/sub
  zs=sorted((base/"zero").glob("*.h5"))
  fs=sorted((base/fieldname).glob("irc_*/*.h5"))
  if (len(zs),len(fs))!=(nzero,nfield):raise RuntimeError(f"{protocol} count mismatch: {len(zs)} {len(fs)}")
  pairmap=defaultdict(dict)
  for kind,files in (("zero",zs),("field",fs)):
   for p in files:
    if p.is_symlink():raise RuntimeError(f"Unexpected symlink: {p}")
    with h5py.File(p,"r") as h:
     for name,shape in (("ci_vectors",(nroots,36,84)),("mo_coeff",(115,115)),("e_states_Eh",(nroots,))):
      if name not in h or h[name].shape!=shape or not np.isfinite(h[name][()]).all():raise RuntimeError(f"Invalid {name}: {p}")
     if not bool(h.attrs.get("converged",False)):raise RuntimeError(f"Not converged: {p}")
     if kind=="field":
      signed=float(h.attrs["signed_field_au"])
      if not np.isfinite(signed) or signed==0:raise RuntimeError(f"Invalid field: {p}")
      irc=p.parent.name
      if not (base/"zero"/(irc+".h5")).is_file():raise RuntimeError(f"Missing zero anchor: {p}")
      key=(irc,str(Decimal(str(abs(signed))).normalize()))
      sign="plus" if signed>0 else "minus"
      if sign in pairmap[key]:raise RuntimeError(f"Duplicate task: {key} {sign}")
      pairmap[key][sign]=p.relative_to(root).as_posix()
    rel=p.relative_to(root).as_posix();selected.add(rel)
    records.append(dict(protocol=protocol,kind=kind,path=rel,bytes=p.stat().st_size,sha256=sha256(p)))
  if len(pairmap)!=npairs or any(set(x)!={"plus","minus"} for x in pairmap.values()):raise RuntimeError(f"Unpaired files in {protocol}")
  for (irc,mag),x in sorted(pairmap.items()):pairs.append(dict(protocol=protocol,irc=irc,field_magnitude_au=mag,plus_checkpoint=x["plus"],minus_checkpoint=x["minus"]))
  stats[protocol]=dict(zero=len(zs),field=len(fs),pairs=len(pairmap),field_irc_geometries=len({k[0] for k in pairmap}))
 if len({x["sha256"] for x in records})!=599:raise RuntimeError("Duplicate wavefunction SHA256")
 for sub in ("HO2_quantum_geometry/HO2_qgeom_pilot_v1p0","HO2_quantum_geometry/HO2_qgeom_protocol_validation_v1","HO2_qgeom_pilot_work","HO2_qgeom_protocol_validation_work"):
  for p in (root/sub).rglob("*"):
   if not p.is_file() or p.is_symlink() or any(s in {"__pycache__",".git","scratch","tmp"} for s in p.parts):continue
   if p.suffix.lower() in {".py",".json",".csv",".md",".txt",".yaml",".yml",".npz"}:selected.add(p.relative_to(root).as_posix())
 for name in ("manifest.csv","audit_report.json"):
  p=root/"HO2_wavefunction_audit"/name
  if p.is_file():selected.add(p.relative_to(root).as_posix())
 out.mkdir(parents=True,exist_ok=True)
 with (out/"wavefunction_manifest.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=["protocol","kind","path","bytes","sha256"]);w.writeheader();w.writerows(records)
 with (out/"pair_mapping.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=["protocol","irc","field_magnitude_au","plus_checkpoint","minus_checkpoint"]);w.writeheader();w.writerows(pairs)
 (out/"SHA256SUMS").write_text("".join(f"{sha256(root/p)}  {p}\n" for p in sorted(selected)))
 report=dict(status="PASS",counts=stats,unique_wavefunction_hashes=599,archived_source_files=len(selected),scope="Archive integrity and convergence flags, not an independent wavefunction-overlap or full response rerun")
 (out/"audit_report.json").write_text(json.dumps(report,indent=2)+"\n")
 (out/"README_ARCHIVE.md").write_text("HO2 raw CASSCF wavefunctions: SA2 21+252, SA4 21+148, SS 21+136 zero/field checkpoint files. 268 complete positive-negative field pairs. Verify with sha256sum -c SHA256SUMS. The original failed full-matrix extrapolation/global range gates remain limitations; not an independent algorithm validation.\n")
 archive=out/"HO2_wavefunctions_v1.tar.gz"
 with tarfile.open(archive,"w:gz") as t:
  for p in sorted(selected):t.add(root/p,arcname=p,recursive=False)
  for n in ("SHA256SUMS","wavefunction_manifest.csv","pair_mapping.csv","audit_report.json","README_ARCHIVE.md"):t.add(out/n,arcname=n,recursive=False)
 report.update(archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=sha256(archive))
 (out/"release_report.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps(report,indent=2))
if __name__=="__main__":main()
